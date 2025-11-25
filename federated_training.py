from __future__ import annotations

import os
import requests
import logging
import json
import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import mlflow
import mlflow.tensorflow

from src.model import DiabetesModel
from src.robust_mlflow_client import RobustMLflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFFFederatedLearningOrchestrator:
    """
    Orchestrates federated learning using TensorFlow Federated (TFF).
    """

    def __init__(
        self,
        data_dir: str = "./federated_data",
        n_clients: int | None = None,
        batch_size: int = 32,
        shuffle_buffer: int = 512,
        seed: int = 42,
        client_learning_rate: float = 1e-3,
        server_learning_rate: float = 1.0,
    ) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle_buffer = max(shuffle_buffer, batch_size)
        self.seed = seed
        self.client_learning_rate = client_learning_rate
        self.server_learning_rate = server_learning_rate

        self.client_ids = self._discover_clients(n_clients)
        if not self.client_ids:
            raise ValueError(f"No client folders found in {data_dir}")

        (
            self.train_datasets,
            self.client_validation_arrays,
            self.global_val_features,
            self.global_val_labels,
        ) = self._build_datasets()

        any_dataset = next(iter(self.train_datasets.values()))
        self.element_spec = any_dataset.element_spec
        self.input_dim = self.element_spec[0].shape[-1]

        # TFF 0.87.0 uses TensorFlow Federated optimizers, not Keras optimizers
        print(f"🔧 Using TensorFlow Federated {tff.__version__}")
        
        # Use TFF optimizer factories for compatibility
        self.training_process = tff.learning.algorithms.build_weighted_fed_avg(
            model_fn=self._model_fn,
            client_optimizer_fn=tff.learning.optimizers.build_adam(
                learning_rate=self.client_learning_rate
            ),
            server_optimizer_fn=tff.learning.optimizers.build_sgdm(
                learning_rate=self.server_learning_rate, momentum=0.9
            ),
        )

        print(
            f"\n✓ TFF orchestrator ready with {len(self.client_ids)} clients "
            f"from '{self.data_dir}'"
        )

    def _discover_clients(self, n_clients: int | None) -> List[str]:
        candidates = sorted(
            entry
            for entry in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, entry))
        )
        if n_clients is not None:
            candidates = candidates[:n_clients]
        return candidates

    def _load_split(self, client_id: str, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        csv_path = os.path.join(self.data_dir, client_id, filename)
        df = pd.read_csv(csv_path)
        features = df.drop(columns=["Diabetes_binary"]).to_numpy(dtype=np.float32)
        labels = df["Diabetes_binary"].to_numpy(dtype=np.float32)
        return features, labels

    def _build_tf_dataset(
        self, features: np.ndarray, labels: np.ndarray
    ) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        shuffle_size = min(self.shuffle_buffer, len(labels))
        if shuffle_size > 1:
            dataset = dataset.shuffle(
                buffer_size=shuffle_size,
                seed=self.seed,
                reshuffle_each_iteration=True,
            )
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def _build_datasets(
        self,
    ) -> Tuple[
        Dict[str, tf.data.Dataset],
        Dict[str, Tuple[np.ndarray, np.ndarray]],
        np.ndarray,
        np.ndarray,
    ]:
        train_datasets: Dict[str, tf.data.Dataset] = {}
        val_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        global_val_features: List[np.ndarray] = []
        global_val_labels: List[np.ndarray] = []

        for client_id in self.client_ids:
            train_features, train_labels = self._load_split(client_id, "train_data.csv")
            val_features, val_labels = self._load_split(client_id, "val_data.csv")

            train_datasets[client_id] = self._build_tf_dataset(
                train_features, train_labels
            )

            val_arrays[client_id] = (val_features, val_labels)
            global_val_features.append(val_features)
            global_val_labels.append(val_labels)

            print(
                f"{client_id}: {len(train_labels)} train samples | "
                f"{len(val_labels)} val samples"
            )

        all_val_features = (
            np.concatenate(global_val_features, axis=0)
            if global_val_features
            else np.array([], dtype=np.float32)
        )
        all_val_labels = (
            np.concatenate(global_val_labels, axis=0)
            if global_val_labels
            else np.array([], dtype=np.float32)
        )

        return train_datasets, val_arrays, all_val_features, all_val_labels

    def _model_fn(self) -> tff.learning.models.VariableModel:
        keras_model = DiabetesModel.create_model(input_dim=self.input_dim, compile_model=False)
        return tff.learning.models.from_keras_model(
            keras_model=keras_model,
            input_spec=self.element_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

    def _select_clients(
        self, round_num: int, clients_per_round: int | None
    ) -> Sequence[str]:
        if clients_per_round is None or clients_per_round >= len(self.client_ids):
            return self.client_ids

        rng = np.random.default_rng(self.seed + round_num)
        return rng.choice(self.client_ids, size=clients_per_round, replace=False)

    def _materialize_keras_model(self, state: Any) -> tf.keras.Model:
        keras_model = DiabetesModel.create_model(input_dim=self.input_dim)
        model_weights = self.training_process.get_model_weights(state)
        model_weights.assign_weights_to(keras_model)
        return keras_model

    def _evaluate_global_validation(self, keras_model: tf.keras.Model) -> Dict[str, float]:
        if self.global_val_features.size == 0:
            return {}

        loss, accuracy, auc = keras_model.evaluate(
            self.global_val_features,
            self.global_val_labels,
            verbose=0,
        )
        return {"loss": float(loss), "accuracy": float(accuracy), "auc": float(auc)}

    def run_federated_training(
        self,
        n_rounds: int = 10,
        clients_per_round: int | None = None,
    ) -> tf.keras.Model:
        """Run federated training with MLflow tracking"""
        
        print("\n" + "=" * 70)
        print("🚀 STARTING TFF FEDERATED LEARNING")
        print("=" * 70)

        # Initialize robust MLflow client - using local MLflow server
        # Use environment variable from Jenkins, fallback to localhost for local runs
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        print(f"MLflow Tracking URI: {mlflow_uri}")
        
        # Set MLflow tracking URI explicitly
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Validate MLflow URI format
        if not mlflow_uri.startswith(('http://', 'https://', 'file://')):
            print(f"⚠️ Warning: MLflow URI format may be invalid: {mlflow_uri}")
        
        # Test basic connectivity before creating client
        print("🔍 Testing MLflow server connectivity...")
        try:
            import requests
            if mlflow_uri.startswith('http'):
                response = requests.get(f"{mlflow_uri}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ MLflow server is responding")
                else:
                    print(f"⚠️ MLflow server returned status {response.status_code}")
            else:
                print("ℹ️ Non-HTTP URI detected, skipping connectivity test")
        except Exception as e:
            print(f"⚠️ MLflow server connectivity test failed: {e}")
            print("   This may indicate MLflow server is not running or accessible")
        
        # Create robust MLflow client with minimal retries
        mlflow_client = RobustMLflowClient(
            tracking_uri=mlflow_uri,
            max_retries=1,        # Reduced to 1 retry for faster failure detection
            retry_delay=2.0       # Reduced delay to 2.0 seconds
        )
        
        # Setup experiment
        experiment_name = "diabetes-federated-learning"
        mlflow_client.setup_experiment(experiment_name)
        
        # Prepare fallback paths
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_dir = f"fallback_metrics/{timestamp}"
        os.makedirs(fallback_dir, exist_ok=True)
        
        # Start MLflow run and training
        with mlflow_client.start_run(f"fed_training_{n_rounds}_rounds") as run:
            
            # Prepare parameters for logging
            params = {
                "n_rounds": n_rounds,
                "n_clients": len(self.client_ids),
                "batch_size": self.batch_size,
                "client_learning_rate": self.client_learning_rate,
                "server_learning_rate": self.server_learning_rate,
                "shuffle_buffer": self.shuffle_buffer,
                "seed": self.seed,
            }
            
            # Log parameters to MLflow or save as fallback
            if mlflow_client.is_enabled():
                success = mlflow_client.log_params(params)
                if not success:
                    print("⚠️ MLflow params logging failed, saving to JSON fallback")
                    self.save_metrics_json({"parameters": params}, f"{fallback_dir}/parameters.json")
            else:
                print("⚠️ MLflow disabled, saving parameters to JSON fallback")
                self.save_metrics_json({"parameters": params}, f"{fallback_dir}/parameters.json")
            
            # Initialize final_model variable
            final_model = None
            
            # Initialize training
            state = self.training_process.initialize()
            
            # Training loop
            for round_num in range(1, n_rounds + 1):
                participating_clients = self._select_clients(round_num, clients_per_round)
                round_data = [self.train_datasets[cid] for cid in participating_clients]

                result = self.training_process.next(state, round_data)
                state = result.state

                keras_model = self._materialize_keras_model(state)
                val_metrics = self._evaluate_global_validation(keras_model)

                # Log metrics per round to MLflow or save as fallback
                if val_metrics:
                    round_metrics = {
                        "val_loss": val_metrics['loss'],
                        "val_accuracy": val_metrics['accuracy'],
                        "val_auc": val_metrics['auc'],
                        "round": round_num
                    }
                    
                    if mlflow_client.is_enabled():
                        success = mlflow_client.log_metrics({
                            "val_loss": val_metrics['loss'],
                            "val_accuracy": val_metrics['accuracy'],
                            "val_auc": val_metrics['auc'],
                        }, step=round_num)
                        
                        if not success:
                            print(f"⚠️ MLflow metrics logging failed for round {round_num}, saving to JSON fallback")
                            self.save_metrics_json(round_metrics, f"{fallback_dir}/round_{round_num:02d}_metrics.json")
                    else:
                        self.save_metrics_json(round_metrics, f"{fallback_dir}/round_{round_num:02d}_metrics.json")
                
                metrics_str = (
                    f"val_loss={val_metrics.get('loss', float('nan')):.4f}, "
                    f"val_accuracy={val_metrics.get('accuracy', float('nan')):.4f}, "
                    f"val_auc={val_metrics.get('auc', float('nan')):.4f}"
                    if val_metrics
                    else "validation data unavailable"
                )

                print(
                    f"Round {round_num:02d}/{n_rounds}: "
                    f"{len(participating_clients)} clients → {metrics_str}"
                )
            
                metrics_str = (
                    f"val_loss={val_metrics.get('loss', float('nan')):.4f}, "
                    f"val_accuracy={val_metrics.get('accuracy', float('nan')):.4f}, "
                    f"val_auc={val_metrics.get('auc', float('nan')):.4f}"
                    if val_metrics
                    else "validation data unavailable"
                )

                print(
                    f"Round {round_num:02d}/{n_rounds}: "
                    f"{len(participating_clients)} clients → {metrics_str}"
                )
            
            # Get final model
            final_model = self._materialize_keras_model(state)
            
            # Evaluate on all clients
            print("\n" + "=" * 70)
            print("📊 EVALUATING FINAL MODEL ON ALL CLIENTS")
            print("=" * 70)
            client_metrics = self.evaluate_clients(final_model)
            
            # Calculate average metrics
            avg_loss = sum(m["loss"] for m in client_metrics.values()) / len(client_metrics)
            avg_accuracy = sum(m["accuracy"] for m in client_metrics.values()) / len(client_metrics)
            avg_auc = sum(m["auc"] for m in client_metrics.values()) / len(client_metrics)
            
            # Prepare final metrics
            final_metrics = {
                "final_avg_loss": avg_loss,
                "final_avg_accuracy": avg_accuracy,
                "final_avg_auc": avg_auc,
            }
            
            # Add per-client metrics
            for client_id, metrics in client_metrics.items():
                final_metrics.update({
                    f"{client_id}_final_loss": metrics["loss"],
                    f"{client_id}_final_accuracy": metrics["accuracy"],
                    f"{client_id}_final_auc": metrics["auc"],
                })
            
            # Log final metrics to MLflow or save as fallback
            if mlflow_client.is_enabled():
                # Log averaged metrics
                avg_success = mlflow_client.log_metrics({
                    "final_avg_loss": avg_loss,
                    "final_avg_accuracy": avg_accuracy,
                    "final_avg_auc": avg_auc,
                })
                
                # Log per-client metrics
                client_success = True
                for client_id, metrics in client_metrics.items():
                    success = mlflow_client.log_metrics({
                        f"{client_id}_final_loss": metrics["loss"],
                        f"{client_id}_final_accuracy": metrics["accuracy"],
                        f"{client_id}_final_auc": metrics["auc"],
                    })
                    client_success = client_success and success
                
                # Save fallback if any logging failed
                if not (avg_success and client_success):
                    print("⚠️ MLflow final metrics logging failed, saving to JSON fallback")
                    self.save_metrics_json(final_metrics, f"{fallback_dir}/final_metrics.json")
            else:
                print("⚠️ MLflow disabled, saving final metrics to JSON fallback")
                self.save_metrics_json(final_metrics, f"{fallback_dir}/final_metrics.json")
            
            # Save model locally first
            model_path = "models/tff_federated_diabetes_model.h5"
            self.save_model(final_model, model_path)
            
            # Log model to MLflow or create fallback info
            print("\n📦 Logging model to MLflow...")
            model_logged = False
            
            if mlflow_client.is_enabled():
                model_success = mlflow_client.log_model(
                    final_model,
                    artifact_path="model",
                    registered_model_name="diabetes-federated-model"
                )
                
                artifact_success = mlflow_client.log_artifact(model_path, artifact_path="model_files")
                
                model_logged = model_success and artifact_success
                
                if not model_logged:
                    print("⚠️ MLflow model logging failed, creating fallback info")
            
            # Create model fallback info if MLflow failed or disabled
            if not model_logged:
                model_info = {
                    "model_path": os.path.abspath(model_path),
                    "model_type": "federated_learning",
                    "framework": "tensorflow_federated",
                    "algorithm": "FedAvg",
                    "task": "binary_classification",
                    "dataset": "BRFSS_2015_diabetes",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "final_metrics": final_metrics
                }
                self.save_metrics_json(model_info, f"{fallback_dir}/model_info.json")
                print(f"✅ Model info saved to fallback: {fallback_dir}/model_info.json")
            
            # Log tags
            mlflow_client.set_tags({
                "model_type": "federated_learning",
                "framework": "tensorflow_federated",
                "algorithm": "FedAvg",
                "task": "binary_classification",
                "dataset": "BRFSS_2015_diabetes",
            })
            
            print("\n" + "=" * 70)
            print("✅ FEDERATED LEARNING COMPLETE")
            print("=" * 70)
            print(f"Final Avg Accuracy: {avg_accuracy:.4f}")
            print(f"Final Avg AUC: {avg_auc:.4f}")
            print(f"Final Avg Loss: {avg_loss:.4f}")
            
            if mlflow_client.is_enabled():
                print(f"MLflow Run ID: {mlflow_client.get_run_id()}")
            else:
                print(f"📁 Fallback metrics saved to: {fallback_dir}")
                print("   Use validate_mlflow_model.py and download_mlflow_model.py")
                print("   with --fallback-dir option to access results")
            
            print("=" * 70)
        
        return final_model

    def evaluate_clients(self, keras_model: tf.keras.Model) -> Dict[str, Dict[str, float]]:
        """Evaluate model on each client's validation set"""
        per_client_metrics: Dict[str, Dict[str, float]] = {}
        for client_id, (features, labels) in self.client_validation_arrays.items():
            loss, acc, auc = keras_model.evaluate(features, labels, verbose=0)
            per_client_metrics[client_id] = {
                "loss": float(loss),
                "accuracy": float(acc),
                "auc": float(auc),
            }
            print(
                f"{client_id} → loss={loss:.4f}, "
                f"accuracy={acc:.4f}, auc={auc:.4f}"
            )
        return per_client_metrics

    @staticmethod
    def save_model(model: tf.keras.Model, output_path: str) -> None:
        """Save model to disk"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model.save(output_path)
        print(f"✓ Saved model to '{output_path}'")
    
    @staticmethod
    def save_metrics_json(metrics: Dict[str, Any], output_path: str) -> None:
        """Save metrics to JSON file as fallback when MLflow fails"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add timestamp and metadata
        metrics_with_metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "mlflow_fallback": True,
            "metrics": metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_with_metadata, f, indent=2, default=str)
        
        print(f"✅ Metrics saved to JSON fallback: {output_path}")
    
    @staticmethod
    def load_metrics_json(json_path: str) -> Dict[str, Any]:
        """Load metrics from JSON fallback file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data.get('metrics', {})
        except Exception as e:
            print(f"❌ Failed to load metrics from {json_path}: {e}")
            return {}


if __name__ == "__main__":
    # Configuration
    orchestrator = TFFFederatedLearningOrchestrator(
        data_dir="./federated_data",
        n_clients=3,
        batch_size=32,
        shuffle_buffer=1024,
        seed=42,
        client_learning_rate=1e-3,
        server_learning_rate=1.0,
    )

    # Run federated training
    final_model = orchestrator.run_federated_training(
        n_rounds=10,
        clients_per_round=None,  # Use all clients
    )
    
    print("\n✅ Training complete! Model saved and logged to MLflow.")