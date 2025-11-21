from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

from src.model import DiabetesModel

# Add to federated_training.py

import mlflow
import mlflow.tensorflow


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

        self.training_process = tff.learning.algorithms.build_weighted_fed_avg(
            model_fn=self._model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam(
                learning_rate=self.client_learning_rate
            ),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(
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
        print("\n" + "=" * 70)
        print("🚀 STARTING TFF FEDERATED LEARNING")
        print("=" * 70)

        # At the start of run_federated_training():
        mlflow.set_tracking_uri("http://localhost:5000")  # Or your MLflow server
        mlflow.set_experiment("diabetes-federated-learning")

        with mlflow.start_run(run_name=f"fed_training_{n_rounds}_rounds"):
            # Log parameters
            mlflow.log_params({
                "n_rounds": n_rounds,
                "n_clients": len(self.client_ids),
                "batch_size": self.batch_size,
                "client_lr": self.client_learning_rate,
                "server_lr": self.server_learning_rate,
            })
            
        state = self.training_process.initialize()
        for round_num in range(1, n_rounds + 1):
            participating_clients = self._select_clients(round_num, clients_per_round)
            round_data = [self.train_datasets[cid] for cid in participating_clients]

            result = self.training_process.next(state, round_data)
            state = result.state

            keras_model = self._materialize_keras_model(state)
            val_metrics = self._evaluate_global_validation(keras_model)

            metrics_str = (
                f"val_loss={val_metrics.get('loss', float('nan')):.4f}, "
                f"val_accuracy={val_metrics.get('accuracy', float('nan')):.4f}, "
                f"val_auc={val_metrics.get('auc', float('nan')):.4f}"
                if val_metrics
                else "validation data unavailable"
            )

            # Log metrics per round
            mlflow.log_metrics({
                    "val_loss": val_metrics.get('loss', 0),
                    "val_accuracy": val_metrics.get('accuracy', 0),
                    "val_auc": val_metrics.get('auc', 0),
                }, step=round_num)
            

            print(
                f"Round {round_num:02d}/{n_rounds}: "
                f"{len(participating_clients)} clients → {metrics_str}"
            )

        # Log final model
        mlflow.tensorflow.log_model(
            keras_model,
            "model",
            registered_model_name="diabetes-federated-model"
        )
        
        # Log artifacts
        mlflow.log_artifact("models/tff_federated_diabetes_model.h5")

        print("\n" + "=" * 70)
        print("✅ FEDERATED LEARNING COMPLETE")
        print("=" * 70)

        

        return self._materialize_keras_model(state)

    def evaluate_clients(self, keras_model: tf.keras.Model) -> Dict[str, Dict[str, float]]:
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model.save(output_path)
        print(f"\n✓ Saved TFF-trained model to '{output_path}'")


if __name__ == "__main__":
    orchestrator = TFFFederatedLearningOrchestrator(
        data_dir="./federated_data",
        n_clients=3,
        batch_size=32,
        shuffle_buffer=1024,
        seed=8,
    )

    final_model = orchestrator.run_federated_training(
        n_rounds=10,
        clients_per_round=None,
    )

    orchestrator.evaluate_clients(final_model)
    orchestrator.save_model(final_model, "models/tff_federated_diabetes_model.h5")