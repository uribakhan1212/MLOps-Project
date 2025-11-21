# src/inference_server.py
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any

from flask import Flask, request, jsonify, current_app, g
import tensorflow as tf
import numpy as np
from prometheus_flask_exporter import PrometheusMetrics

# --- Configuration defaults (can be overridden via env vars) ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/tff_federated_diabetes_model.h5")
LOG_FILE = os.getenv("LOG_FILE", "/var/log/inference_server.log")
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 1_000_000))  # ~1MB default
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", 0.5))

# Feature names (keep as you had)
FEATURE_NAMES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity',
    'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
    'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',
    'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

# --- Model server helper (kept simple) ---
class ModelServer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.feature_names = FEATURE_NAMES

    def load(self):
        if self.model is None:
            # load the model once
            current_app.logger.info(f"Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            current_app.logger.info("Model loaded successfully")

    def preprocess_single(self, data: Dict[str, Any]) -> np.ndarray:
        # ensure deterministic ordering of features
        features = [data.get(feat, 0) for feat in self.feature_names]
        return np.array(features, dtype=np.float32)

    def preprocess_batch(self, records: List[Dict[str, Any]]) -> np.ndarray:
        arr = [self.preprocess_single(rec) for rec in records]
        return np.stack(arr, axis=0).astype(np.float32)

    def predict_batch(self, features_batch: np.ndarray) -> List[Dict[str, Any]]:
        # model.predict on batch (fast)
        preds = self.model.predict(features_batch, verbose=0)
        # handle shape (N,1) or (N,) 
        preds = np.asarray(preds).reshape(-1)
        results = []
        for p in preds:
            risk = "HIGH" if float(p) > PREDICTION_THRESHOLD else "LOW"
            results.append({
                'diabetes_probability': float(p),
                'risk_level': risk,
                'threshold': PREDICTION_THRESHOLD
            })
        return results

# --- App factory ---
def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

    # Setup logging: stdout + rotating file
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(stream_handler)
    app.logger.addHandler(file_handler)

    # Prometheus metrics
    metrics = PrometheusMetrics(app)
    # you can expose default metrics; customize if needed

    # create and attach model server to app
    app.model_server = ModelServer(MODEL_PATH)
    
    # Load model immediately during app creation (works with all Flask versions)
    try:
        app.model_server.load()
        app.logger.info("Model loaded successfully during startup")
    except Exception as e:
        app.logger.exception(f"Failed to load model during startup: {e}")
        # keep server up but readiness will report not ready

    @app.route('/health', methods=['GET'])
    def health_check():
        """Simple liveness check."""
        return jsonify({'status': 'alive'}), 200

    @app.route('/readiness', methods=['GET'])
    def readiness_check():
        """Readiness checks whether model is loaded and ready for prediction."""
        ready = app.model_server.model is not None
        status = 'ready' if ready else 'not_ready'
        code = 200 if ready else 503
        return jsonify({'status': status, 'model_loaded': bool(ready)}), code

    @app.route('/predict', methods=['POST'])
    def predict():
        """
        Single-record predict. Accepts a JSON object with feature keys.
        Example: { "HighBP": 1, "BMI": 23.5, ... }
        """
        try:
            # Block empty body or wrong content type
            if not request.is_json:
                return jsonify({'error': 'Request content-type must be application/json'}), 415

            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON body provided'}), 400

            # lazy-load model if needed
            if app.model_server.model is None:
                app.model_server.load()

            # Preprocess single record -> batch of 1
            features = app.model_server.preprocess_single(data)
            batch = np.expand_dims(features, axis=0)
            result = app.model_server.predict_batch(batch)[0]

            app.logger.info(f"Prediction served (single): {result['risk_level']}")
            return jsonify(result), 200

        except Exception as e:
            app.logger.exception(f"Prediction error: {e}")
            return jsonify({'error': 'internal server error'}), 500

    @app.route('/batch_predict', methods=['POST'])
    def batch_predict():
        """
        Batch predict endpoint.
        Accepts {"records": [ {..}, {..}, ... ]}
        """
        try:
            if not request.is_json:
                return jsonify({'error': 'Request content-type must be application/json'}), 415

            payload = request.get_json()
            records = payload.get('records') if isinstance(payload, dict) else None
            if not records or not isinstance(records, list):
                return jsonify({'error': 'Provide records as a non-empty list under the key "records"'}), 400

            # limit batch size to avoid OOM - configurable via env if needed
            max_batch = int(os.getenv("MAX_BATCH_SIZE", 512))
            if len(records) > max_batch:
                return jsonify({'error': f'Batch too large. Max allowed: {max_batch}'}), 413

            # lazy load model
            if app.model_server.model is None:
                app.model_server.load()

            # Preprocess all records once and call model.predict once
            features_batch = app.model_server.preprocess_batch(records)
            predictions = app.model_server.predict_batch(features_batch)

            app.logger.info(f"Batch prediction served: {len(predictions)} records")
            return jsonify({'predictions': predictions}), 200

        except Exception as e:
            app.logger.exception(f"Batch prediction error: {e}")
            return jsonify({'error': 'internal server error'}), 500

    @app.route('/model_info', methods=['GET'])
    def model_info():
        return jsonify({
            'model_type': 'Federated Neural Network',
            'framework': 'TensorFlow (served by Flask)',
            'input_features': app.model_server.feature_names,
            'output': 'Diabetes probability (0-1)'
        }), 200

    # Global error handlers (optional)
    @app.errorhandler(413)
    def request_entity_too_large(e):
        return jsonify({'error': 'Request payload too large'}), 413

    return app


# Expose the app variable for Gunicorn/uWSGI
app = create_app()

# Optional: run via `python -m src.inference_server` in development
if __name__ == "__main__":
    # For local dev only. Use Gunicorn in production.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5003)), debug=False)
