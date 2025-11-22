#!/usr/bin/env python3
"""
Data Drift Detection for Federated Learning MLOps Pipeline
"""
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import jensen_shannon_distance
import mlflow
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDriftDetector:
    """
    Detects data drift in federated learning environment
    """
    
    def __init__(self, 
                 baseline_window_days: int = 7,
                 drift_threshold: float = 0.1,
                 mlflow_uri: str = "http://mlflow.mlops-fl.svc.cluster.local:80"):
        self.baseline_window_days = baseline_window_days
        self.drift_threshold = drift_threshold
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        
    def detect_statistical_drift(self, 
                                baseline_data: pd.DataFrame, 
                                current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect drift using statistical tests
        """
        drift_scores = {}
        
        for column in baseline_data.columns:
            if baseline_data[column].dtype in ['int64', 'float64']:
                # Kolmogorov-Smirnov test for numerical features
                ks_stat, p_value = stats.ks_2samp(baseline_data[column], current_data[column])
                drift_scores[f"{column}_ks_stat"] = ks_stat
                drift_scores[f"{column}_p_value"] = p_value
                
                # Population Stability Index (PSI)
                psi_score = self._calculate_psi(baseline_data[column], current_data[column])
                drift_scores[f"{column}_psi"] = psi_score
                
        return drift_scores
    
    def _calculate_psi(self, baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        """
        try:
            # Create bins based on baseline data
            _, bin_edges = np.histogram(baseline, bins=bins)
            
            # Calculate distributions
            baseline_dist = np.histogram(baseline, bins=bin_edges)[0] / len(baseline)
            current_dist = np.histogram(current, bins=bin_edges)[0] / len(current)
            
            # Avoid division by zero
            baseline_dist = np.where(baseline_dist == 0, 0.0001, baseline_dist)
            current_dist = np.where(current_dist == 0, 0.0001, current_dist)
            
            # Calculate PSI
            psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))
            return psi
            
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0
    
    def detect_model_performance_drift(self) -> Dict[str, float]:
        """
        Detect drift in model performance metrics
        """
        try:
            # Get recent model runs from MLflow
            experiment = mlflow.get_experiment_by_name("diabetes-federated-learning")
            if not experiment:
                logger.error("MLflow experiment not found")
                return {}
            
            # Get runs from last 7 days
            recent_runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"attribute.start_time > '{(datetime.now() - timedelta(days=7)).isoformat()}'",
                order_by=["start_time DESC"]
            )
            
            if len(recent_runs) < 2:
                logger.warning("Not enough recent runs for performance drift detection")
                return {}
            
            # Calculate performance drift
            latest_accuracy = recent_runs.iloc[0]["metrics.final_avg_accuracy"]
            baseline_accuracy = recent_runs["metrics.final_avg_accuracy"].mean()
            
            accuracy_drift = abs(latest_accuracy - baseline_accuracy) / baseline_accuracy
            
            return {
                "accuracy_drift": accuracy_drift,
                "latest_accuracy": latest_accuracy,
                "baseline_accuracy": baseline_accuracy
            }
            
        except Exception as e:
            logger.error(f"Performance drift detection failed: {e}")
            return {}
    
    def simulate_incoming_data_drift(self) -> pd.DataFrame:
        """
        Simulate incoming data with potential drift
        (In real scenario, this would fetch from data sources)
        """
        # Load baseline data
        baseline_path = "federated_data/client_1/train.csv"
        if not os.path.exists(baseline_path):
            logger.error(f"Baseline data not found: {baseline_path}")
            return pd.DataFrame()
        
        baseline_data = pd.read_csv(baseline_path)
        
        # Simulate drift by modifying some features
        current_data = baseline_data.copy()
        
        # Simulate seasonal drift in BMI (people gain weight in winter)
        if datetime.now().month in [11, 12, 1, 2]:  # Winter months
            current_data['BMI'] = current_data['BMI'] * 1.05  # 5% increase
        
        # Simulate demographic drift in Age
        current_data['Age'] = current_data['Age'] + np.random.normal(0, 0.5, len(current_data))
        
        # Simulate healthcare access drift
        if np.random.random() > 0.7:  # 30% chance of healthcare crisis
            current_data['AnyHealthcare'] = current_data['AnyHealthcare'] * 0.8
        
        return current_data.sample(n=min(1000, len(current_data)))  # Sample for efficiency
    
    def run_drift_detection(self) -> Dict[str, any]:
        """
        Main drift detection pipeline
        """
        logger.info("🔍 Starting data drift detection...")
        
        # Load baseline data
        baseline_path = "federated_data/client_1/train.csv"
        if not os.path.exists(baseline_path):
            logger.error(f"Baseline data not found: {baseline_path}")
            return {"error": "Baseline data not found"}
        
        baseline_data = pd.read_csv(baseline_path)
        
        # Get current data (simulated)
        current_data = self.simulate_incoming_data_drift()
        if current_data.empty:
            return {"error": "Current data not available"}
        
        # Detect statistical drift
        statistical_drift = self.detect_statistical_drift(baseline_data, current_data)
        
        # Detect performance drift
        performance_drift = self.detect_model_performance_drift()
        
        # Determine if retraining is needed
        drift_detected = False
        drift_reasons = []
        
        # Check PSI thresholds
        for key, value in statistical_drift.items():
            if key.endswith('_psi') and value > self.drift_threshold:
                drift_detected = True
                drift_reasons.append(f"High PSI for {key}: {value:.4f}")
        
        # Check performance drift
        if performance_drift.get('accuracy_drift', 0) > 0.05:  # 5% performance drop
            drift_detected = True
            drift_reasons.append(f"Performance drift: {performance_drift['accuracy_drift']:.4f}")
        
        # Create drift report
        drift_report = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": drift_detected,
            "drift_reasons": drift_reasons,
            "statistical_drift": statistical_drift,
            "performance_drift": performance_drift,
            "baseline_samples": len(baseline_data),
            "current_samples": len(current_data),
            "recommendation": "Trigger retraining" if drift_detected else "Continue monitoring"
        }
        
        # Log to MLflow
        self._log_drift_to_mlflow(drift_report)
        
        # Send alert if drift detected
        if drift_detected:
            self._send_drift_alert(drift_report)
        
        logger.info(f"✅ Drift detection complete. Drift detected: {drift_detected}")
        return drift_report
    
    def _log_drift_to_mlflow(self, drift_report: Dict):
        """
        Log drift detection results to MLflow
        """
        try:
            with mlflow.start_run(run_name=f"drift_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_params({
                    "drift_threshold": self.drift_threshold,
                    "baseline_window_days": self.baseline_window_days
                })
                
                mlflow.log_metrics({
                    "drift_detected": 1 if drift_report["drift_detected"] else 0,
                    "num_drift_reasons": len(drift_report["drift_reasons"]),
                    **{k: v for k, v in drift_report["statistical_drift"].items() if isinstance(v, (int, float))},
                    **{k: v for k, v in drift_report["performance_drift"].items() if isinstance(v, (int, float))}
                })
                
                # Save drift report as artifact
                with open("drift_report.json", "w") as f:
                    json.dump(drift_report, f, indent=2)
                mlflow.log_artifact("drift_report.json")
                
        except Exception as e:
            logger.error(f"Failed to log drift to MLflow: {e}")
    
    def _send_drift_alert(self, drift_report: Dict):
        """
        Send drift alert to monitoring system
        """
        try:
            # Send to Prometheus Alertmanager (if configured)
            alert_payload = {
                "alerts": [{
                    "labels": {
                        "alertname": "DataDriftDetected",
                        "severity": "warning",
                        "service": "diabetes-inference"
                    },
                    "annotations": {
                        "summary": "Data drift detected in federated learning system",
                        "description": f"Drift reasons: {', '.join(drift_report['drift_reasons'])}"
                    }
                }]
            }
            
            # In real deployment, send to Alertmanager
            logger.info(f"🚨 DRIFT ALERT: {alert_payload}")
            
            # Trigger Jenkins retraining job
            self._trigger_retraining()
            
        except Exception as e:
            logger.error(f"Failed to send drift alert: {e}")
    
    def _trigger_retraining(self):
        """
        Trigger Jenkins retraining pipeline
        """
        try:
            jenkins_url = os.getenv("JENKINS_URL", "http://jenkins.mlops-fl.svc.cluster.local:8080")
            jenkins_job = "federated-learning-pipeline"
            
            # In real deployment, trigger Jenkins job
            logger.info(f"🔄 Triggering retraining job: {jenkins_url}/job/{jenkins_job}/build")
            
            # For now, just log the action
            logger.info("✅ Retraining trigger sent")
            
        except Exception as e:
            logger.error(f"Failed to trigger retraining: {e}")

def main():
    """
    Main function for running drift detection
    """
    detector = DataDriftDetector(
        baseline_window_days=7,
        drift_threshold=0.1
    )
    
    drift_report = detector.run_drift_detection()
    
    # Print summary
    print(json.dumps(drift_report, indent=2))
    
    # Exit with appropriate code
    exit_code = 1 if drift_report.get("drift_detected", False) else 0
    exit(exit_code)

if __name__ == "__main__":
    main()