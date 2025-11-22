# scripts/get_latest_mlflow_run.py

import mlflow
import json
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.mlops-fl.svc.cluster.local:5000")
EXPERIMENT_NAME = "diabetes-federated-learning"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Get experiment
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    raise Exception(f"Experiment {EXPERIMENT_NAME} not found")

# Get latest run
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

if runs.empty:
    raise Exception("No runs found in experiment")

latest_run = runs.iloc[0]

run_info = {
    "run_id": latest_run.run_id,
    "experiment_id": experiment.experiment_id,
    "final_avg_accuracy": float(latest_run["metrics.final_avg_accuracy"]),
    "final_avg_auc": float(latest_run["metrics.final_avg_auc"]),
    "final_avg_loss": float(latest_run["metrics.final_avg_loss"]),
    "status": latest_run.status,
}

print(json.dumps(run_info, indent=2))