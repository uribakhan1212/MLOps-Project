# scripts/validate_mlflow_model.py

import mlflow
import json
import os
import argparse
import glob
import shutil
import datetime
from typing import Dict, Any, Optional

def load_from_mlflow(tracking_uri: str, experiment_name: str) -> Optional[Dict[str, Any]]:
    """Load metrics from MLflow"""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        
        # Get latest run
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Experiment '{experiment_name}' not found in MLflow")
            return None
            
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            print(f"❌ No runs found in experiment '{experiment_name}'")
            return None
        
        latest_run = runs.iloc[0]
        
        metrics = {
            "final_avg_accuracy": float(latest_run["metrics.final_avg_accuracy"]),
            "final_avg_auc": float(latest_run["metrics.final_avg_auc"]),
            "final_avg_loss": float(latest_run["metrics.final_avg_loss"]),
            "source": "mlflow",
            "run_id": latest_run["run_id"]
        }
        
        print("✅ Successfully loaded metrics from MLflow")
        return metrics
        
    except Exception as e:
        print(f"❌ Failed to load from MLflow: {e}")
        return None

def load_from_fallback(fallback_dir: str) -> Optional[Dict[str, Any]]:
    """Load metrics from JSON fallback files"""
    try:
        # Find the most recent fallback directory if not specified
        if not fallback_dir:
            fallback_pattern = "fallback_metrics/*"
            fallback_dirs = glob.glob(fallback_pattern)
            if not fallback_dirs:
                print("❌ No fallback directories found")
                return None
            fallback_dir = max(fallback_dirs)  # Most recent by name (timestamp)
            print(f"🔍 Using most recent fallback: {fallback_dir}")
        
        # Load final metrics
        final_metrics_path = os.path.join(fallback_dir, "final_metrics.json")
        if not os.path.exists(final_metrics_path):
            print(f"❌ Final metrics file not found: {final_metrics_path}")
            return None
        
        with open(final_metrics_path, 'r') as f:
            data = json.load(f)
        
        metrics = data.get('metrics', {})
        
        # Ensure required metrics exist
        required_metrics = ["final_avg_accuracy", "final_avg_auc", "final_avg_loss"]
        for metric in required_metrics:
            if metric not in metrics:
                print(f"❌ Required metric '{metric}' not found in fallback")
                return None
        
        # Load round-by-round metrics for additional context
        round_metrics = load_round_metrics(fallback_dir)
        if round_metrics:
            metrics["round_metrics"] = round_metrics
            print(f"✅ Loaded {len(round_metrics)} round metrics")
        
        metrics["source"] = "fallback"
        metrics["fallback_dir"] = fallback_dir
        
        print(f"✅ Successfully loaded metrics from fallback: {fallback_dir}")
        return metrics
        
    except Exception as e:
        print(f"❌ Failed to load from fallback: {e}")
        return None

def load_round_metrics(fallback_dir: str) -> Optional[Dict[int, Dict[str, Any]]]:
    """Load per-round metrics from fallback directory"""
    try:
        round_metrics = {}
        round_files = glob.glob(os.path.join(fallback_dir, "round_*_metrics.json"))
        
        for round_file in sorted(round_files):
            with open(round_file, 'r') as f:
                data = json.load(f)
            
            round_data = data.get('metrics', {})
            round_num = round_data.get('round')
            
            if round_num is not None:
                round_metrics[round_num] = {
                    "val_loss": round_data.get('val_loss'),
                    "val_accuracy": round_data.get('val_accuracy'),
                    "val_auc": round_data.get('val_auc')
                }
        
        return round_metrics if round_metrics else None
        
    except Exception as e:
        print(f"⚠️ Could not load round metrics: {e}")
        return None

def backup_existing_file(filepath: str) -> None:
    """Backup existing file with timestamp to avoid overwriting"""
    if os.path.exists(filepath):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{filepath}.backup_{timestamp}"
        shutil.copy2(filepath, backup_path)
        print(f"📁 Backed up existing file: {filepath} -> {backup_path}")

def main():
    parser = argparse.ArgumentParser(description="Validate model metrics from MLflow or fallback")
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8082"),
                       help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="diabetes-federated-learning",
                       help="MLflow experiment name")
    parser.add_argument("--fallback-dir", help="Fallback directory path (auto-detect if not specified)")
    parser.add_argument("--output", default="model_metrics.json", help="Output JSON file")
    parser.add_argument("--force-fallback", action="store_true", help="Skip MLflow and use fallback only")
    
    args = parser.parse_args()
    
    print("🔍 Validating model metrics...")
    print("=" * 50)
    
    metrics = None
    
    # Try MLflow first (unless forced to use fallback)
    if not args.force_fallback:
        print("📡 Attempting to load from MLflow...")
        metrics = load_from_mlflow(args.mlflow_uri, args.experiment_name)
    
    # Try fallback if MLflow failed or was skipped
    if not metrics:
        print("📁 Attempting to load from fallback...")
        metrics = load_from_fallback(args.fallback_dir)
    
    if not metrics:
        print("❌ Failed to load metrics from both MLflow and fallback")
        exit(1)
    
    # Backup existing output file if using fallback (to preserve Jenkins history)
    if metrics.get('source') == 'fallback':
        backup_existing_file(args.output)
    
    # Save metrics to output file in the same format regardless of source
    output_metrics = {
        "final_avg_accuracy": metrics["final_avg_accuracy"],
        "final_avg_auc": metrics["final_avg_auc"], 
        "final_avg_loss": metrics["final_avg_loss"]
    }
    
    # Add round metrics to a separate detailed file for analysis
    if "round_metrics" in metrics:
        detailed_output = args.output.replace('.json', '_detailed.json')
        detailed_metrics = {
            "final_metrics": output_metrics,
            "round_metrics": metrics["round_metrics"],
            "source": metrics["source"],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open(detailed_output, 'w') as f:
            json.dump(detailed_metrics, f, indent=2)
        print(f"📊 Detailed metrics (including rounds) saved to: {detailed_output}")
    
    with open(args.output, 'w') as f:
        json.dump(output_metrics, f, indent=2)
    
    # Display results
    print("\n" + "=" * 50)
    print("✅ Model metrics validated:")
    print(f"  Source: {metrics['source']}")
    print(f"  Accuracy: {metrics['final_avg_accuracy']:.4f}")
    print(f"  AUC: {metrics['final_avg_auc']:.4f}")
    print(f"  Loss: {metrics['final_avg_loss']:.4f}")
    
    if metrics['source'] == 'mlflow':
        print(f"  Run ID: {metrics.get('run_id', 'N/A')}")
    else:
        print(f"  Fallback Dir: {metrics.get('fallback_dir', 'N/A')}")
    
    print(f"  Saved to: {args.output}")
    print("=" * 50)

if __name__ == "__main__":
    main()