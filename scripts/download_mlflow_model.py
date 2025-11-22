# scripts/download_mlflow_model.py

import mlflow
import os
import argparse
import json
import glob
import shutil
import tensorflow as tf
from typing import Optional

def download_from_mlflow(tracking_uri: str, model_name: str, output_dir: str) -> Optional[str]:
    """Download model from MLflow"""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        
        # Get latest version of registered model
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        
        if not latest_versions:
            print(f"❌ No versions found for model {model_name}")
            return None
        
        latest_version = latest_versions[0]
        
        print(f"📥 Downloading model version: {latest_version.version}")
        print(f"   Run ID: {latest_version.run_id}")
        
        # Download model
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.tensorflow.load_model(model_uri)
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/tff_federated_diabetes_model.h5"
        model.save(output_path)
        
        print(f"✅ Model downloaded from MLflow and saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to download from MLflow: {e}")
        return None

def copy_from_fallback(fallback_dir: str, output_dir: str) -> Optional[str]:
    """Copy model from fallback location"""
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
        
        # Load model info
        model_info_path = os.path.join(fallback_dir, "model_info.json")
        if not os.path.exists(model_info_path):
            print(f"❌ Model info file not found: {model_info_path}")
            return None
        
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        # Get model path from info
        source_model_path = model_info.get('metrics', {}).get('model_path')
        if not source_model_path:
            # Try alternative path structure
            source_model_path = model_info.get('model_path')
        
        if not source_model_path or not os.path.exists(source_model_path):
            print(f"❌ Source model not found at: {source_model_path}")
            
            # Try to find model in standard location
            standard_path = "models/tff_federated_diabetes_model.h5"
            if os.path.exists(standard_path):
                source_model_path = standard_path
                print(f"🔍 Found model at standard location: {standard_path}")
            else:
                return None
        
        # Ensure output directory exists and copy model
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.abspath(f"{output_dir}/tff_federated_diabetes_model.h5")
        source_path = os.path.abspath(source_model_path)
        
        if source_path != output_path:
            shutil.copy2(source_path, output_path)
            print(f"📁 Model copied from fallback: {source_path} -> {output_path}")
        else:
            print(f"📁 Model already at target location: {output_path}")
        
        # Verify model can be loaded
        try:
            model = tf.keras.models.load_model(output_path)
            print(f"✅ Model verified and ready at {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Model verification failed: {e}")
            return None
        
    except Exception as e:
        print(f"❌ Failed to copy from fallback: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download model from MLflow or copy from fallback")
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8082"),
                       help="MLflow tracking URI")
    parser.add_argument("--model-name", default="diabetes-federated-model",
                       help="MLflow registered model name")
    parser.add_argument("--output-dir", default="models", help="Output directory for model")
    parser.add_argument("--fallback-dir", help="Fallback directory path (auto-detect if not specified)")
    parser.add_argument("--force-fallback", action="store_true", help="Skip MLflow and use fallback only")
    
    args = parser.parse_args()
    
    print("📦 Downloading/copying model...")
    print("=" * 50)
    
    model_path = None
    
    # Try MLflow first (unless forced to use fallback)
    # if not args.force_fallback:
    #     print("📡 Attempting to download from MLflow...")
    #     model_path = download_from_mlflow(args.mlflow_uri, args.model_name, args.output_dir)
    
    # Try fallback if MLflow failed or was skipped
    if not model_path:
        print("📁 Attempting to copy from fallback...")
        model_path = copy_from_fallback(args.fallback_dir, args.output_dir)
    
    if not model_path:
        print("❌ Failed to get model from both MLflow and fallback")
        exit(1)
    
    print("\n" + "=" * 50)
    print("✅ Model ready for use!")
    print(f"   Path: {model_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()