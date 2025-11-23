#!/usr/bin/env python3
"""
Test script for the enhanced drift detection
"""
import os
import sys

def test_drift_detection():
    """Test the enhanced drift detection script"""
    print("🧪 Testing enhanced drift detection...")
    
    # Check if required files exist
    required_files = [
        'dashboards/data/inference_data.json',
        'scripts/detect_drift_inference.py'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file missing: {file_path}")
            return False
        else:
            print(f"✅ Found: {file_path}")
    
    # Check if federated data exists
    clients = ['client_1', 'client_2', 'client_3']
    training_data_found = False
    
    for client in clients:
        train_path = f'federated_data/{client}/train_data.csv'
        if os.path.exists(train_path):
            print(f"✅ Found training data: {train_path}")
            training_data_found = True
        else:
            print(f"⚠️ Missing training data: {train_path}")
    
    if not training_data_found:
        print("❌ No training data found for any client")
        return False
    
    # Run the drift detection script
    print("\n🔍 Running drift detection...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'scripts/detect_drift_inference.py'
        ], capture_output=True, text=True, timeout=60)
        
        print("📊 Drift detection output:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️ Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Drift detection completed successfully")
            
            # Check if output files were created
            output_files = ['drift_results.json', 'reports/drift_report.html']
            for output_file in output_files:
                if os.path.exists(output_file):
                    print(f"✅ Created: {output_file}")
                else:
                    print(f"⚠️ Missing output: {output_file}")
            
            return True
        else:
            print(f"❌ Drift detection failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Drift detection timed out")
        return False
    except Exception as e:
        print(f"❌ Error running drift detection: {e}")
        return False

if __name__ == "__main__":
    success = test_drift_detection()
    sys.exit(0 if success else 1)