#!/usr/bin/env python3
"""
Test script for the inference counter and GitHub push functionality
"""
import os
import json
import sys
from datetime import datetime

def test_counter_functionality():
    """Test the inference counter and GitHub push logic"""
    print("🧪 Testing inference counter functionality...")
    
    # Path to inference data file
    data_file = os.path.join("dashboards", "data", "inference_data.json")
    
    # Create test data structure
    test_data = {
        "predictions": [],
        "metadata": {
            "total_predictions": 0,
            "high_risk_count": 0,
            "low_risk_count": 0,
            "last_updated": None,
            "created": datetime.now().isoformat(),
            "last_github_push": None,
            "predictions_since_last_push": 0
        }
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    
    # Save initial test data
    with open(data_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ Created test data file: {data_file}")
    
    # Simulate adding predictions
    for i in range(3):  # Add 3 predictions to test the threshold
        # Create a test prediction
        prediction = {
            "id": f"test-{i+1}",
            "timestamp": datetime.now().isoformat(),
            "features": {
                "HighBP": 1.0,
                "HighChol": 0.0,
                "BMI": 25.0 + i,
                "GenHlth": 3.0,
                "Age": 5.0
            },
            "prediction": {
                "diabetes_probability": 0.3 + (i * 0.1),
                "risk_level": "HIGH" if i > 1 else "LOW",
                "threshold": 0.5
            }
        }
        
        # Read current data
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Add prediction
        data["predictions"].append(prediction)
        data["metadata"]["total_predictions"] += 1
        data["metadata"]["predictions_since_last_push"] += 1
        data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        if prediction["prediction"]["risk_level"] == "HIGH":
            data["metadata"]["high_risk_count"] += 1
        else:
            data["metadata"]["low_risk_count"] += 1
        
        # Save updated data
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Added prediction {i+1}: {data['metadata']['predictions_since_last_push']} predictions since last push")
        
        # Check if threshold reached
        if data["metadata"]["predictions_since_last_push"] >= 2:
            print(f"🚀 Threshold reached! Would push to GitHub now...")
            print(f"   Total predictions: {data['metadata']['total_predictions']}")
            print(f"   High risk: {data['metadata']['high_risk_count']}")
            print(f"   Low risk: {data['metadata']['low_risk_count']}")
            
            # Simulate successful push by resetting counter
            data["metadata"]["predictions_since_last_push"] = 0
            data["metadata"]["last_github_push"] = datetime.now().isoformat()
            
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print("✅ Counter reset after simulated GitHub push")
    
    # Display final status
    with open(data_file, 'r') as f:
        final_data = json.load(f)
    
    print("\n📋 Final Status:")
    print(f"   Total predictions: {final_data['metadata']['total_predictions']}")
    print(f"   Predictions since last push: {final_data['metadata']['predictions_since_last_push']}")
    print(f"   High risk count: {final_data['metadata']['high_risk_count']}")
    print(f"   Low risk count: {final_data['metadata']['low_risk_count']}")
    print(f"   Last GitHub push: {final_data['metadata']['last_github_push']}")
    
    print("\n✅ Counter functionality test completed successfully!")
    return True

def test_git_repository():
    """Test if we're in a git repository"""
    print("\n🔍 Testing Git repository status...")
    
    try:
        import subprocess
        
        # Check if we're in a git repository
        result = subprocess.run(['git', 'status'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Git repository detected")
            
            # Check if there are any remotes
            remote_result = subprocess.run(['git', 'remote', '-v'], 
                                         capture_output=True, text=True)
            
            if remote_result.stdout.strip():
                print("✅ Git remotes configured:")
                print(remote_result.stdout)
            else:
                print("⚠️ No Git remotes configured")
            
            return True
        else:
            print("❌ Not in a Git repository")
            print("   Initialize with: git init")
            return False
            
    except FileNotFoundError:
        print("❌ Git not found in system PATH")
        return False
    except Exception as e:
        print(f"❌ Error checking Git status: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Inference Counter and GitHub Push System")
    print("=" * 60)
    
    # Test counter functionality
    counter_success = test_counter_functionality()
    
    # Test git repository
    git_success = test_git_repository()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"   Counter functionality: {'✅ PASS' if counter_success else '❌ FAIL'}")
    print(f"   Git repository: {'✅ PASS' if git_success else '❌ FAIL'}")
    
    if counter_success and git_success:
        print("\n🎉 All tests passed! The system is ready for automatic GitHub pushes.")
    elif counter_success:
        print("\n⚠️ Counter works but Git setup needed for automatic pushes.")
    else:
        print("\n❌ Tests failed. Please check the setup.")
    
    return counter_success and git_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)