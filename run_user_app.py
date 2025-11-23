#!/usr/bin/env python3
"""
Script to run the user inference app
"""

import subprocess
import sys
import os

def main():
    print("🩺 Starting Diabetes Risk Predictor App...")
    print("📋 Make sure you've run './monitor_all_services.sh' first to start all services")
    print("   The app will connect to the inference API on http://localhost:5003")
    print()
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "user_inference_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--theme.base", "light"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())