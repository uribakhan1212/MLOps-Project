#!/usr/bin/env python3
"""
Debug Monitoring Setup
Helps diagnose monitoring issues in WSL environment
"""

import subprocess
import time
import requests
import json

def check_docker_containers():
    """Check Docker container status"""
    print("🐳 Checking Docker containers...")
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        print("Running containers:")
        print(result.stdout)
        
        # Check specific containers
        containers = ['mlops-prometheus', 'mlops-grafana', 'mlops-alertmanager']
        for container in containers:
            result = subprocess.run(['docker', 'inspect', container], capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)[0]
                state = data['State']
                print(f"\n{container}:")
                print(f"  Status: {state['Status']}")
                print(f"  Running: {state['Running']}")
                if not state['Running']:
                    print(f"  Exit Code: {state['ExitCode']}")
                    print(f"  Error: {state.get('Error', 'None')}")
            else:
                print(f"\n{container}: Not found")
                
    except Exception as e:
        print(f"Error checking containers: {e}")

def check_container_logs():
    """Check container logs for errors"""
    print("\n📋 Checking container logs...")
    containers = ['mlops-prometheus', 'mlops-grafana', 'mlops-alertmanager']
    
    for container in containers:
        print(f"\n--- {container} logs ---")
        try:
            result = subprocess.run(['docker', 'logs', '--tail', '20', container], 
                                  capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        except Exception as e:
            print(f"Error getting logs for {container}: {e}")

def check_port_bindings():
    """Check port bindings"""
    print("\n🔌 Checking port bindings...")
    try:
        result = subprocess.run(['docker', 'port', 'mlops-prometheus'], capture_output=True, text=True)
        print(f"Prometheus ports: {result.stdout.strip()}")
        
        result = subprocess.run(['docker', 'port', 'mlops-grafana'], capture_output=True, text=True)
        print(f"Grafana ports: {result.stdout.strip()}")
        
        result = subprocess.run(['docker', 'port', 'mlops-alertmanager'], capture_output=True, text=True)
        print(f"Alertmanager ports: {result.stdout.strip()}")
        
    except Exception as e:
        print(f"Error checking ports: {e}")

def test_connectivity_with_retry():
    """Test connectivity with multiple retries"""
    print("\n🔄 Testing connectivity with retries...")
    
    services = [
        ("Prometheus", "http://localhost:9090/api/v1/status/config"),
        ("Grafana", "http://localhost:3000/api/health"),
        ("Alertmanager", "http://localhost:9093/api/v1/status")
    ]
    
    for service_name, url in services:
        print(f"\nTesting {service_name}...")
        for attempt in range(5):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"  ✅ Attempt {attempt + 1}: Success")
                    break
                else:
                    print(f"  ⚠️ Attempt {attempt + 1}: HTTP {response.status_code}")
            except requests.exceptions.ConnectionError:
                print(f"  ❌ Attempt {attempt + 1}: Connection refused")
            except Exception as e:
                print(f"  ❌ Attempt {attempt + 1}: {e}")
            
            if attempt < 4:
                time.sleep(10)  # Wait 10 seconds between attempts

def check_wsl_networking():
    """Check WSL networking specifics"""
    print("\n🌐 Checking WSL networking...")
    try:
        # Check if we're in WSL
        result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
        if 'microsoft' in result.stdout.lower() or 'wsl' in result.stdout.lower():
            print("✅ Running in WSL environment")
            
            # Check WSL IP
            result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
            print(f"WSL IP addresses: {result.stdout.strip()}")
            
            # Check if Docker Desktop is running
            result = subprocess.run(['docker', 'version'], capture_output=True, text=True)
            if 'Docker Desktop' in result.stdout:
                print("✅ Docker Desktop detected")
            else:
                print("⚠️ Docker Desktop not detected - this might cause networking issues")
                
        else:
            print("Not running in WSL")
            
    except Exception as e:
        print(f"Error checking WSL: {e}")

def suggest_fixes():
    """Suggest potential fixes"""
    print("\n🔧 SUGGESTED FIXES:")
    print("=" * 50)
    print("1. Wait longer for services to start:")
    print("   sleep 60 && python scripts/test_monitoring_setup.py")
    print()
    print("2. Check if ports are already in use:")
    print("   netstat -tulpn | grep -E ':(9090|3000|9093)'")
    print()
    print("3. Restart Docker Desktop (if using WSL):")
    print("   - Close Docker Desktop")
    print("   - Restart Docker Desktop")
    print("   - Wait for it to fully start")
    print()
    print("4. Try accessing via Docker container IP:")
    print("   docker inspect mlops-prometheus | grep IPAddress")
    print()
    print("5. Restart the monitoring stack:")
    print("   cd monitoring")
    print("   docker-compose -f docker-compose.simple.yml down")
    print("   docker-compose -f docker-compose.simple.yml up -d")
    print()
    print("6. Check Docker daemon:")
    print("   sudo systemctl status docker")
    print("   sudo systemctl restart docker")

def main():
    """Main debug function"""
    print("🔍 MLOps Monitoring Debug Tool")
    print("=" * 50)
    
    check_docker_containers()
    check_container_logs()
    check_port_bindings()
    check_wsl_networking()
    test_connectivity_with_retry()
    suggest_fixes()

if __name__ == "__main__":
    main()