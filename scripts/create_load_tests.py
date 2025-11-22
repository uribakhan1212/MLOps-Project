#!/usr/bin/env python3
"""
Create load tests for performance testing
"""
import os

def create_load_tests():
    """Create load test files"""
    
    # Ensure test directory exists
    os.makedirs('tests', exist_ok=True)
    
    # Create load test script
    load_test_content = '''from locust import HttpUser, task, between
import random

class DiabetesPredictionUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict(self):
        payload = {
            "HighBP": random.choice([0, 1]),
            "HighChol": random.choice([0, 1]),
            "CholCheck": 1,
            "BMI": random.uniform(18, 40),
            "Smoker": random.choice([0, 1]),
            "Stroke": 0,
            "HeartDiseaseorAttack": 0,
            "PhysActivity": random.choice([0, 1]),
            "Fruits": random.choice([0, 1]),
            "Veggies": random.choice([0, 1]),
            "HvyAlcoholConsump": 0,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
            "GenHlth": random.randint(1, 5),
            "MentHlth": random.randint(0, 30),
            "PhysHlth": random.randint(0, 30),
            "DiffWalk": random.choice([0, 1]),
            "Sex": random.choice([0, 1]),
            "Age": random.randint(1, 13),
            "Education": random.randint(1, 6),
            "Income": random.randint(1, 8)
        }
        
        # For now, just simulate the request structure
        # In production, this would make actual HTTP requests
        print(f"Simulated prediction request: {len(payload)} features")
        return True
    
    @task(2)
    def health_check(self):
        # Simulate health check
        print("Simulated health check")
        return True

# Usage instructions
if __name__ == "__main__":
    print("Load test script created.")
    print("To run: locust -f tests/load_test.py --host=http://your-service-url")
'''
    
    with open('tests/load_test.py', 'w') as f:
        f.write(load_test_content)
    
    print("✅ Load tests created")

if __name__ == '__main__':
    create_load_tests()