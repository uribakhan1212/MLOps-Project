# test_api.py
import requests

# Test data
test_patient = {
    'HighBP': 1,
    'HighChol': 1,
    'CholCheck': 1,
    'BMI': 28.5,
    'Smoker': 0,
    'Stroke': 0,
    'HeartDiseaseorAttack': 0,
    'PhysActivity': 1,
    'Fruits': 1,
    'Veggies': 1,
    'HvyAlcoholConsump': 0,
    'AnyHealthcare': 1,
    'NoDocbcCost': 0,
    'GenHlth': 3,
    'MentHlth': 2,
    'PhysHlth': 0,
    'DiffWalk': 0,
    'Sex': 1,
    'Age': 9,
    'Education': 5,
    'Income': 7
}

response = requests.post('http://localhost:5003/predict', json=test_patient)
print(response.json())