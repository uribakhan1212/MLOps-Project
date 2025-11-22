#!/usr/bin/env python3
"""
Data drift detection script using Evidently
"""
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import json
import os

def detect_drift():
    """Detect data drift between baseline and current data"""
    
    # Load baseline (reference) data
    baseline_dfs = []
    for client in ['client_1', 'client_2', 'client_3']:
        train_path = f'federated_data/{client}/train_data.csv'
        if os.path.exists(train_path):
            baseline_dfs.append(pd.read_csv(train_path))
    
    if not baseline_dfs:
        print("❌ No baseline data found")
        return
    
    baseline = pd.concat(baseline_dfs)
    baseline = baseline.sample(n=min(10000, len(baseline)), random_state=42)
    
    # Load current data (using validation data as proxy)
    current_dfs = []
    for client in ['client_1', 'client_2', 'client_3']:
        val_path = f'federated_data/{client}/val_data.csv'
        if os.path.exists(val_path):
            current_dfs.append(pd.read_csv(val_path))
    
    if not current_dfs:
        print("❌ No current data found")
        return
    
    current = pd.concat(current_dfs)
    current = current.sample(n=min(5000, len(current)), random_state=42)
    
    # Create drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=baseline, current_data=current)
    
    # Save report
    os.makedirs('reports', exist_ok=True)
    report.save_html('reports/drift_report.html')
    
    # Get drift results
    drift_results = report.as_dict()
    dataset_drift = drift_results['metrics'][0]['result']['dataset_drift']
    
    # Count drifted features
    drift_by_columns = drift_results['metrics'][0]['result'].get('drift_by_columns', {})
    drifted_features = sum(1 for col_drift in drift_by_columns.values() if col_drift.get('drift_detected', False))
    total_features = len(drift_by_columns)
    drift_percentage = drifted_features / total_features if total_features > 0 else 0
    
    result = {
        'dataset_drift': bool(dataset_drift),
        'drifted_features': drifted_features,
        'total_features': total_features,
        'drift_percentage': float(drift_percentage)
    }
    
    with open('drift_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Drift detected: {dataset_drift}")
    print(f"Drifted features: {drifted_features}/{total_features} ({drift_percentage*100:.2f}%)")
    
    return result

if __name__ == '__main__':
    detect_drift()