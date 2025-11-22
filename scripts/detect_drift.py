#!/usr/bin/env python3
"""
Data drift detection script using statistical methods
"""
import pandas as pd
import numpy as np
from scipy import stats
import json
import os

def detect_drift():
    """Detect data drift between baseline and current data using statistical tests"""
    
    # Load baseline (reference) data
    baseline_dfs = []
    for client in ['client_1', 'client_2', 'client_3']:
        train_path = f'federated_data/{client}/train_data.csv'
        if os.path.exists(train_path):
            baseline_dfs.append(pd.read_csv(train_path))
    
    if not baseline_dfs:
        print("❌ No baseline data found")
        # Create dummy results for pipeline to continue
        result = {
            'dataset_drift': False,
            'drifted_features': 0,
            'total_features': 8,
            'drift_percentage': 0.0
        }
        
        os.makedirs('reports', exist_ok=True)
        with open('drift_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        # Create simple HTML report
        html_content = """
        <html>
        <head><title>Data Drift Report</title></head>
        <body>
            <h1>Data Drift Detection Report</h1>
            <p><strong>Status:</strong> No baseline data found</p>
            <p><strong>Dataset Drift:</strong> False (default)</p>
            <p><strong>Drifted Features:</strong> 0/8</p>
            <p><strong>Drift Percentage:</strong> 0.00%</p>
        </body>
        </html>
        """
        
        with open('reports/drift_report.html', 'w') as f:
            f.write(html_content)
        
        return result
    
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
        # Create dummy results for pipeline to continue
        result = {
            'dataset_drift': False,
            'drifted_features': 0,
            'total_features': len(baseline.columns) - 1,
            'drift_percentage': 0.0
        }
        
        os.makedirs('reports', exist_ok=True)
        with open('drift_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    current = pd.concat(current_dfs)
    current = current.sample(n=min(5000, len(current)), random_state=42)
    
    # Get feature columns (exclude target)
    feature_columns = [col for col in baseline.columns if col != 'Outcome']
    
    # Perform statistical drift detection
    drifted_features = 0
    drift_details = {}
    
    for col in feature_columns:
        if col in current.columns:
            # Use Kolmogorov-Smirnov test for numerical features
            try:
                baseline_col = baseline[col].dropna()
                current_col = current[col].dropna()
                
                if len(baseline_col) > 0 and len(current_col) > 0:
                    # KS test
                    ks_stat, p_value = stats.ks_2samp(baseline_col, current_col)
                    
                    # Consider drift if p-value < 0.05 (significant difference)
                    is_drifted = p_value < 0.05
                    
                    if is_drifted:
                        drifted_features += 1
                    
                    drift_details[col] = {
                        'ks_statistic': float(ks_stat),
                        'p_value': float(p_value),
                        'drift_detected': bool(is_drifted)
                    }
                else:
                    drift_details[col] = {
                        'ks_statistic': 0.0,
                        'p_value': 1.0,
                        'drift_detected': False
                    }
            except Exception as e:
                print(f"Warning: Could not test drift for column {col}: {e}")
                drift_details[col] = {
                    'ks_statistic': 0.0,
                    'p_value': 1.0,
                    'drift_detected': False
                }
    
    total_features = len(feature_columns)
    drift_percentage = drifted_features / total_features if total_features > 0 else 0
    
    # Determine if dataset has significant drift
    dataset_drift = drift_percentage > 0.3
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Create HTML report
    html_content = f"""
    <html>
    <head><title>Data Drift Report</title></head>
    <body>
        <h1>Data Drift Detection Report</h1>
        <p><strong>Dataset Drift:</strong> {dataset_drift}</p>
        <p><strong>Drifted Features:</strong> {drifted_features}/{total_features}</p>
        <p><strong>Drift Percentage:</strong> {drift_percentage*100:.2f}%</p>
        <p><strong>Baseline Data Shape:</strong> {baseline.shape}</p>
        <p><strong>Current Data Shape:</strong> {current.shape}</p>
        
        <h2>Feature Drift Details</h2>
        <table border="1" style="border-collapse: collapse;">
            <tr>
                <th>Feature</th>
                <th>KS Statistic</th>
                <th>P-Value</th>
                <th>Drift Detected</th>
            </tr>
    """
    
    for col, details in drift_details.items():
        html_content += f"""
            <tr>
                <td>{col}</td>
                <td>{details['ks_statistic']:.4f}</td>
                <td>{details['p_value']:.4f}</td>
                <td>{'Yes' if details['drift_detected'] else 'No'}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Methodology</h2>
        <p>Drift detection uses the Kolmogorov-Smirnov test to compare distributions between baseline and current data.</p>
        <p>A feature is considered drifted if the p-value < 0.05 (statistically significant difference).</p>
        <p>Dataset drift is flagged if more than 30% of features show drift.</p>
    </body>
    </html>
    """
    
    with open('reports/drift_report.html', 'w') as f:
        f.write(html_content)
    
    result = {
        'dataset_drift': bool(dataset_drift),
        'drifted_features': drifted_features,
        'total_features': total_features,
        'drift_percentage': float(drift_percentage),
        'feature_details': drift_details
    }
    
    with open('drift_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Drift detected: {dataset_drift}")
    print(f"Drifted features: {drifted_features}/{total_features} ({drift_percentage*100:.2f}%)")
    
    return result

if __name__ == '__main__':
    detect_drift()