#!/usr/bin/env python3
"""
Enhanced data drift detection script using inference data
Detects drift using inference_data.json and adds drifted data to training sets
"""
import pandas as pd
import numpy as np
from scipy import stats
import json
import os
import random
from datetime import datetime

def load_inference_data():
    """Load inference data from dashboards/data/inference_data.json"""
    inference_path = 'dashboards/data/inference_data.json'
    
    if not os.path.exists(inference_path):
        print(f"❌ Inference data not found at {inference_path}")
        return None
    
    try:
        with open(inference_path, 'r') as f:
            data = json.load(f)
        
        # Extract features from predictions
        inference_records = []
        for prediction in data.get('predictions', []):
            features = prediction.get('features', {})
            if features:
                inference_records.append(features)
        
        if not inference_records:
            print("❌ No inference records found in data")
            return None
        
        # Convert to DataFrame
        inference_df = pd.DataFrame(inference_records)
        print(f"✅ Loaded {len(inference_df)} inference records")
        return inference_df
        
    except Exception as e:
        print(f"❌ Error loading inference data: {e}")
        return None

def load_baseline_data():
    """Load baseline training data from federated clients"""
    baseline_dfs = []
    
    for client in ['client_1', 'client_2', 'client_3']:
        train_path = f'federated_data/{client}/train_data.csv'
        if os.path.exists(train_path):
            df = pd.read_csv(train_path)
            baseline_dfs.append(df)
            print(f"✅ Loaded baseline data from {client}: {len(df)} records")
    
    if not baseline_dfs:
        print("❌ No baseline training data found")
        return None
    
    # Combine all baseline data
    baseline = pd.concat(baseline_dfs, ignore_index=True)
    
    # Sample to manage memory if dataset is large
    if len(baseline) > 10000:
        baseline = baseline.sample(n=10000, random_state=42)
        print(f"📊 Sampled baseline data to 10000 records")
    
    return baseline

def detect_drift_ks_test(baseline_df, inference_df, threshold=0.05):
    """
    Detect drift using Kolmogorov-Smirnov test
    """
    # Get common feature columns (exclude target column)
    baseline_features = [col for col in baseline_df.columns if col != 'Diabetes_binary']
    inference_features = list(inference_df.columns)
    
    # Find common features
    common_features = list(set(baseline_features) & set(inference_features))
    
    if not common_features:
        print("❌ No common features found between baseline and inference data")
        return None
    
    print(f"📊 Testing drift for {len(common_features)} common features")
    
    drift_results = {}
    drifted_features = []
    
    for feature in common_features:
        try:
            baseline_values = baseline_df[feature].dropna()
            inference_values = inference_df[feature].dropna()
            
            if len(baseline_values) == 0 or len(inference_values) == 0:
                continue
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(baseline_values, inference_values)
            
            # Calculate mean difference for additional insight
            baseline_mean = baseline_values.mean()
            inference_mean = inference_values.mean()
            mean_diff = abs(inference_mean - baseline_mean)
            mean_diff_pct = (mean_diff / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
            
            # Detect drift if p-value is below threshold
            is_drifted = p_value < threshold
            
            if is_drifted:
                drifted_features.append(feature)
            
            drift_results[feature] = {
                'ks_statistic': float(ks_stat),
                'p_value': float(p_value),
                'baseline_mean': float(baseline_mean),
                'inference_mean': float(inference_mean),
                'mean_difference': float(mean_diff),
                'mean_diff_percentage': float(mean_diff_pct),
                'drift_detected': bool(is_drifted)
            }
            
            print(f"  {feature}: KS={ks_stat:.4f}, p={p_value:.4f}, drift={'YES' if is_drifted else 'NO'}")
            
        except Exception as e:
            print(f"⚠️ Error testing drift for {feature}: {e}")
            drift_results[feature] = {
                'error': str(e),
                'drift_detected': False
            }
    
    return drift_results, drifted_features

def add_inference_data_to_training(inference_df, drifted_features):
    """
    Add inference data to a random client's training data if drift is detected
    """
    if not drifted_features:
        print("✅ No drift detected - no data to add to training")
        return False
    
    print(f"🔄 Adding inference data to training due to drift in: {', '.join(drifted_features)}")
    
    # Choose a random client to add the data to
    clients = ['client_1', 'client_2', 'client_3']
    chosen_client = random.choice(clients)
    
    train_path = f'federated_data/{chosen_client}/train_data.csv'
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found for {chosen_client}")
        return False
    
    try:
        # Load existing training data
        existing_train = pd.read_csv(train_path)
        print(f"📊 Current training data for {chosen_client}: {len(existing_train)} records")
        
        # Prepare inference data for training
        # We need to add a target column (Diabetes_binary) - we'll use a default value or prediction
        inference_for_training = inference_df.copy()
        
        # Add a default target value (you might want to use actual predictions here)
        # For now, we'll use a random binary value or you can modify this logic
        inference_for_training['Diabetes_binary'] = np.random.choice([0, 1], size=len(inference_for_training))
        
        # Ensure column order matches existing training data
        if 'Diabetes_binary' in existing_train.columns:
            # Reorder columns to match existing training data
            column_order = existing_train.columns.tolist()
            inference_for_training = inference_for_training.reindex(columns=column_order, fill_value=0)
        
        # Combine existing training data with inference data
        updated_train = pd.concat([existing_train, inference_for_training], ignore_index=True)
        
        # Save updated training data
        updated_train.to_csv(train_path, index=False)
        
        print(f"✅ Added {len(inference_for_training)} inference records to {chosen_client}")
        print(f"📊 Updated training data for {chosen_client}: {len(updated_train)} records")
        
        # Log the addition
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'client': chosen_client,
            'added_records': len(inference_for_training),
            'total_records': len(updated_train),
            'drifted_features': drifted_features,
            'reason': 'drift_detected'
        }
        
        # Save log
        os.makedirs('reports', exist_ok=True)
        log_file = 'reports/training_data_updates.json'
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"❌ Error adding inference data to training: {e}")
        return False

def create_drift_report(drift_results, drifted_features, baseline_shape, inference_shape, data_added=False):
    """Create comprehensive drift detection report"""
    
    total_features = len(drift_results)
    num_drifted = len(drifted_features)
    drift_percentage = (num_drifted / total_features) * 100 if total_features > 0 else 0
    
    # Determine overall drift status based on threshold
    dataset_drift = drift_percentage > 2.0  # Using 2% threshold as requested
    
    # Create JSON results
    result = {
        'dataset_drift': dataset_drift,
        'drifted_features': num_drifted,
        'total_features': total_features,
        'drift_percentage': drift_percentage / 100,  # Convert to decimal for Jenkins
        'drifted_feature_names': drifted_features,
        'baseline_records': baseline_shape[0] if baseline_shape else 0,
        'inference_records': inference_shape[0] if inference_shape else 0,
        'data_added_to_training': data_added,
        'timestamp': datetime.now().isoformat(),
        'threshold_used': 0.02,
        'feature_details': drift_results
    }
    
    # Save JSON results for Jenkins
    with open('drift_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Drift Detection Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .drift-detected {{ color: #d32f2f; font-weight: bold; }}
            .no-drift {{ color: #388e3c; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .drifted {{ background-color: #ffebee; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Data Drift Detection Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Data Source:</strong> dashboards/data/inference_data.json</p>
        </div>
        
        <h2>Summary</h2>
        <p><strong>Dataset Drift:</strong> <span class="{'drift-detected' if dataset_drift else 'no-drift'}">{dataset_drift}</span></p>
        <p><strong>Drifted Features:</strong> {num_drifted}/{total_features}</p>
        <p><strong>Drift Percentage:</strong> {drift_percentage:.2f}%</p>
        <p><strong>Threshold:</strong> 2.00%</p>
        <p><strong>Baseline Data:</strong> {baseline_shape[0] if baseline_shape else 0} records</p>
        <p><strong>Inference Data:</strong> {inference_shape[0] if inference_shape else 0} records</p>
        <p><strong>Data Added to Training:</strong> {'Yes' if data_added else 'No'}</p>
        
        <h2>Feature Drift Details</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>KS Statistic</th>
                <th>P-Value</th>
                <th>Baseline Mean</th>
                <th>Inference Mean</th>
                <th>Mean Difference</th>
                <th>Drift Detected</th>
            </tr>
    """
    
    for feature, details in drift_results.items():
        if 'error' in details:
            html_content += f"""
            <tr>
                <td>{feature}</td>
                <td colspan="6">Error: {details['error']}</td>
            </tr>
            """
        else:
            row_class = 'drifted' if details['drift_detected'] else ''
            html_content += f"""
            <tr class="{row_class}">
                <td>{feature}</td>
                <td>{details['ks_statistic']:.4f}</td>
                <td>{details['p_value']:.4f}</td>
                <td>{details['baseline_mean']:.4f}</td>
                <td>{details['inference_mean']:.4f}</td>
                <td>{details['mean_difference']:.4f} ({details['mean_diff_percentage']:.2f}%)</td>
                <td>{'Yes' if details['drift_detected'] else 'No'}</td>
            </tr>
            """
    
    html_content += """
        </table>
        
        <h2>Recommendations</h2>
        <ul>
    """
    
    if dataset_drift:
        html_content += """
            <li><strong>Significant drift detected!</strong> Consider retraining the model.</li>
            <li>Inference data has been added to training data for the next retraining cycle.</li>
            <li>Monitor model performance closely.</li>
        """
    else:
        html_content += """
            <li>No significant drift detected. Continue monitoring.</li>
            <li>Current model should perform well on recent data.</li>
        """
    
    html_content += """
        </ul>
    </body>
    </html>
    """
    
    # Save HTML report
    os.makedirs('reports', exist_ok=True)
    with open('reports/drift_report.html', 'w') as f:
        f.write(html_content)
    
    return result

def main():
    """Main drift detection function"""
    print("🔍 Starting enhanced drift detection with inference data...")
    
    # Load inference data
    inference_df = load_inference_data()
    if inference_df is None:
        print("❌ Cannot proceed without inference data")
        return
    
    # Load baseline data
    baseline_df = load_baseline_data()
    if baseline_df is None:
        print("❌ Cannot proceed without baseline data")
        return
    
    print(f"📊 Baseline data shape: {baseline_df.shape}")
    print(f"📊 Inference data shape: {inference_df.shape}")
    
    # Detect drift
    drift_results, drifted_features = detect_drift_ks_test(baseline_df, inference_df)
    
    if drift_results is None:
        print("❌ Drift detection failed")
        return
    
    # Add inference data to training if drift detected
    data_added = False
    if drifted_features:
        print(f"🚨 Drift detected in {len(drifted_features)} features: {', '.join(drifted_features)}")
        data_added = add_inference_data_to_training(inference_df, drifted_features)
    else:
        print("✅ No drift detected")
    
    # Create comprehensive report
    result = create_drift_report(
        drift_results, 
        drifted_features, 
        baseline_df.shape, 
        inference_df.shape,
        data_added
    )
    
    print(f"\n📋 Drift Detection Summary:")
    print(f"   Dataset Drift: {result['dataset_drift']}")
    print(f"   Drifted Features: {result['drifted_features']}/{result['total_features']}")
    print(f"   Drift Percentage: {result['drift_percentage']*100:.2f}%")
    print(f"   Data Added to Training: {data_added}")
    print(f"   Report saved: reports/drift_report.html")
    print(f"   Results saved: drift_results.json")

if __name__ == "__main__":
    main()