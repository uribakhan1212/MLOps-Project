from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


def create_federated_clients(df, n_clients=3, stratify_col='Diabetes_binary', random_state=None):
    """
    Split data into federated clients with UNEVEN distribution.
    Uses Dirichlet distribution to generate random data volume ratios.
    """
    
    # Separate features and target
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    # Create client data dictionaries
    clients_data = {f'client_{i+1}': {'X': None, 'y': None} for i in range(n_clients)}
    
    # 1. Generate random proportions for data volume
    # alpha=1.0 gives uniform randomness (e.g., [0.1, 0.8, 0.1] or [0.33, 0.33, 0.33])
    # We perform this generation even if random_state is set to ensure reproducibility if desired
    if random_state is not None:
        np.random.seed(random_state)
        
    # Generate random ratios that sum to 1
    proportions = np.random.dirichlet(np.repeat(1.0, n_clients))
    
    print(f"\nGenerated client data ratios: {np.round(proportions, 2)}")

    # 2. Calculate exact sample count for each client
    total_samples = len(df)
    # Multiply total by ratio and convert to integer
    client_ns = (proportions * total_samples).astype(int)
    # Fix rounding errors (assign remaining samples to the last client)
    client_ns[-1] = total_samples - sum(client_ns[:-1])

    # 3. Shuffle indices to ensure randomness before splitting
    indices = np.random.permutation(total_samples)
    
    # 4. Assign data chunks to clients
    start_idx = 0
    for i, n_samples in enumerate(client_ns):
        client_key = f'client_{i+1}'
        
        # Select indices for this client
        client_indices = indices[start_idx : start_idx + n_samples]
        
        # Assign data
        clients_data[client_key]['X'] = X.iloc[client_indices].reset_index(drop=True)
        clients_data[client_key]['y'] = y.iloc[client_indices].reset_index(drop=True)
        
        # Move start index
        start_idx += n_samples
    
    # Verify distribution
    for client_name, data in clients_data.items():
        print(f"\n{client_name}:")
        print(f"  Samples: {len(data['y'])} ({len(data['y'])/total_samples*100:.1f}% of total)")
        print(f"  Class distribution: {Counter(data['y'])}")
        print(f"  Positive rate: {(data['y'].sum() / len(data['y']) * 100):.2f}%")
        # Simple feature check
        print(f"  Mean BMI: {data['X']['BMI'].mean():.2f}")
    
    return clients_data


def preprocess_client_data(clients_data):
    """
    Preprocess data for each client independently
    This simulates local data processing at each hospital
    """
    
    preprocessed_clients = {}
    
    for client_name, data in clients_data.items():
        X = data['X'].copy()
        y = data['y'].copy()
        
        # 1. Handle any missing values (if present)
        X.fillna(X.median(), inplace=True)
        
        # 2. Feature scaling (fit scaler on client's own data)
        # In FL, each client normalizes based on local statistics
        scaler = StandardScaler()
        
        # Only scale numerical features (BMI is the main one)
        numerical_cols = ['BMI']
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        # 3. Create train/validation split for each client
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        preprocessed_clients[client_name] = {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'scaler': scaler,
            'feature_names': X.columns.tolist()
        }
        
        print(f"\n{client_name} - Preprocessing complete:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Training positive rate: {(y_train.sum()/len(y_train)*100):.2f}%")
    
    return preprocessed_clients


def save_client_data(preprocessed_clients, output_dir='../federated_data'):
    """
    Save each client's data separately
    Simulates distributed data storage
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    for client_name, data in preprocessed_clients.items():
        client_dir = os.path.join(output_dir, client_name)
        os.makedirs(client_dir, exist_ok=True)
        
        # Save train and validation sets
        pd.concat([
            data['X_train'], 
            data['y_train'].rename('Diabetes_binary')
        ], axis=1).to_csv(f'{client_dir}/train_data.csv', index=False)
        
        pd.concat([
            data['X_val'], 
            data['y_val'].rename('Diabetes_binary')
        ], axis=1).to_csv(f'{client_dir}/val_data.csv', index=False)
        
        print(f"✓ Saved {client_name} data to {client_dir}")


def federated_data_pipeline(csv_path, n_clients=3, split_type='iid'):
    """
    Complete pipeline for federated data preparation
    """
    
    # 1. Load data
    print("Step 1: Loading data...")
    df = pd.read_csv(csv_path)
    
    # 2. Data exploration
    print("\nStep 2: Data exploration...")
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(df.columns)-1}")
    print(f"Class distribution: {Counter(df['Diabetes_binary'])}")
    feature_range = df.drop('Diabetes_binary', axis=1).max() - df.drop('Diabetes_binary', axis=1).min()
    print("Feature ranges:")
    print(feature_range.to_string())
    
    # 3. Create federated clients
    print("\nStep 3: Creating federated clients...")
    if split_type == 'iid':
        clients_data = create_federated_clients(df, n_clients)
    else:
        raise ValueError(f"Invalid split type: {split_type}")
    
    # 4. Preprocess each client's data
    print("\nStep 4: Preprocessing client data...")
    preprocessed_clients = preprocess_client_data(clients_data)
    
    # 5. Save data
    print("\nStep 5: Saving client data...")
    save_client_data(preprocessed_clients)
    
    # 6. Generate summary statistics
    print("\n" + "="*60)
    print("FEDERATED DATA PREPARATION COMPLETE")
    print("="*60)
    
    return preprocessed_clients

# Execute pipeline
clients = federated_data_pipeline(
    'diabetes_binary_5050split_health_indicators_BRFSS2015.csv',
    n_clients=3  # Change to 'age_based' or 'income_based' for non-IID
)