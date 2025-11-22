#!/usr/bin/env python3
"""
Data validation script for federated learning datasets
"""
import pandas as pd
import os
import sys

def validate_data():
    """Validate federated data integrity"""
    clients = ['client_1', 'client_2', 'client_3']
    
    for client in clients:
        train_path = f'federated_data/{client}/train_data.csv'
        val_path = f'federated_data/{client}/val_data.csv'
        
        # Check file existence
        if not os.path.exists(train_path):
            print(f"❌ Missing: {train_path}")
            sys.exit(1)
        if not os.path.exists(val_path):
            print(f"❌ Missing: {val_path}")
            sys.exit(1)
        
        # Load and validate
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        # Check columns
        expected_cols = 22  # 21 features + 1 target
        if train_df.shape[1] != expected_cols:
            print(f"❌ {client}: Expected {expected_cols} columns, got {train_df.shape[1]}")
            sys.exit(1)
        
        # Check for nulls
        if train_df.isnull().sum().sum() > 0:
            print(f"❌ {client}: Contains null values")
            sys.exit(1)
        
        print(f"✓ {client}: {len(train_df)} train, {len(val_df)} val samples")
    
    print("✅ All data validation checks passed!")

if __name__ == '__main__':
    validate_data()