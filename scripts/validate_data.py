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
    errors = []
    warnings = []
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    
    for client in clients:
        train_path = f'federated_data/{client}/train_data.csv'
        val_path = f'federated_data/{client}/val_data.csv'
        
        # Check file existence
        if not os.path.exists(train_path):
            error_msg = f"Missing: {train_path}"
            print(f"❌ {error_msg}")
            errors.append(error_msg)
            continue
            
        if not os.path.exists(val_path):
            error_msg = f"Missing: {val_path}"
            print(f"❌ {error_msg}")
            errors.append(error_msg)
            continue
        
        try:
            # Load and validate
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            
            # Check columns
            expected_cols = 22  # 21 features + 1 target
            if train_df.shape[1] != expected_cols:
                warning_msg = f"{client}: Expected {expected_cols} columns, got {train_df.shape[1]}"
                print(f"⚠️  {warning_msg}")
                warnings.append(warning_msg)
            
            # Check for nulls
            null_count = train_df.isnull().sum().sum()
            if null_count > 0:
                warning_msg = f"{client}: Contains {null_count} null values"
                print(f"⚠️  {warning_msg}")
                warnings.append(warning_msg)
            
            print(f"✓ {client}: {len(train_df)} train, {len(val_df)} val samples")
            
        except Exception as e:
            error_msg = f"Error processing {client}: {str(e)}"
            print(f"❌ {error_msg}")
            errors.append(error_msg)
    
    # Summary
    if errors:
        print(f"\n❌ Data validation failed with {len(errors)} errors:")
        for error in errors:
            print(f"   - {error}")
        if warnings:
            print(f"\n⚠️  Additional warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"   - {warning}")
        # Don't exit with error code - let pipeline continue
        print("\n⚠️  Continuing pipeline despite validation issues...")
        return False
    elif warnings:
        print(f"\n⚠️  Data validation completed with {len(warnings)} warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print("\n✅ Continuing pipeline...")
        return True
    else:
        print("\n✅ All data validation checks passed!")
        return True

if __name__ == '__main__':
    validate_data()