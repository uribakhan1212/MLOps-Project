import pandas as pd
import numpy as np
import tensorflow as tf
from src.model import DiabetesModel

class FederatedClient:
    """
    Represents a single client (hospital/clinic) in federated learning
    """
    
    def __init__(self, client_id, data_path):
        """
        Initialize client
        
        Args:
            client_id: Unique identifier (e.g., 'client_1')
            data_path: Path to client's data directory
        """
        self.client_id = client_id
        self.data_path = data_path
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
        print(f"✓ Initialized {client_id}")
    
    def load_data(self):
        """Load client's local data"""
        train_df = pd.read_csv(f"{self.data_path}/train_data.csv")
        val_df = pd.read_csv(f"{self.data_path}/val_data.csv")
        
        # Separate features and labels
        self.X_train = train_df.drop('Diabetes_binary', axis=1).values
        self.y_train = train_df['Diabetes_binary'].values
        self.X_val = val_df.drop('Diabetes_binary', axis=1).values
        self.y_val = val_df['Diabetes_binary'].values
        
        print(f"{self.client_id}: Loaded {len(self.X_train)} training samples")
        return True
    
    def initialize_model(self, global_weights=None):
        """
        Initialize or update model with global weights
        
        Args:
            global_weights: Weights from server (None for first round)
        """
        input_dim = self.X_train.shape[1]
        self.model = DiabetesModel.create_model(input_dim)
        
        if global_weights is not None:
            self.model.set_weights(global_weights)
            print(f"{self.client_id}: Updated model with global weights")
        else:
            print(f"{self.client_id}: Initialized new model")
    
    def train(self, epochs=5, batch_size=32):
        """
        Train model on local data
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            training_history: Dictionary with training metrics
        """
        print(f"\n{self.client_id}: Starting local training...")
        
        # Train on local data
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0  # Set to 1 to see training progress
        )
        
        # Evaluate on validation set
        val_loss, val_acc, val_auc = self.model.evaluate(
            self.X_val, self.y_val, verbose=0
        )
        
        print(f"{self.client_id}: Training complete")
        print(f"  → Val Loss: {val_loss:.4f}")
        print(f"  → Val Accuracy: {val_acc:.4f}")
        print(f"  → Val AUC: {val_auc:.4f}")
        
        return {
            'loss': history.history['loss'],
            'accuracy': history.history['accuracy'],
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_auc': val_auc
        }
    
    def get_weights(self):
        """Get current model weights"""
        return self.model.get_weights()
    
    def get_num_samples(self):
        """Return number of training samples"""
        return len(self.X_train)