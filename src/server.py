import numpy as np
from src.model import DiabetesModel

class FederatedServer:
    """
    Central server for federated learning
    Aggregates model updates from clients
    """
    
    def __init__(self):
        """Initialize server"""
        self.global_model = None
        self.global_weights = None
        self.round_number = 0
        
        print("✓ Federated Server initialized")
    
    def initialize_global_model(self, input_dim=21):
        """Create initial global model"""
        self.global_model = DiabetesModel.create_model(input_dim)
        self.global_weights = self.global_model.get_weights()
        print("✓ Global model initialized")
    
    def aggregate_weights(self, client_weights_list, client_sample_sizes):
        """
        Federated Averaging (FedAvg) algorithm
        
        Args:
            client_weights_list: List of weight arrays from each client
            client_sample_sizes: List of sample counts from each client
        
        Returns:
            aggregated_weights: New global model weights
        """
        print(f"\n🔄 Aggregating weights from {len(client_weights_list)} clients...")
        
        # Calculate total samples
        total_samples = sum(client_sample_sizes)
        
        # Weighted average based on number of samples
        aggregated_weights = []
        
        # For each layer's weights
        for layer_idx in range(len(client_weights_list[0])):
            # Initialize with zeros
            layer_weights = np.zeros_like(client_weights_list[0][layer_idx])
            
            # Weighted sum
            for client_idx, client_weights in enumerate(client_weights_list):
                weight = client_sample_sizes[client_idx] / total_samples
                layer_weights += weight * client_weights[layer_idx]
            
            aggregated_weights.append(layer_weights)
        
        self.global_weights = aggregated_weights
        self.global_model.set_weights(aggregated_weights)
        self.round_number += 1
        
        print(f"✓ Aggregation complete - Round {self.round_number}")
        
        return aggregated_weights
    
    def get_global_weights(self):
        """Return current global weights"""
        return self.global_weights
    
    def evaluate_global_model(self, test_data, test_labels):
        """Evaluate global model on test set"""
        loss, acc, auc = self.global_model.evaluate(
            test_data, test_labels, verbose=0
        )
        
        print(f"\n📊 Global Model Performance (Round {self.round_number}):")
        print(f"  → Test Loss: {loss:.4f}")
        print(f"  → Test Accuracy: {acc:.4f}")
        print(f"  → Test AUC: {auc:.4f}")
        
        return {'loss': loss, 'accuracy': acc, 'auc': auc}