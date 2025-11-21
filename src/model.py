import tensorflow as tf
from tensorflow import keras

class DiabetesModel:
    """Neural network for diabetes prediction"""
    
    @staticmethod
    def create_model(input_dim=21, compile_model=True):
        """
        Create a simple feedforward neural network
        """
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        if compile_model:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
        
        return model
    
    @staticmethod
    def get_weights(model):
        """Extract model weights"""
        return model.get_weights()
    
    @staticmethod
    def set_weights(model, weights):
        """Set model weights"""
        model.set_weights(weights)