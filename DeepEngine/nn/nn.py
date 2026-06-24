"""
CONNECTING LAYERS

Abstracting every layer above inside the Neural Network layer so that it acts like an orchestrator 
and helps creating the forward pass and backpropagation
"""

import numpy as np
from tqdm import tqdm

from layers.layer import Linear
from activation.activation import Activation
from Function.functions import Function, crossentropy
from optimizer.optimizer import SGD


class NeuralNetwork:
    """
    Neural Network that orchestrates layers, activations, and training
    """
    
    def __init__(self, layer_dims, activation_type):
        """
        Initialize neural network with given layer dimensions and activation type
        
        Args:
            layer_dims: List of integers specifying the number of neurons in each layer
            activation_type: Type of activation function ('relu', 'tanh', 'sigmoid')
        """
        self.activation_type = activation_type
        self.modules = []
        
        # Build the network by connecting layers with activations
        for i in range(len(layer_dims) - 1):
            linear_layer = Linear(layer_dims[i], layer_dims[i+1])
            self.modules.append(linear_layer)
            
            # Add activation function to all layers except the last one
            if i < len(layer_dims) - 2 and activation_type in ['relu', 'tanh', 'sigmoid']:
                self.modules.append(Activation(activation_type))

    def _feed_forward(self, input_data):
        """Forward pass through all modules"""
        out = input_data
        for module in self.modules:
            out = module(out)
        return out

    def _backward(self, grad):
        """Backward pass through all modules in reverse order"""
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def parameters(self):
        """Get all trainable parameters (Linear layers)"""
        params = []
        for module in self.modules:
            if isinstance(module, Linear):
                params.append(module)
        return params

    def compile(self, optimizer, loss, metrics):
        """Placeholder for compilation (currently not implemented)"""
        pass

    def fit(self, x, y, epochs, learning_rate, batch_size, dataset=None, num_classes=10):
        """
        Train the neural network
        
        Args:
            x: Training input features
            y: Training labels
            epochs: Number of training epochs
            learning_rate: Learning rate for SGD optimizer
            batch_size: Size of mini-batches
            dataset: Dataset name (optional)
            num_classes: Number of output classes
        """
        # Optimizer should be created ONCE, before the epoch loop
        optimizer_instance = SGD(self.parameters(), learning_rate)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs} :")
            total_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for batch_start in tqdm(range(0, len(x), batch_size)):
                batch_end = batch_start + batch_size
                batch_x = x.iloc[batch_start:batch_end].values
                batch_y = y.iloc[batch_start:batch_end].values

                # Forward pass
                logits = self._feed_forward(batch_x)

                # Softmax
                fn = Function()
                prob = fn.softmax(logits)

                # One-hot encode true labels
                one_hot_y = np.zeros((len(batch_y), num_classes))
                one_hot_y[np.arange(len(batch_y)), batch_y.flatten()] = 1.0

                # Calculate loss
                batch_y_reshaped_for_loss = batch_y.reshape(-1, 1)
                current_loss = crossentropy(prob, batch_y_reshaped_for_loss)
                total_loss += current_loss
                num_batches += 1

                # Backward pass
                grad = (np.array(prob) - one_hot_y) / batch_size
                self._backward(grad)

                # Update weights
                optimizer_instance.step()
                optimizer_instance.zero_grad()

            avg_loss = total_loss / num_batches
            print(f"Loss: {avg_loss:.4f}")
