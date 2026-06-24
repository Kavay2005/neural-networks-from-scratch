"""
Stochastic Gradient Descent Optimizer

It is the fundamental optimizer that updates each and every weight one by one. 
The problem with this is that it is highly inefficient because it does not optimize 
vectorization technique provided by most GPUs.

How does it work?
1. It updates the weight one by one traversing through each layer and each weight and input.
2. If you take standard, pure SGD literally—where it processes the entire dataset one single row at a time—
   it can become incredibly slow. If you have 1,000,000 training examples, updating the weights 1,000,000 times 
   per epoch means a massive amount of overhead.

Intuition behind it:
1. Imagine you are stranded at the top of a foggy mountain in the middle of the night, and your goal is to 
   find the lowest point in the valley. Because of the fog, you cannot see the bottom.

2. What do you do? You look at the ground beneath your feet, feel which way slopes downward, and take a small 
   step in that direction. You repeat this process over and over until the ground flattens out.

    - The Mountain: The loss function (the model's error).
    - The Steps: Adjusting the weights.
    - The Direction: The negative gradient (calculated during backpropagation).
"""

import numpy as np


class SGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.lr = learning_rate

    def step(self):
        """Update all parameters using computed gradients"""
        for param in self.parameters:
            # param is an object containing 'w', 'b', 'dw', 'db' attributes
            param.w -= self.lr * param.dw
            param.b -= self.lr * param.db

    def zero_grad(self):
        """Reset all gradients to zero"""
        for param in self.parameters:
            param.dw = np.zeros_like(param.w)  # Simulate the shape of w and b then make zero vector of same shape
            param.db = np.zeros_like(param.b)
