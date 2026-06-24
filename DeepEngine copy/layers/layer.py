"""
DEFINING LAYERS OF FEED FORWARD NEURAL NETWORK

Right now I have only defined one class which is the Linear class but as the time goes on 
I am going to simulate entire Autograd engine to have clear understanding of pytorch.
"""

import numpy as np
import random


class Linear:
    def __init__(self, nin, nout):
        self.grad = 0
        # Corrected weight initialization: (nin, nout) for x @ w
        self.w = np.array([[np.random.randn()*0.01 for _ in range(nout)] for _ in range(nin)])
        # Bias should be (1, nout) to broadcast correctly
        self.b = np.array([[np.random.randn()*0.01 for _ in range(nout)]])
        self.x_input = None  # Store input for backward pass

    def __call__(self, x):
        # Store the input for use in the backward pass
        self.x_input = x
        # x: (batch_size, nin)
        # self.w: (nin, nout)
        # self.b: (1, nout) -> will convert to (batch_size, nout)
        out = x @ self.w + self.b
        return out

    def backward(self, dout):
        self.dw = self.x_input.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)  # Sum over batch dimension, keep dims for broadcasting
        dx = dout @ self.w.T
        return dx
