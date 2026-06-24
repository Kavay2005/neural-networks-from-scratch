"""
ADDING NON-LINEARITY

Adding Non-Linear function defined under the class to make it modular for every activation function 
(which I am going to update as the time goes on, to also practice the nn's time to time)
"""

import numpy as np


class Activation:
    def __init__(self, act):
        self.act = act.lower()
        self.input_data = 0
        self.output_data = 0

    def __call__(self, x):
        self.input_data = x
        if self.act == 'relu':
            self.output_data = np.maximum(0, self.input_data)
        elif self.act == 'tanh':
            self.output_data = (np.exp(2*self.input_data)-1)/(np.exp(2*self.input_data)+1)
        elif self.act == 'sigmoid':
            self.output_data = 1/(1+np.exp(-self.input_data))
        else:
            raise ValueError(f"Unknown activation function: {self.act}")
        return self.output_data

    def backward(self, dout):
        if self.act == 'relu':
            return dout*(self.input_data > 0)
        elif self.act == 'tanh':
            return dout*(1-self.output_data**2)
        elif self.act == 'sigmoid':
            s = self.output_data
            return dout*s*(1-s)
        else:
            raise ValueError(f"Unknown activation function: {self.act}")
