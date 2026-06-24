"""
Function utilities - Softmax and Cross Entropy Loss
"""

import numpy as np


class Function:
    def __init__(self):
        self.probs = None
        self.out = []

    def softmax(self, logits):
        for _ in logits:
            self.probs = np.exp(_) / np.sum(np.exp(_))
            self.out.append(self.probs)
        return self.out


def crossentropy(x, y):
    """
    Calculate cross entropy loss
    
    Args:
        x: list of probability arrays (output from softmax)
        y: 2D numpy array of true labels (matrix)
    
    Returns:
        Mean cross entropy loss
    """
    correct_class_probs = []
    for k in range(len(x)):  # Iterate over each sample in the batch
        predicted_probs = x[k]  # This is a (10,) array of probabilities for one sample
        true_label_for_sample = y[k][0]  # y[k] is like array([idx]), so y[k][0] gives the integer index

        # Get the probability for the true class
        prob_of_true_class = predicted_probs[true_label_for_sample]
        correct_class_probs.append(prob_of_true_class)

    # Convert to numpy array for vectorized log operation
    correct_probs_np = np.array(correct_class_probs)

    # Calculate the negative log likelihood and then the mean
    loss_values = -np.log(correct_probs_np)  # loss= -log(p)
    return np.mean(loss_values)
