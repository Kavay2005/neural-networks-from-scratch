
Architecture:
784 → 128 → 64 → 32 → 10

Observation:
Forward pass output shape:
(32,10)

Result:
Feed forward network is Working.

Back Propogation:
Gradients for different layers using similar appraoch to micrograd and wrapping them in classes for modular design.

Loss calculation:
Using Cross Entropy and sfotmax layer loss has been calculated and trvaersed back via back-propogation.

Model fitting: 
Neural Network class wraps the whole data and helps model fitting just like Tesnorflow.

Result: 
Current Result runs the training loop and uses Stochastic gradient descent for optimization and trains on actual data
But the problem is: the model becomes stagnant after 2-4 epochs and does not reduce loss anymore
Yet to find about the reason .