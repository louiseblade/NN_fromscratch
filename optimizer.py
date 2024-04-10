import numpy as np

class CustomOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_parameters(self, layers):
        for layer in layers:
            layer.weights -= self.learning_rate * layer.dweights
            layer.biases -= self.learning_rate * layer.dbiases