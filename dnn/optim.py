import numpy as np
import neural_network_func as nn

class SGD:
    def __init__(self, model: nn.Module, initial_learning_rate=0.001, decay_rate=0.0, l2_lambda=0.0):
        self.model = model
        self.initial_learning_rate = initial_learning_rate
        self.lr = initial_learning_rate
        self.decay_rate = decay_rate
        self.l2_lambda = l2_lambda
    
    def step(self):
        for layer in self.model.layers:
            if not layer.requires_grad:
                continue
            grad_weights = layer.grad_weights.copy() + self.l2_lambda * layer.weights
            grad_bias = layer.grad_bias.copy()
            if self.decay_rate:
                grad_weights += layer.grad_weights * self.decay_rate
                grad_bias += layer.grad_bias * self.decay_rate
            layer.weights -= self.lr * grad_weights
            layer.bias -= self.lr * grad_bias