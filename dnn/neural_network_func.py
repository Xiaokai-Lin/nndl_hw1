import numpy as np

class Linear:
    """Linear layer (fully connected layer) implementation.
    """
    def __init__(self, in_features: int, out_features: int):
        # For SGD
        self.requires_grad = True
        
        # Initialize weights with random values
        self.weights = np.random.randn(out_features, in_features)
        
        # Generate a bias (column) vector
        self.bias = np.zeros((out_features, 1))
        # self.bias = np.random.randn(out_features, 1)
        
        self.grad_weights = None
        self.grad_bias = None
    
    def forward(self, x):
        # x: input features (batch_size, in_features)
        self.input = x
        
        # Wx + b
        self.output = np.dot(self.weights, self.input.T) + self.bias
        return self.output.T  # Return output (batch_size, out_features)
    
    def backward(self, grad_output):
        # grad_output: gradient of loss with respect to layer output (batch_size, out_features)
        
        # Update gradients
        self.grad_weights = (grad_output.T @ self.input) / self.input.shape[0]
        self.grad_bias = np.mean(grad_output, axis=0, keepdims=True).T
        
        # Compute gradient of loss with respect to layer input
        grad_input = np.dot(grad_output, self.weights)
        # print(grad_input)
    
        
        return grad_input  # Return gradient of loss with respect to layer input (batch_size, in_features)
    
    def __call__(self, x):
        return self.forward(x)


class SoftMax:
    def __init__(self):
        # For SGD
        self.requires_grad = False
        self.output = None
    
    def forward(self, x):
        # Forward pass of Softmax activation function
        
        # For numerical stability, subtract the maximum value of x from x
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    
    def backward(self, grad_output):
        return grad_output
    
    def __call__(self, x):
        return self.forward(x)
   
class relu:
    def __init__(self):
        # For SGD
        self.requires_grad = False
        self.mask = None
    
    def forward(self, x):
        # Forward pass of ReLU activation function
        self.mask = (x <= 0)  # Create a boolean mask where True indicates x <= 0
        out = x.copy()  # Create a copy of input x
        out[self.mask] = 0  # Set elements where x <= 0 to 0
        return out
    
    def backward(self, grad_output):
        # Backward pass of ReLU activation function
        grad_input = grad_output.copy()  # Copy the gradient of loss with respect to output
        
        # Set gradients to 0 where corresponding elements in mask are True (x <= 0)
        grad_input[self.mask] = 0
        
        return grad_input
    
    def __call__(self, x):
        return self.forward(x)

class Sigmoid:
    def __init__(self):
        # For SGD
        self.requires_grad = False
        self.output = None
    
    def forward(self, x):
        # Forward pass of Sigmoid activation function
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, grad_output):
        # Backward pass of Sigmoid activation function
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input
    
    def __call__(self, x):
        return self.forward(x)


# Loss function
class CrossEntropyLoss:
    def __init__(self, model, l2_lambda: float = 0.0):
        # For SGD
        self.requires_grad = False
        self.predictions = None
        self.targets = None
        self.batch_size = None
        self.num_cls = None
        self.model = model
        self.l2_lambda = l2_lambda
    
    def forward(self, predictions, targets):
        # Forward pass of cross-entropy loss function
        self.predictions = predictions
        self.targets = targets
        self.batch_size = predictions.shape[0]
        self.num_cls = predictions.shape[1]
        # Compute cross-entropy loss
        # num_samples = self.predictions.shape[0]
        # self.batch_size = num_samples
        # exp_scores = np.exp(predictions)
        # softmax_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Compute cross-entropy loss
        cross_entropy = -np.log(predictions[range(self.batch_size), targets] + 1e-7)
        
        # Compute L2 regularization term
        l2_loss = 0.0
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                l2_loss += np.sum(layer.weights ** 2)
        
        self.loss = np.mean(cross_entropy) + 0.5 * self.l2_lambda * l2_loss / self.batch_size
        # print(self.loss)
        return self
    
    def backward(self):
        # Backward pass of cross-entropy loss function
        # num_samples = self.predictions.shape[0]
        grad_input = self.predictions.copy()
        grad_input[range(self.batch_size), self.targets] -= 1
        grad_input /= self.batch_size
        # print(grad_input.shape)
        
        self.model.backward(grad_input)
    
    def item(self):
        return self.loss
    
    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)

class Module:
    def __init__(self):
        self.layers = []

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad_output=None):
        grad_output = grad_output
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        
        # return grad_output

    def __call__(self, x):
        return self.forward(x)
    
