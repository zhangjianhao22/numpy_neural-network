import numpy as np

def backward(self, X, y, output):
    m = X.shape[0]
    dz2 = output - y
    dW2 = np.dot(self.a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = np.dot(dz2, self.W2.T) * self.relu_grad(self.a1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    self.W1 = self.W1 - self.learning_rate * dW1
    self.b1 = self.b1 - self.learning_rate * db1
    self.W2 = self.W2 - self.learning_rate * dW2
    self.b2 = self.b2 - self.learning_rate * db2
