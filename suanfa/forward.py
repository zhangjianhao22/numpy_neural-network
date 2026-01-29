import numpy as np

def forward_pass(X, W1, b1, W2, b2):
    # 第一层：线性变换 + ReLU激活
    Z1 = X.dot(W1) + b1  # 线性输出
    A1 = np.maximum(0, Z1)  # ReLU激活输出

    # 第二层：线性变换 + Sigmoid激活
    Z2 = A1.dot(W2) + b2  # 线性输出
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid输出

    return Z1, A1, Z2, A2