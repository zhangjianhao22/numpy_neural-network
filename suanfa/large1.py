
# 两层神经网络，输入层784，隐藏层128，输出层10，激活函数relu+softmax

import numpy as np
import struct
from urllib.request import urlretrieve
import os
import gzip


# 下载MNIST数据集
def download_mnist():
    # base_url = 'http://yann.lecun.com/exdb/mnist/'
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    if not os.path.exists('data'):
        os.makedirs('data')

    for file in files:
        if not os.path.exists(f'data/{file}'):
            print(f'下载 {file}...')
            urlretrieve(base_url + file, f'data/{file}')
        else:
            print(f'{file} 已存在')

def load_mnist():
    """加载MNIST数据集"""
    def read_images(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows * cols)
            return images / 255.0  # 归一化到[0, 1]

    def read_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    print("加载MNIST数据集...")
    X_train = read_images('data/train-images-idx3-ubyte.gz')
    y_train = read_labels('data/train-labels-idx1-ubyte.gz')
    X_test = read_images('data/t10k-images-idx3-ubyte.gz')
    y_test = read_labels('data/t10k-labels-idx1-ubyte.gz')

    return X_train, y_train, X_test, y_test

def one_hot_encode(y, num_classes=10):
    """将标签转换为one-hot编码"""
    n = len(y)
    y_one_hot = np.zeros((n, num_classes))
    y_one_hot[np.arange(n), y] = 1
    return y_one_hot

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        初始化两层神经网络
        参数:
            input_size: 输入层[1,784]
            hidden_size: 隐藏层大小128
            output_size: 输出层大小10
            learning_rate: 学习率0.1
        """
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        self.learning_rate = learning_rate

    def relu(self, x):
        return np.maximum(0.0, x)

    def relu_grad(self, x):
        return (x > 0).astype(np.float32)

    def softmax(self, x):
        """Softmax函数"""
        # 减去最大值防止数值不稳定
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """前向传播"""
        # 第一层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        # 第二层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    def backward(self, X, y, output):
        """反向传播"""
        m = X.shape[0]  # 样本数量

        # 计算输出层的误差
        dz2 = output - y  # 使用交叉熵损失时，softmax的梯度简化

        # 计算第二层的梯度
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # 计算隐藏层的误差
        dz1 = np.dot(dz2, self.W2.T) * self.relu_grad(self.a1)


        # 计算第一层的梯度
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # 更新参数
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def compute_loss(self, y_pred, y_true):
        """计算交叉熵损失"""
        m = y_true.shape[0]
        # 添加小值防止log(0)
        log_likelihood = -np.log(y_pred[np.arange(m), y_true.argmax(axis=1)] + 1e-8)
        loss = np.sum(log_likelihood) / m
        return loss

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """训练网络"""
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size

        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            # 打乱数据
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0

            # 小批量训练
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # 前向传播
                output = self.forward(X_batch)

                # 计算损失
                batch_loss = self.compute_loss(output, y_batch)
                epoch_loss += batch_loss * X_batch.shape[0]

                # 反向传播
                self.backward(X_batch, y_batch, output)

            # 计算平均损失
            avg_loss = epoch_loss / n_samples
            train_losses.append(avg_loss)

            # 验证集评估
            val_acc = self.evaluate(X_val, y_val)
            val_accuracies.append(val_acc)

            # if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        return train_losses, val_accuracies

    def predict(self, X):
        """预测"""
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def evaluate(self, X, y):
        """评估准确率"""
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy

def main():
    # 下载并加载数据
    download_mnist()
    X_train, y_train, X_test, y_test = load_mnist()

    # 准备数据
    y_train_onehot = one_hot_encode(y_train)
    y_test_onehot = one_hot_encode(y_test)

    # 划分验证集（从训练集中取10%）
    split_idx = int(0.9 * len(X_train))
    X_train_final, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train_final, y_val = y_train_onehot[:split_idx], y_train_onehot[split_idx:]

    print(f"训练集大小: {X_train_final.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")

    # 创建神经网络
    input_size = 784  # 28x28 pixels
    hidden_size = 128  # 隐藏层神经元数量
    output_size = 10   # 10个数字类别
    learning_rate = 0.1

    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

    # 训练网络
    print("\n开始训练...")
    train_losses, val_accuracies = nn.train(
        X_train_final, y_train_final,
        X_val, y_val,
        epochs=50,
        batch_size=64
    )

    # 测试集评估
    test_accuracy = nn.evaluate(X_test, y_test_onehot)
    print(f"\n最终测试集准确率: {test_accuracy:.4f}")
    print("\n显示测试集前10个样本的预测结果:")
    predictions = nn.predict(X_test[:10])
    true_labels = y_test[:10]

    for i in range(10):
        print(f"样本 {i+1}: 预测类{predictions[i]}, 实际类={true_labels[i]}, {'正确' if predictions[i] == true_labels[i] else '错误'}")

if __name__ == "__main__":
    main()