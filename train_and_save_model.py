import numpy as np
import pickle
from keras.datasets import mnist
from keras.utils import to_categorical

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.bias2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights1) + self.bias1
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = self.softmax(np.dot(self.hidden_output, self.weights2) + self.bias2)
        return self.output

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        d_output = self.output - y
        d_weights2 = np.dot(self.hidden_output.T, d_output) / m
        d_bias2 = np.sum(d_output, axis=0, keepdims=True) / m
        d_hidden = np.dot(d_output, self.weights2.T) * self.sigmoid_derivative(self.hidden_input)
        d_weights1 = np.dot(X.T, d_hidden) / m
        d_bias1 = np.sum(d_hidden, axis=0, keepdims=True) / m
        self.weights2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2
        self.weights1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1

    def train(self, X, y, epochs=100, batch_size=64, learning_rate=0.01):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            if epoch % 10 == 0:
                loss = -np.sum(y_batch * np.log(self.output + 1e-15)) / y_batch.shape[0]
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def save_model(self, filepath='model_weights.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'weights1': self.weights1,
                'bias1': self.bias1,
                'weights2': self.weights2,
                'bias2': self.bias2
            }, f)
        print(f"Model saved to {filepath}")

# Train the network and save the model once
nn = NeuralNetwork(784, 512, 10)
nn.train(train_images, train_labels, epochs=100, batch_size=64, learning_rate=0.001)
nn.save_model('model_weights.pkl')

# After training your model, evaluate it on the test set
test_predictions = nn.predict(test_images)
test_accuracy = np.mean(test_predictions == np.argmax(test_labels, axis=1))
print(f"Test accuracy: {test_accuracy * 100:.2f}%")