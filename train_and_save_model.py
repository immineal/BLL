import numpy as np
import pickle
from keras.datasets import mnist
from keras.utils import to_categorical
from concurrent.futures import ThreadPoolExecutor

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
        hidden_input = np.dot(X, self.weights1) + self.bias1
        hidden_output = self.sigmoid(hidden_input)
        output = self.softmax(np.dot(hidden_output, self.weights2) + self.bias2)
        return hidden_input, hidden_output, output

    def backward(self, X, y, hidden_input, hidden_output, output, learning_rate=0.01):
        m = X.shape[0]
        d_output = output - y
        d_weights2 = np.dot(hidden_output.T, d_output) / m
        d_bias2 = np.sum(d_output, axis=0, keepdims=True) / m
        d_hidden = np.dot(d_output, self.weights2.T) * self.sigmoid_derivative(hidden_input)
        d_weights1 = np.dot(X.T, d_hidden) / m
        d_bias1 = np.sum(d_hidden, axis=0, keepdims=True) / m
        self.weights2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2
        self.weights1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1

    def train_batch(self, X_batch, y_batch, learning_rate):
        hidden_input, hidden_output, output = self.forward(X_batch)
        self.backward(X_batch, y_batch, hidden_input, hidden_output, output, learning_rate)

    def train(self, X, y, epochs=50, batch_size=64, learning_rate=0.01, num_threads=16):
        for epoch in range(epochs):
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Split data into batches and submit each batch to a separate thread
                futures = []
                for i in range(0, X.shape[0], batch_size):
                    X_batch = X[i:i+batch_size]
                    y_batch = y[i:i+batch_size]
                    futures.append(executor.submit(self.train_batch, X_batch, y_batch, learning_rate))
                
                # Wait for all threads in the epoch to complete
                for future in futures:
                    future.result()
            
            if epoch % 10 == 0:
                _, _, output = self.forward(X)  # Use the entire dataset to compute loss
                loss = -np.sum(y * np.log(output + 1e-15)) / y.shape[0]
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
    def predict(self, X):
        _, _, output = self.forward(X)
        return np.argmax(output, axis=1)

    def save_model(self, filepath='model_weights.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'weights1': self.weights1,
                'bias1': self.bias1,
                'weights2': self.weights2,
                'bias2': self.bias2
            }, f)
        print(f"Model saved to {filepath}")

if __name__ == "__main__":
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Initialize and train the neural network with multithreading
    nn = NeuralNetwork(784, 256, 10)
    nn.train(train_images, train_labels, epochs=100, batch_size=64, learning_rate=0.01, num_threads=16)
    nn.save_model('model_weights.pkl')

    # Evaluate the model on the test set
    test_predictions = nn.predict(test_images)
    test_accuracy = np.mean(test_predictions == np.argmax(test_labels, axis=1))
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
