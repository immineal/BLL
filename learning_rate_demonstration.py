import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist # Importing MNIST dataset
from keras.utils import to_categorical
import pandas as pd

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Flatten the images
train_images = train_images.reshape((60000, 28*28))
test_images = test_images.reshape((10000, 28*28))

# Normalize pixel values to be between 0 and 1
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def categorical_crossentropy(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-15)) / m

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights1) + self.bias1
        self.hidden_output = sigmoid(self.hidden_input)
        self.output = softmax(np.dot(self.hidden_output, self.weights2) + self.bias2)
        return self.output
    
    def backward(self, X, y, learning_rate=0.1):
        m = X.shape[0]
        
        # Calculate gradients
        d_output = self.output - y
        d_weights2 = np.dot(self.hidden_output.T, d_output) / m
        d_bias2 = np.sum(d_output, axis=0, keepdims=True) / m
        
        d_hidden = np.dot(d_output, self.weights2.T) * sigmoid_derivative(self.hidden_input)
        d_weights1 = np.dot(X.T, d_hidden) / m
        d_bias1 = np.sum(d_hidden, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.weights2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2
        self.weights1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1
        
    def train(self, X, y, epochs=100, batch_size=64, learning_rate=0.1):
        loss_history = []
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
                
            if epoch % 1 == 0:
                loss = categorical_crossentropy(y_batch, output)
                loss_history.append(loss)
                print(f'Epoch {epoch}, Loss: {loss:.4f}, (lr: {learning_rate})')
        
        return loss_history
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Function to train and plot loss for different learning rates
def plot_learning_rate_impact(learning_rates):
    all_loss_data = {}

    plt.figure(figsize=(12, 6))
    
    for lr in learning_rates:
        nn = NeuralNetwork(input_size, hidden_size, output_size)
        loss_history = nn.train(train_images, train_labels, epochs=50, batch_size=32, learning_rate=lr)
        plt.plot(np.arange(len(loss_history)), loss_history, label=f'Learning Rate: {lr}')
        all_loss_data[f'lr_{lr}'] = loss_history
    
    # Save the loss data to a CSV file
    loss_df = pd.DataFrame(all_loss_data)
    loss_df.to_csv('learning_rate_loss_data.csv', index_label='Epoch')
    
    plt.title('Impact of Learning Rate on Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Cross-Entropy Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('learning_rate_impact_log.png')
    plt.show()

# Define neural network parameters
input_size = 784  # 28x28 pixels
hidden_size = 128
output_size = 10  # 10 classes (0-9 digits)

# Define learning rates to test
learning_rates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Plot and save the impact of different learning rates
plot_learning_rate_impact(learning_rates)