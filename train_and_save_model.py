import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist # Importing MNIST dataset
from keras.utils import to_categorical

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
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
                
            if epoch % 10 == 0:
                loss = categorical_crossentropy(y_batch, output)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Initialize the neural network
input_size = 784  # 28x28 pixels
hidden_size = 128
output_size = 10  # 10 classes (0-9 digits)
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the neural network
nn.train(train_images, train_labels, epochs=100, batch_size=64, learning_rate=0.1)

# Evaluate on test data
predictions = nn.predict(test_images)
accuracy = np.mean(predictions == np.argmax(test_labels, axis=1))
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Display some predictions
for i in range(100):
    idx = np.random.randint(0, test_images.shape[0])
    image = test_images[idx].reshape((28, 28))
    pred_label = predictions[idx]
    true_label = np.argmax(test_labels[idx])
    
    plt.subplot(2, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted: {pred_label}, True: {true_label}')
    plt.axis('off')

plt.tight_layout()
plt.show()