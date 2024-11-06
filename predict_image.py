import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt

# Neural Network Class without training method (same as before)
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = None
        self.bias1 = None
        self.weights2 = None
        self.bias2 = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        hidden_input = np.dot(X, self.weights1) + self.bias1
        hidden_output = self.sigmoid(hidden_input)
        output = self.softmax(np.dot(hidden_output, self.weights2) + self.bias2)
        return output

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def load_model(self, filepath='model_weights.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            self.weights1 = model_data['weights1']
            self.bias1 = model_data['bias1']
            self.weights2 = model_data['weights2']
            self.bias2 = model_data['bias2']
        print(f"Model loaded from {filepath}")

# Preprocess the input image
def preprocess_image(image_path):
    # Load the image in grayscale
    image = Image.open(image_path).convert('L')
    # Resize the image to 28x28
    image = image.resize((28, 28))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Invert the image (MNIST has white digits on black)
    image_array = 255 - image_array  # Invert the colors (if necessary)
    # Normalize the image to 0-1 range
    image_array = image_array.astype('float32') / 255
    # Flatten the image to a 1D array
    image_array = image_array.flatten()
    return image_array

# Visualize the input image, prediction, and probabilities
def visualize_result(image_path, prediction, output_probabilities):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image)
    
    # Create the plot with 2 rows and 1 column
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the image in the first subplot
    ax1.imshow(image_array, cmap='gray')
    ax1.set_title("Input Image")
    ax1.axis('off')

    # Plot the stem diagram in the second subplot for output probabilities
    ax2.stem(np.arange(10), output_probabilities.flatten(), basefmt=" ")
    ax2.set_title(f"Prediction: {prediction[0]}")
    ax2.set_xlabel('Digit Class')
    ax2.set_ylabel('Probability')
    ax2.set_xticks(np.arange(10))
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

# Load the trained neural network model
nn = NeuralNetwork(784, 128, 10)
nn.load_model('model_weights.pkl')

# Set the path to your image file here
image_path = 'predict.png'  # Replace with the path to the image file you want to test

# Preprocess the image
image_data = preprocess_image(image_path)

# Make a prediction
prediction = nn.predict(image_data)

# Get the raw output probabilities for debugging
output_probabilities = nn.forward(image_data)

# Visualize the input image, the prediction, and the output probabilities
visualize_result(image_path, prediction, output_probabilities)

# Show which digit the model thought was most probable
print(f"The predicted digit for the image is: {prediction[0]}")
