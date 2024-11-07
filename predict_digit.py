import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = None
        self.bias1 = None
        self.weights2 = None
        self.bias2 = None

    def load_model(self, filepath='model_weights.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            self.weights1 = model_data['weights1']
            self.bias1 = model_data['bias1']
            self.weights2 = model_data['weights2']
            self.bias2 = model_data['bias2']
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, X):
        hidden_input = np.dot(X, self.weights1) + self.bias1
        hidden_output = self.sigmoid(hidden_input)
        output = self.softmax(np.dot(hidden_output, self.weights2) + self.bias2)
        return output

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))  # Ensure image is 28x28
    img_data = np.array(img).astype('uint8')  # No normalization here, values are 0 to 255
    print("Image values fed to the neural network:")
    print(img_data)  # Print the image values
    return img_data.reshape(1, 28 * 28)  # Flatten to 784-dim vector

def plot_probabilities(probabilities):
    digits = np.arange(10)  # Digits 0-9
    plt.bar(digits, probabilities[0])
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title('Raw Probabilities for Each Digit')
    plt.xticks(digits)
    plt.show()

if __name__ == "__main__":
    nn = NeuralNetwork(784, 512, 10)
    nn.load_model('model_weights.pkl')

    # Process and predict for a custom image
    image_path = 'predict.png'  # Replace with your image filename
    image_data = preprocess_image(image_path)
    probabilities = nn.forward(image_data)
    print(f"Predicted digit: {np.argmax(probabilities, axis=1)[0]}")
    
    #Plot the probabilities
    plot_probabilities(probabilities)