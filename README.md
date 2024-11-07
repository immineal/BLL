# Machine Learning Mechanisms, Ethics, and Trajectories

This repository contains scripts for visualizations and demonstrations used in the BLL (Besondere Lernleistung) titled "On Mechanisms, Ethical Considerations and Developmental Trajectories in Machine Learning" by Aaron Schneberger and Linus Linhof.

## Overview

Our project explores the intricacies of Machine Learning, focusing on its mechanisms, ethical implications, and future developmental paths. The scripts in this repository implement a simple neural network and generate various visualizations to support our study.

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Pandas
- Keras (keras.datasets and keras.utils)

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Scripts

| File | Description |
|------|-------------|
| `nn.py` | Implements a simple Neural Network for handwritten digit recognition |
| `learning_rate_demonstration.py` | Benchmarks various learning rates to visualize their impact on algorithm efficiency |
| `plot_activation_functions.py` | Generates figures of activation functions discussed in the BLL |
| `work_progress.py` | Creates a cumulative graph of our work progress for the "Description of work progress" chapter |
| `train_and_save_model.py` | Trains and saves a neural network model for handwritten digit recognition |
| `predict_digit.py` | Uses a trained model to predict digits from hand-drawn images |

## How to Train the Model

To train the neural network model, run the `train_and_save_model.py` script. This script uses the MNIST dataset to train a simple neural network and save the model weights to a file named `model_weights.pkl`. The process is broken down as follows:

### Step 1: Load and Preprocess Data

The script automatically loads the MNIST dataset, flattens the images from 28x28 pixels to a 784-dimensional vector, and normalizes the pixel values to be between 0 and 1. It also one-hot encodes the labels (digits 0–9) for the training and test sets.

### Step 2: Train the Model

The `train_and_save_model.py` script defines a simple neural network with one hidden layer and an output layer. It trains the model using batch gradient descent and saves the weights after training is complete. You can adjust the number of epochs, batch size, and learning rate in the script.

To start training, simply execute the script:

```bash
python train_and_save_model.py
```

This will begin the training process and print the loss after each epoch. It will also show the total time taken for training.

### Step 3: Save the Model

Once training is complete, the model weights are saved to the file `model_weights.pkl`. You can then use this file for predictions with the `predict_digit.py` script.

## How to Use the Model for Predictions

To use the trained model for predictions, use the `predict_digit.py` script. This script allows you to input a custom image of a handwritten digit, preprocess it, and predict the digit using the trained neural network.

### Step 1: Prepare the Image

Place an image you want to classify in the same directory as the `predict_digit.py` script. The image should be a grayscale image (e.g., PNG or JPG) containing a handwritten digit. Ensure the image is resized to 28x28 pixels.

### Step 2: Run the Prediction Script

Execute the `predict_digit.py` script to load the trained model and predict the digit in the image:

```bash
python predict_digit.py
```

The script will print the predicted digit and display a bar chart with the probabilities for each digit (0–9).

### Step 3: Interpret the Results

The script will print the predicted digit based on the input image and display the predicted probabilities for each digit. The digit with the highest probability is considered the predicted result.

Example Output:
```
Image values fed to the neural network:
[[ 0  0  0 ...  0  0  0]
 [ 0  0  0 ...  0  0  0]
 [ 0  0  0 ...  0  0  0]
 ...
 [ 0  0  0 ...  0  0  0]]
Predicted digit: 5
```

## How to Demonstrate the Impact of Learning Rates

The `learning_rate_demonstration.py` script demonstrates the impact of different learning rates on the training process by training the neural network on the MNIST dataset with various learning rates. It generates a plot showing how the loss changes with different learning rates.

To run the script and visualize the impact of learning rates:

```bash
python learning_rate_demonstration.py
```

The script will train the model for each learning rate, calculate the loss, and save a plot of the learning rate impact.

Example Output:
```
Epoch 0, Loss: 2.3012, (lr: 0.001)
Epoch 1, Loss: 2.2000, (lr: 0.001)
...
```
After training, a plot will be generated showing the loss at each epoch for each learning rate, and the results will be saved to a CSV file.

## License

This project is licensed under the [MIT License] - see the LICENSE file for details.