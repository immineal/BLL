import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU function
def relu(x):
    return np.maximum(0, x)

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Function to create and save LaTeX formula image
def save_latex_formula(formula, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    ax.text(0.5, 0.5, f"${formula}$", size=30, ha='center')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# Plot Sigmoid with formula
x = np.linspace(-10, 10, 100)
y_sigmoid = sigmoid(x)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y_sigmoid)
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.grid()

sigmoid_formula = r"\sigma(x) = \frac{1}{1 + e^{-x}}"
plt.subplot(1, 2, 2)
plt.text(0.5, 0.5, f"${sigmoid_formula}$", size=30, ha='center')
plt.axis('off')

plt.tight_layout()
plt.savefig('sigmoid_with_formula.png')
plt.close()

# Plot ReLU with formula
y_relu = relu(x)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y_relu)
plt.title("ReLU Function")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.grid()

relu_formula = r"\text{ReLU}(x) = \max(0, x)"
plt.subplot(1, 2, 2)
plt.text(0.5, 0.5, f"${relu_formula}$", size=30, ha='center')
plt.axis('off')

plt.tight_layout()
plt.savefig('relu_with_formula.png')
plt.close()

# Plot Softmax with formula
x_softmax = np.linspace(-2, 2, 100)
y_softmax = softmax(x_softmax)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_softmax, y_softmax)
plt.title("Softmax Function")
plt.xlabel("x")
plt.ylabel("Softmax(x)")
plt.grid()

softmax_formula = r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}"
plt.subplot(1, 2, 2)
plt.text(0.5, 0.5, f"${softmax_formula}$", size=30, ha='center')
plt.axis('off')

plt.tight_layout()
plt.savefig('softmax_with_formula.png')
plt.close()

print("Plots with formulas saved as PNG images.")
