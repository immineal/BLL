# Machine Learning Mechanisms, Ethics, and Trajectories

This repository contains scripts for visualizations and demonstrations used in the BLL (Besondere Lernleistung) titled "On Mechanisms, Ethical Considerations and Developmental Trajectories in Machine Learning" by Aaron Schneberger and Linus Linhof.

## Overview

Our project explores the intricacies of Machine Learning, focusing on its mechanisms, ethical implications, and future developmental paths. The scripts in this repository implement a simple neural network and generate various visualizations to support our study.

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Pandas
- Keras
- Pillow

## Scripts

| File                        | Description |
|-----------------------------|-------------|
| `train_and_save_model.py`    | Implements a neural network to train on the MNIST dataset and save model weights to a file (`model_weights.pkl`) |
| `predict_digit.py`           | Loads a trained model (`model_weights.pkl`) and predicts a digit from an image file |
| `learning_rate_demonstration.py` | Benchmarks various learning rates to visualize their impact on training efficiency and loss |
| `plot_activation_functions.py` | Generates figures of activation functions discussed in the BLL |
| `work_progress.py`           | Creates a cumulative graph of our work progress for the "Description of work progress" chapter |

## Model Details

- The neural network in this project is trained on the MNIST dataset, which consists of images of handwritten digits (0-9).
- The model achieves good accuracy for most digits but has lower performance on 7s and 9s.
- The model weights are saved in `model_weights.pkl`.

## License

This project is licensed under the [MIT License] - see the [LICENSE](LICENSE) file for details.
