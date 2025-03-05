"""
MNIST Classifier Model Package

This package contains modules for training, evaluating, and using
a convolutional neural network for MNIST digit classification.
"""

# Import and expose the MNISTClassifier class
from .model import MNISTClassifier

# Import and expose training functions
from .train import train_model, plot_training_history, save_model

# Import and expose inference functions
from .inference import MNISTPredictor

# Define package version
__version__ = '0.1.0'
