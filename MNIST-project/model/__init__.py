# Import key classes and functions to make them available when importing the package
from .model import MNISTClassifier
from .train import train_model, plot_training_history, save_model
from .inference import MNISTPredictor

# This allows users to do:
# from model import MNISTClassifier, train_model, MNISTPredictor
