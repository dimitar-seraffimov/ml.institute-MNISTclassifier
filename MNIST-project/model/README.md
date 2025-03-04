# MNIST Classifier Model

This directory contains the PyTorch implementation of a Convolutional Neural Network (CNN) for MNIST digit classification.

## Files

- `model.py`: Defines the CNN architecture
- `train.py`: Contains the training loop and evaluation functions
- `inference.py`: Provides utilities for making predictions with the trained model
- `test_model.py`: A script to test the model on a small subset of MNIST
- `mnist_classifier_demo.ipynb`: A Jupyter notebook demonstrating the model

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:

1. **Convolutional Layers**:

   - First layer: 1 input channel (grayscale) → 32 output channels, 3x3 kernel
   - Second layer: 32 input channels → 64 output channels, 3x3 kernel
   - Each followed by ReLU activation and 2x2 max pooling

2. **Fully Connected Layers**:

   - First layer: 64 _ 7 _ 7 input features → 128 output features
   - Second layer: 128 input features → 10 output features (one for each digit)

3. **Regularization**:
   - Dropout (25%) after convolutional layers
   - Dropout (50%) after first fully connected layer

## Training Process

The training process in `train.py` includes:

1. Loading the MNIST dataset
2. Training the model using Stochastic Gradient Descent (SGD) with momentum
3. Evaluating the model on a test set
4. Plotting training history
5. Saving the trained model

## Inference

The `inference.py` file provides the `MNISTPredictor` class for making predictions:

1. Loads the trained model
2. Preprocesses input images (resize, normalize)
3. Makes predictions and returns:
   - Predicted digit (0-9)
   - Confidence score
   - Probabilities for all digits

## Testing the Model

You can test the model using the provided `test_model.py` script:

```bash
python test_model.py
```

This will:

1. Train the model on a small subset of MNIST
2. Save the trained model
3. Make predictions on test images
4. Visualize the results

## Using the Jupyter Notebook

The `mnist_classifier_demo.ipynb` notebook provides an interactive demonstration of the model:

1. Model definition and training
2. Visualization of training history
3. Testing on sample images
4. Interactive drawing (if supported by your Jupyter environment)

You can run this notebook locally or on Google Colab.

## Online Resources

You can also use online resources to test and explore MNIST classification:

1. **Google Colab**: Upload the notebook and run it in the cloud

   - [Google Colab](https://colab.research.google.com/)

2. **Kaggle Notebooks**: Explore MNIST datasets and models

   - [Kaggle MNIST](https://www.kaggle.com/c/digit-recognizer)

3. **PyTorch Examples**: Official PyTorch MNIST examples
   - [PyTorch Examples](https://github.com/pytorch/examples/tree/master/mnist)

## Requirements

- PyTorch
- torchvision
- NumPy
- Matplotlib
- PIL (Pillow)
