# MNIST Digit Classifier

A simple end-to-end application for classifying handwritten digits using the MNIST dataset.

## Project Overview

This project implements a complete MNIST digit classifier with the following components:

1. **PyTorch Model**: A convolutional neural network trained on the MNIST dataset
2. **Streamlit Web Interface**: Interactive web app for drawing digits and getting predictions
3. **PostgreSQL Database**: Logs predictions and user feedback
4. **Docker Containerization**: Easy deployment with Docker Compose

## Project Structure

```
├── model/                  # PyTorch model architecture
├── streamlit/              # Streamlit web application
│   └── app.py              # Main Streamlit application
├── database/               # Database setup
│   └── init.sql            # SQL initialization script
├── saved_models/           # Directory for saved model weights
│   └── mnist_classifier.pth # Trained model weights
├── train_model.py          # Script to train the model
├── run_streamlit.py        # Script to run the Streamlit app
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Dockerfile for the application
└── requirements.txt        # Python dependencies
```

## Quick Start

### Option 1: Using Docker (Recommended)

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/mnist-classifier.git
   cd mnist-classifier
   ```

2. Build and start the containers:

   ```bash
   docker-compose up -d
   ```

3. Access the web application at http://localhost:8501

### Option 2: Local Development

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/mnist-classifier.git
   cd mnist-classifier
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Train the model (optional, a pre-trained model is included):

   ```bash
   python train_model.py
   ```

4. Set up the PostgreSQL database:

   - Install PostgreSQL if not already installed
   - Create a database named `mnist_db`
   - Run the initialization script: `psql -U postgres -d mnist_db -f database/init.sql`

5. Create a `.env` file with the following content:

   ```
   MODEL_PATH=saved_models/mnist_classifier.pth
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=mnist_db
   DB_USER=postgres
   DB_PASSWORD=postgres
   ```

6. Run the Streamlit app:
   ```bash
   python run_streamlit.py
   ```

## Features

### 1. Drawing Interface

- Interactive grid for drawing digits
- Clear button to reset the canvas

### 2. Prediction

- Real-time prediction of drawn digits
- Confidence scores for each digit class
- Visualization of prediction probabilities

### 3. Feedback System

- Option to provide the correct label for incorrect predictions
- Logging of user feedback to improve model evaluation

### 4. Statistics Dashboard

- Overall accuracy statistics
- Digit-specific performance metrics
- Recent prediction history

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:

1. **Convolutional Layers**:

   - First layer: 1 input channel → 10 output channels, 5x5 kernel
   - Second layer: 10 input channels → 20 output channels, 5x5 kernel
   - Each followed by ReLU activation and 2x2 max pooling

2. **Fully Connected Layers**:

   - First layer: 320 input features → 50 output features
   - Second layer: 50 input features → 10 output features (one for each digit)

3. **Regularization**:
   - Dropout after second convolutional layer
   - Dropout after first fully connected layer

## Database Schema

The PostgreSQL database includes:

1. **predictions table**:

   - Stores prediction details (timestamp, predicted digit, confidence, true label)
   - Optionally stores the image data

2. **Views for statistics**:
   - `prediction_accuracy`: Overall accuracy metrics
   - `digit_statistics`: Per-digit performance metrics

## Docker Setup

The Docker setup includes:

1. **Web Application Container**:

   - Python 3.9 with PyTorch and Streamlit
   - Runs the Streamlit application

2. **PostgreSQL Container**:
   - Stores prediction data
   - Initialized with the schema from `database/init.sql`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```bash
# Train the model
python run_model.py --mode train

# Test the model
python run_model.py --mode test --test-samples 8

# Both train and test
python run_model.py --mode both
```

## Files

- `model.py`: defines the CNN architecture
- `train.py`: contains the training loop and evaluation functions
- `inference.py`: provides utilities for making predictions with the trained model
- `mnist_classifier.ipynb`: Jupyter notebook demonstrating the model

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:

1. **Convolutional Layers**:

   - First layer: 1 input channel (grayscale) → 10 output channels, 5x5 kernel
   - Second layer: 10 input channels → 20 output channels, 5x5 kernel
   - Each followed by ReLU activation and 2x2 max pooling

2. **Fully Connected Layers**:

   - First layer: 320 input features → 50 output features
   - Second layer: 50 input features → 10 output features (one for each digit)

3. **Regularization**:
   - Dropout after second convolutional layer
   - Dropout after first fully connected layer

## Training Process

The training process in `train.py` includes:

1. Using pre-loaded MNIST dataset via DataLoaders
2. Training the model using Adam optimizer with Cross Entropy Loss
3. Evaluating the model on a test set after each epoch
4. Tracking and plotting training history (losses and accuracy)
5. Saving the trained model

## Inference

The `inference.py` file provides the `MNISTPredictor` class for making predictions:

1. Loads the trained model
2. Preprocesses input images (resize, normalize)
3. Handles different input formats (PIL Image, numpy array, or PyTorch tensor)
4. Makes predictions and returns:
   - Predicted digit (0-9)
   - Confidence score
   - Probabilities for all digits

## Using the Model

You can use the model in several ways:

### 1. Using run_model.py

The main script `run_model.py` in the parent directory provides a command-line interface:

```bash
# Train the model
python run_model.py --mode train

# Test the model
python run_model.py --mode test --test-samples 8

# Both train and test
python run_model.py --mode both
```

Command line options:

- `--mode`: train, test, or both
- `--epochs`: number of training epochs
- `--batch-size`: batch size for training
- `--learning-rate`: learning rate for optimizer
- `--model-path`: path to save/load the model
- `--no-cuda`: disable CUDA training
- `--test-samples`: number of test samples to visualize

### 2. Using the Jupyter Notebook

The `mnist_classifier.ipynb` notebook provides an interactive demonstration of the model:

1. Model definition and training
2. Visualization of training history
3. Testing on sample images
4. Visualization of prediction probabilities

## Requirements

- PyTorch
- torchvision
- NumPy
- Matplotlib
- PIL (Pillow)

These dependencies are listed in the `requirements.txt` file in the parent directory.
