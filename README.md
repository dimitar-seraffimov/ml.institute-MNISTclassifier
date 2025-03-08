# MNIST Classifier Model

PyTorch implementation of a Convolutional Neural Network (CNN) for MNIST digit classification.

Run the script `python run_model.py` in the parent directory, it trains the model and generates a visualisation for the prediction of 8 numbers.

Example visualisation (can be found in /results folder after each generation):
![test_predictions](results/test_predictions.png)

# MNIST Digit Classifier

A web application for classifying handwritten digits using a trained convolutional neural network (CNN) model.

The latest version of the web application can be accessed on:

## Features

- Draw digits on a canvas
- Get real-time predictions with confidence scores
- View statistics on prediction accuracy
- Provide feedback by specifying the true digit
- Track performance metrics in a PostgreSQL database
- Visualize confidence distribution across all digits

The model is a Convolutional Neural Network (CNN) with the following architecture:

1. **Convolutional Layers**:

   - First layer: 1 input channel (grayscale) → 32 output channels, 3×3 kernel
   - Second layer: 32 input channels → 64 output channels, 3×3 kernel
   - ReLU activation after each convolutional layer

2. **Pooling & Regularization**:

   - Max pooling with 2×2 kernel
   - Dropout (0.25) after pooling
   - Additional dropout (0.5) before final classification

3. **Fully Connected Layers**:
   - First layer: 64 × 12 × 12 input features → 128 output features
   - Second layer: 128 input features → 10 output features (one for each digit)

## Database Structure

The application uses a PostgreSQL database to store:

- Prediction history (digit, confidence, timestamp)
- True labels (when provided by users)
- Performance statistics

## Project Structure

- `model/`: Neural network model definition and training code
- `streamlit/`: Web application code
- `saved_models/`: Trained model weights
- `results/`: Visualization outputs
- `tests/`: Test scripts to verify functionality

## Technologies Used

- PyTorch for model development and inference
- Streamlit for the web interface
- PostgreSQL for data storage
- Docker for containerization and deployment
- Google Drive for deploying the web application