# MNIST Digit Classifier Project

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. It's designed as a comprehensive example of building, training, and deploying a machine learning model.

## Project Structure

```
├── model/                  # PyTorch model training and inference
│   ├── model.py            # Model architecture definition
│   ├── train.py            # Training script
│   ├── inference.py        # Inference utilities
│   ├── test_model.py       # Testing script
│   └── mnist_classifier_demo.ipynb  # Interactive demo notebook
├── app/                    # Streamlit web application (to be implemented)
├── database/               # Database setup (to be implemented)
├── docker/                 # Docker configuration (to be implemented)
├── run_model.py            # Script to run training and testing
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- PIL (Pillow)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ml.institute-MNISTclassifier.git
   cd ml.institute-MNISTclassifier/MNIST-project
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Running the Model

### Training

To train the model:

```bash
python run_model.py --mode train --epochs 10
```

This will:

- Download the MNIST dataset (if not already downloaded)
- Train the model for 10 epochs
- Save the trained model to `model/mnist_classifier.pth`
- Generate training history plots

### Testing

To test the model on sample images:

```bash
python run_model.py --mode test
```

This will:

- Load the trained model
- Make predictions on random test images
- Display and save the results

### Training and Testing

To train and then test the model:

```bash
python run_model.py --mode both --epochs 5
```

### Command-Line Arguments

- `--mode`: Mode of operation (`train`, `test`, or `both`)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--learning-rate`: Learning rate
- `--momentum`: SGD momentum
- `--model-path`: Path to save/load the model
- `--no-cuda`: Disable CUDA training
- `--seed`: Random seed
- `--test-samples`: Number of test samples to visualize

## Interactive Demo

You can also use the Jupyter notebook for an interactive demo:

```bash
jupyter notebook model/mnist_classifier_demo.ipynb
```

This notebook provides:

- Step-by-step explanation of the model
- Interactive training and visualization
- Drawing interface for testing your own digits (if supported by your Jupyter environment)

## Model Architecture

The model is a Convolutional Neural Network (CNN) with:

- 2 convolutional layers with ReLU activation and max pooling
- 2 fully connected layers
- Dropout for regularization

This architecture is chosen for its balance between simplicity and effectiveness for the MNIST classification task.

## Performance

The model typically achieves 98-99% accuracy on the MNIST test set after 5-10 epochs of training.

## Next Steps

Future enhancements to this project will include:

- Streamlit web application for interactive digit recognition
- Database integration for logging predictions
- Docker containerization for easy deployment
- Deployment to a self-managed server

## License

MIT
