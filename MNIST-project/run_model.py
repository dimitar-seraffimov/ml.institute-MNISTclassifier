import os
import argparse
import torch
from model.model import MNISTClassifier
from model.train import train_model, plot_training_history, save_model
from model.inference import MNISTPredictor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MNIST Classifier Training and Testing')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'],
                        help='Mode: train, test, or both (default: train)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--model-path', type=str, default='saved_models/mnist_classifier.pth',
                        help='Path to save/load the model (default: saved_models/mnist_classifier.pth)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--test-samples', type=int, default=5,
                        help='Number of test samples to visualize (default: 5)')
    return parser.parse_args()

def train(args):
    """Train the model."""
    print(f"Training MNIST classifier for {args.epochs} epochs...")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Check if CUDA is available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Define transformations for the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST training dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Load MNIST test dataset
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # Create model and move to device
    model = MNISTClassifier().to(device)
    
    # Train model
    print("Starting model training...")
    trained_model, history = train_model(
        model, train_loader, test_loader, 
        epochs=args.epochs, 
        lr=args.learning_rate, 
        momentum=args.momentum
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    save_model(trained_model, args.model_path)
    
    print(f"Training completed successfully! Model saved to {args.model_path}")
    
    return trained_model, history

def test(args):
    """Test the model on sample images."""
    print(f"Testing MNIST classifier using model from {args.model_path}...")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Define transformations for the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST test dataset
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Create predictor
    predictor = MNISTPredictor(args.model_path)
    
    # Select random test images
    indices = np.random.choice(len(test_dataset), args.test_samples, replace=False)
    
    # Create a figure for visualization
    fig, axes = plt.subplots(1, args.test_samples, figsize=(args.test_samples * 3, 3))
    
    # If only one sample, make axes iterable
    if args.test_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Get image and label
        image, label = test_dataset[idx]
        
        # Make prediction
        result = predictor.predict(image.squeeze().numpy())
        
        # Display image
        axes[i].imshow(image.squeeze().numpy(), cmap='gray')
        
        # Set title with prediction and ground truth
        pred_digit = result['predicted_digit']
        confidence = result['confidence']
        axes[i].set_title(f"Pred: {pred_digit} ({confidence:.2f})\nTrue: {label}")
        axes[i].axis('off')
    
    # Save the figure
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/test_predictions.png')
    plt.show()
    
    print(f"Test predictions saved to 'results/test_predictions.png'")

def main():
    """Main function."""
    args = parse_args()
    
    # Create necessary directories
    os.makedirs('model', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    if args.mode in ['train', 'both']:
        train(args)
    
    if args.mode in ['test', 'both']:
        test(args)

if __name__ == "__main__":
    main() 