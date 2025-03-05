import os
import argparse
import torch
from model import MNISTClassifier, train_model, plot_training_history, save_model, MNISTPredictor
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
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for training (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--model-path', type=str, default='saved_models/mnist_classifier.pth',
                        help='Path to save/load the model (default: saved_models/mnist_classifier.pth)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training')
    parser.add_argument('--test-samples', type=int, default=5,
                        help='Number of test samples to visualize (default: 5)')
    return parser.parse_args()

def train(args):
    """Train the model."""
    print(f"--- train function in run_model.py...")
    
    # check if CUDA is available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # define transformations for the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST training dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Load MNIST test dataset
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=1)
    
    # create model and move to device
    model = MNISTClassifier().to(device)
    
    # train model
    print("Starting model training...")
    trained_model, history = train_model(
        model, train_loader, test_loader, 
        epochs=args.epochs, 
        lr=args.learning_rate, 
    )
    
    # plot training history
    plot_training_history(history)
    
    # save model
    save_model(trained_model, args.model_path)
    
    print(f"Training completed successfully! Model saved to {args.model_path}")
    
    return trained_model, history

def test(args):
    """Test the model on sample images."""
    print(f"Testing MNIST classifier using model from {args.model_path}...")
    print("Using random sampling (different samples each run)")
    
    # Load MNIST test dataset with the same transformations used during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # create predictor - this loads the model from the specified path
    predictor = MNISTPredictor(args.model_path)
    
    # select random test images
    indices = np.random.choice(len(test_dataset), args.test_samples, replace=False)
    
    # create a figure for visualization
    fig, axes = plt.subplots(1, args.test_samples, figsize=(args.test_samples * 3, 3))
    
    # if only one sample, make axes iterable
    if args.test_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # get image and label
        image, label = test_dataset[idx]
        
        # make prediction - pass the tensor directly to the predictor
        # the predictor will handle the preprocessing
        result = predictor.predict(image)
        
        # display image (convert to numpy for matplotlib)
        # ensure we're displaying a 2D image without channel dimension
        img_display = image.squeeze().numpy()
        axes[i].imshow(img_display, cmap='gray')
        
        # set title with prediction and ground truth
        pred_digit = result['predicted_digit']
        confidence = result['confidence']
        axes[i].set_title(f"Predicted: {pred_digit} ({confidence:.3f})\nTrue: {label}")
        axes[i].axis('off')
    
    # save the figure
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/test_predictions.png')
    plt.show()
    
    print(f"Test predictions saved to 'results/test_predictions.png'")

def main():
    """Main function."""
    args = parse_args()
    
    # create necessary directories
    os.makedirs('model', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    if args.mode in ['train', 'both']:
        train(args)
    
    if args.mode in ['test', 'both']:
        test(args)

if __name__ == "__main__":
    main() 