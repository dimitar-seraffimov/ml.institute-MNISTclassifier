import os
import argparse
import torch
from model import MNISTClassifier, train_model, plot_training_history, save_model, MNISTPredictor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MNIST Classifier Training and Testing')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'],
                        help='Mode: train, test, or both (default: train)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--model-path', type=str, default='saved_models/mnist_classifier.pth',
                        help='Path to save/load the model (default: saved_models/mnist_classifier.pth)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training')
    parser.add_argument('--test-samples', type=int, default=8,
                        help='Number of test samples to visualize (default: 8)')
    return parser.parse_args()

def train(args):
    """Train the model."""
    print(f"--- train function in run_model.py...")
    
    # check if CUDA is available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # create model and move to device
    model = MNISTClassifier().to(device)
    
    # define transformations for the MNIST dataset with much more aggressive augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomAffine(
            degrees=30,  #  rotation up to 30 degrees
            translate=(0.2, 0.2),  #  translation
            scale=(0.7, 1.3),  #  scaling variation
            shear=15  #  shearing
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))  # randomly erase small parts of the image
    ])

    # for test use only normalize without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load MNIST training dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)

    # split training data into train and validation sets
    # use 20% of training data for validation
    validation_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - validation_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, validation_size])

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # load MNIST test dataset - use test_transform without augmentation
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    
    # train model
    print("Starting model training...")
    trained_model, history = train_model(
        model, train_loader, valid_loader,
        epochs=args.epochs, 
        lr=args.learning_rate,
    )
    
    # evaluate on the real test set
    print("Evaluating model on the test set...")
    trained_model.eval()
    correct = 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = trained_model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / len(test_dataset)
    print(f'Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.2f}%')
    
    # plot training history
    plot_training_history(history)
    
    # save model
    save_model(trained_model, args.model_path)
    
    print(f"Training completed successfully! Model saved to {args.model_path}")
    
    return trained_model, history

def test(args):
    """Test the model on sample images."""
    print(f"Testing MNIST classifier using model from {args.model_path}...")
    
    # load MNIST test dataset with the same transformations used during training
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
        
        # convert PyTorch tensor to PIL Image before prediction
        # denormalize the tensor
        inv_normalize = transforms.Compose([
            transforms.Normalize(mean=[-0.1307/0.3081], std=[1/0.3081])
        ])
        denormalized_image = inv_normalize(image)
        
        # convert to PIL Image
        pil_image = transforms.ToPILImage()(denormalized_image)
        
        # make prediction using the PIL Image
        result = predictor.predict(pil_image)
        
        # display image (convert to numpy for matplotlib)
        # ensure we're displaying a 2D image without channel dimension
        img_display = image.squeeze().numpy()
        axes[i].imshow(img_display, cmap='gray')
        
        # set title with prediction and ground truth
        # unpack the tuple returned by predict method
        pred_digit, confidence, _ = result
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