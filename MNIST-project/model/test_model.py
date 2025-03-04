import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from model import MNISTClassifier
from train import train_model
from inference import MNISTPredictor

def test_training():
    """
    Test the model training process on a small subset of MNIST.
    """
    print("Testing model training...")
    
    # Define transformations for the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load a small subset of MNIST for quick testing
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # Use only 1000 samples for quick testing
    train_subset = torch.utils.data.Subset(train_dataset, range(1000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    # Use only 200 samples for quick testing
    test_subset = torch.utils.data.Subset(test_dataset, range(200))
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64)
    
    # Create model
    model = MNISTClassifier()
    
    # Train model for just 2 epochs for quick testing
    print("Starting quick model training (2 epochs)...")
    trained_model, history = train_model(model, train_loader, test_loader, epochs=2)
    
    # Save model
    os.makedirs('model/test_output', exist_ok=True)
    torch.save(trained_model.state_dict(), 'model/test_output/test_mnist_classifier.pth')
    
    print(f"Final test accuracy: {history['test_accuracies'][-1]:.2f}%")
    return trained_model, history

def test_inference(model_path='model/test_output/test_mnist_classifier.pth'):
    """
    Test the model inference on sample MNIST images.
    """
    print("\nTesting model inference...")
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Create predictor
    predictor = MNISTPredictor(model_path)
    
    # Select 5 random test images
    indices = np.random.choice(len(test_dataset), 5, replace=False)
    
    # Create a figure for visualization
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
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
    plt.tight_layout()
    plt.savefig('model/test_output/test_predictions.png')
    print(f"Test predictions saved to 'model/test_output/test_predictions.png'")
    
    return fig

def main():
    """
    Main function to test the MNIST classifier.
    """
    # Test training
    trained_model, history = test_training()
    
    # Test inference
    test_inference()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 