import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def train_model(model, train_loader, test_loader, epochs, lr):
    """
    Train the MNIST classifier model.
    
    Args:
        model: the neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: number of training epochs
        lr: learning rate
        device: device to run the model on
    
    Returns:
        Trained model and training history
    """

    # get the device from the model
    device = next(model.parameters()).device
    print(f"--- train_model function in train.py using device from model: {device}")

    # define loss function and optimiser with weight decay
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    # training history
    train_losses = []
    test_losses = []
    test_accuracies = []

    model.train()

    # training loop for each epoch
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            # optimise the model's weights
            optimiser.zero_grad()
            # forward pass
            output = model(data)

            # calculate loss
            loss_function = criterion(output, target)
            # backward pass propogate the loss and optimise the model's weights
            loss_function.backward()
            optimiser.step()

            # accumulate batch loss
            train_loss += loss_function.item()
            
            # print progress
            if batch_idx % 20 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss_function.item():.6f}')
        
        # average training loss for this epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        print(f'--- Evaluation phase for epoch: {epoch} ---')

        # Set model to evaluation mode for testing
        model.eval()
        correct = 0
        test_loss = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                # forward pass
                output = model(data)
                # calculate loss
                test_loss += criterion(output, target).item()
                # get predictions
                pred = output.argmax(dim=1, keepdim=True)
                # count correct predictions
                correct += pred.eq(target.view_as(pred)).sum().item()
            
        # calculate accuracy
        accuracy = 100. * correct  / len(test_loader.dataset)
        test_accuracies.append(accuracy)
        
        # average test loss for this epoch
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'Test Accuracy: {accuracy:.2f}%')

    # return the model and training history
    history = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }
    
    return model, history

# generate and save a training history plot to the model/plots directory
def plot_training_history(history):
    """
    Plot the training history.
    
    Args:
        history: Dictionary containing training history
    """
    plt.figure(figsize=(12, 4))
    
    # plot training and test loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['test_losses'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    
    # plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['test_accuracies'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Test Accuracy')
    
    # save the plot
    os.makedirs('model/plots', exist_ok=True)
    plt.savefig('model/plots/training_history.png')
    plt.close()

def save_model(model, path='saved_models/mnist_classifier.pth'):
    """
    Save the trained model.
    
    Args:
        model: Trained PyTorch model
        path: Path to save the model
    """
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # save the model
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}') 