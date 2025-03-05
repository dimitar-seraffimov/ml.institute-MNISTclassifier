import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    """
    Train the MNIST classifier model.
    
    Args:
        model: the neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: number of training epochs
        lr: learning rate
    
    Returns:
        Trained model and training history
    """

    print(f"--- train_model function in train.py...")


    # define loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=lr)
    
    # training history
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    # training loop
    for epoch in range(1, epochs + 1):
        # put the model in training mode!
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # data and target are set on the correct device in the run_model.py file
            # zero the parameter gradients
            optimiser.zero_grad()
            
            # forward pass
            output = model(data)
            
            # calculate loss
            loss_function = criterion(output, target)
            
            # backward pass propogate the loss and optimise the model
            loss_function.backward()
            optimiser.step()
            
            # accumulate loss
            train_loss += loss_function.item()
            
            # print progress
            if batch_idx % 20 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss_function.item():.6f}')
        
        # average training loss for this epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # evaluation phase for each epoch
        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:                
                # forward pass
                output = model(data)
                
                # calculate loss
                test_loss += criterion(output, target).item()
                
                # get predictions
                pred = output.argmax(dim=1, keepdim=True)
                
                # count correct predictions
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        # average test loss and accuracy
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        accuracy = 100. * correct / len(test_loader.dataset)
        test_accuracies.append(accuracy)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.0f}, '
              f'Test Accuracy: {accuracy:.0f}%')
    
    return model, {'train_losses': train_losses, 
                  'test_losses': test_losses, 
                  'test_accuracies': test_accuracies}

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