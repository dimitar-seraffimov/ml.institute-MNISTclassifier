import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
    - 2 Convolutional layers with ReLU activation and max pooling
    - 2 Fully connected layers
    - Dropout for regularisation
    
    This architecture is chosen for its balance between simplicity and effectiveness
    for the MNIST classification task. The convolutional layers extract spatial features
    from the digit images, while the fully connected layers perform the final classification.
    """
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        # first convolutional layer: 1 input channel (grayscale), 32 output channels
        # kernel size 3x3, stride 1, padding 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        
        # second convolutional layer: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
       
        # max pooling layer with kernel size 2x2
        self.pool = nn.MaxPool2d(kernel_size=2)
       
        # dropout layer with 25% probability for regularisation
        self.dropout1 = nn.Dropout2d(0.25)
        
        # first fully connected layer: 64*7*7 input features, 128 output features
        # (7x7 is the size after two 2x2 max pooling operations on a 28x28 image)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # dropout layer with 50% probability
        self.dropout2 = nn.Dropout(0.5)
        
        # Output layer: 128 input features, 10 output features (one for each digit)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # first convolutional block: conv -> relu -> max pooling
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # second convolutional block: conv -> relu -> max pooling
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # dropout for regularisation
        x = self.dropout1(x)
        
        # flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)
        
        # first fully connected layer with ReLU activation
        x = self.fc1(x)
        x = F.relu(x)
        
        # dropout for regularisation
        x = self.dropout2(x)
        
        # Output layer (logits)
        x = self.fc2(x)
        
        # return log softmax probabilities
        output = F.log_softmax(x, dim=1)
        return output 