import torch.nn as nn
import torch.nn.functional as F
import torch

class MNISTClassifier(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
    - 2 Convolutional layers with ReLU activation and max pooling
    - 2 Fully connected layers
    - Dropout for regularisation
    
    """
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        # convolutional layer 1
        # input channel - 1 (grayscale image)
        # output channel - 32
        # kernel size - 3
        # stride - 1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # convolutional layer 2
        # input channel - 32
        # output channel - 64
        # kernel size - 3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # fully connected layer 1
        # formula for calculating the input size of the fully connected layer:
        # (W - F + P) / S
        # W = width of the image
        # F = filter size
        # P = padding
        # S = stride
        # for this case, the formula looks like: (28 - 3 + 2*0) / 1 = 24
        # the image size is now 24x24 (24/2/2), so after the max pooling layers, the image size is 12x12
        # that's why we have 64 * 12 * 12

        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        # fully connected layer 2
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = F.relu(x)
        # conv2
        x = self.conv2(x)
        x = F.relu(x)

        # max pooling to reduce the size of the image
        x = F.max_pool2d(x, 2)
        # dropout to regularize the model
        x = self.dropout1(x)
        # flatten the output of the convolutional layers
        x = torch.flatten(x, 1)
        
        # fully connected layer 1
        x = self.fc1(x)
        x = F.relu(x)
        
        # dropout to regularize the model
        x = self.dropout2(x)
        
        # fully connected layer 2
        x = self.fc2(x)

        # return the output of the model
        return x