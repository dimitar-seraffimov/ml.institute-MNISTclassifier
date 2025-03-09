import torch.nn as nn
import torch.nn.functional as F
import torch

class MNISTClassifier(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
    - 2 Convolutional layers with ReLU activation, batch normalization and max pooling
    - 2 Fully connected layers with high dropout for regularization
    - Reduced capacity to prevent overfitting
    """
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        # convolutional layer 1
        # input channel - 1 (grayscale image)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # starting with 28x28 input:
        # formula for calculating the input size of the fully connected layer:
        # (Width - kernel_size + 2 * padding / stride) + 1
        # for this case, the formula looks like: (28 - 3 + 2*1 / 1) + 1 = 28
        # the image size is now 28x28 (28/2/2), so after the two max pooling layers, the image size will be 7x7
        # that's why we have 128 * 7 * 7

        # fully connected layer 1
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        # dropout with increased rate
        self.dropout1 = nn.Dropout(0.5)
        
        # fully connected layer 2 (output layer)
        self.fc2 = nn.Linear(64, 10)


    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, 0.25, self.training)

        # conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, 0.25, self.training)

        # flatten the output of the convolutional layers to pass it to the fully connected layers
        x = torch.flatten(x, 1)
        
        # fully connected layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # fully connected layer 2 (output layer)
        x = self.fc2(x)

        # return the output of the model
        return x