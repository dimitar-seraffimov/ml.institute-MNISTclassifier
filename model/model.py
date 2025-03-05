import torch.nn as nn
import torch.nn.functional as F

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
        # first convolutional layer: 
        # 1 input channel (grayscale), 10 output channels
        # kernel size 5x5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        
        # second convolutional layer: 
        # 10 input channels, 20 output channels
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
       
        # using standard dropout instead of spatial dropout to avoid dimension warnings
        # standard dropout works on any input shape
        self.conv2_drop = nn.Dropout(0.25)

        # First fully connected layer
        self.fc1 = nn.Linear(320, 50)
        
        
        # Output layer: 
        # 50 features, 10 output features (one for each digit)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # First convolutional block: conv -> relu -> max pooling
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # Second convolutional block:
        # apply dropout to the output of the second convolutional layer
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # reshape the output of the second convolutional block
        x = x.view(-1, 320)
        
        # first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # dropout for regularisation
        x = F.dropout(x, training=self.training)
        
        # Output layer (logits)
        x = self.fc2(x)
        
        # return log softmax probabilities
        return F.log_softmax(x, dim=1) 