import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import MNISTClassifier

class MNISTPredictor:
    """
    Utility class for making predictions with a trained MNIST classifier.
    
    This class handles loading the trained model and preprocessing input images
    to make them compatible with the model's expected input format.
    """
    def __init__(self, model_path='model/mnist_classifier.pth'):
        """
        Initialise the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model weights
        """
        # load the model architecture
        self.model = MNISTClassifier()
        
        # load the trained weights
        self.model.load_state_dict(torch.load(model_path))
        
        # set the model to evaluation mode
        self.model.eval()
        
        # define the transformation pipeline for input images
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess an input image for the model.
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Preprocessed tensor ready for model input
        """
        # convert numpy array to PIL Image if necessary
        if isinstance(image, np.ndarray):
            # if the image is a drawing from Streamlit (RGBA), convert to grayscale
            if len(image.shape) == 3 and image.shape[2] == 4:
                # use alpha channel as the image (white drawing on black background)
                image = image[:, :, 3]
            
            # convert to PIL Image
            image = Image.fromarray(image)
        
        # ensure the image is in grayscale mode
        if image.mode != 'L':
            image = image.convert('L')
        
        # apply transformations
        tensor = self.transform(image)
        
        # add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict(self, image):
        """
        Make a prediction for an input image.
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Dictionary containing:
            - predicted_digit: The predicted digit (0-9)
            - confidence: Confidence score for the prediction (0-1)
            - probabilities: List of probabilities for all digits
        """
        # preprocess the image
        tensor = self.preprocess_image(image)
        
        # make prediction
        with torch.no_grad():
            output = self.model(tensor)
            
            # get probabilities
            probabilities = F.softmax(output, dim=1)[0]
            
            # get the predicted digit and its probability
            predicted_digit = output.argmax(dim=1, keepdim=True).item()
            confidence = probabilities[predicted_digit].item()
            
            # convert probabilities to list
            all_probabilities = probabilities.tolist()
        
        return {
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'probabilities': all_probabilities
        } 