import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import logging
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import from model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import model architecture
from model.model import MNISTClassifier

class MNISTPredictor:
    """
    Class for making predictions on MNIST digit images.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the trained model file
        """
        # set model path
        self.model_path = model_path or os.getenv('MODEL_PATH', 'saved_models/mnist_classifier.pth')
        
        # define transformations
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST dataset mean and standard deviation
        ])
        
        # initialize model
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # try to load the model
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")


    def _load_model(self):
        """
        Load the trained model from the specified path.
        """
        # check if model path exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        # initialize model
        self.model = MNISTClassifier()
        
        try:
            # load model parameters
            logger.info(f"Loading model from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def preprocess_image(self, image):
        """
        Preprocess an image for prediction.
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Preprocessed tensor
        """
        try:
            # convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                # check if image is grayscale (2D array)
                if len(image.shape) == 2:
                    # convert to 3D array with single channel
                    image = np.expand_dims(image, axis=2)
                
                # convert to PIL Image
                image = Image.fromarray(image.astype('uint8'))
            
            # apply transformations
            tensor = self.transform(image)
            # add batch dimension
            tensor = tensor.unsqueeze(0)
            return tensor
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            # return a zeros tensor as fallback
            return torch.zeros(1, 1, 28, 28)
    
    def predict(self, image):
        """
        Make a prediction on an image.
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            tuple: (predicted_digit, confidence, probabilities)
        """
        try:
            # preprocess image
            tensor = self.preprocess_image(image)
            
            # move tensor to device
            tensor = tensor.to(self.device)
            
            # make prediction
            with torch.no_grad():
                outputs = self.model(tensor)
                
                # convert log softmax to probabilities
                # Apply softmax to get proper probabilities that sum to 1
                probs = F.softmax(outputs, dim=1)
                
                # get predicted class
                _, predicted = torch.max(outputs.data, 1)
                
                # get predicted digit and confidence
                predicted_digit = predicted.item()
                probabilities = probs[0].cpu().numpy()
                confidence = probabilities[predicted_digit]
                
                return predicted_digit, float(confidence), list(probabilities)
        
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            # return fallback prediction
            return 0, 0.0, [0.1] * 10
    

# test function
if __name__ == "__main__":
    # get model path from command line argument or environment variable
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv('MODEL_PATH')
    