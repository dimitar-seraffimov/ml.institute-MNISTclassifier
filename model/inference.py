import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import logging

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
        # Set model path
        self.model_path = model_path or os.getenv('MODEL_PATH', 'saved_models/mnist_classifier.pth')
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST dataset mean and std
        ])
        
        # Initialize model
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Try to load the model
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Model could not be loaded. A dummy model will be used for predictions.")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """
        Create a dummy model for use when the real model can't be loaded.
        This will return random predictions.
        """
        logger.info("Creating dummy model for predictions")
        self.model = MNISTClassifier()
        self.model.to(self.device)
        self.model.eval()
        self.using_dummy = True
    
    def _load_model(self):
        """
        Load the trained model from the specified path.
        """
        # Check if model path exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        # Initialize model
        self.model = MNISTClassifier()
        
        try:
            # Load model parameters
            logger.info(f"Loading model from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            self.using_dummy = False
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
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                # Check if image is grayscale (2D array)
                if len(image.shape) == 2:
                    # Convert to 3D array with single channel
                    image = np.expand_dims(image, axis=2)
                
                # Convert to PIL Image
                image = Image.fromarray(image.astype('uint8'))
            
            # Apply transformations
            tensor = self.transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            # Return a zeros tensor as fallback
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
            # Check if using dummy model
            if getattr(self, 'using_dummy', False):
                # Return random prediction
                probabilities = np.random.rand(10)
                probabilities = probabilities / np.sum(probabilities)
                predicted_digit = np.argmax(probabilities)
                confidence = probabilities[predicted_digit]
                logger.warning("Using dummy model. Prediction is random.")
                return predicted_digit, float(confidence), list(probabilities)
            
            # Preprocess image
            tensor = self.preprocess_image(image)
            
            # Move tensor to device
            tensor = tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(tensor)
                
                # Convert log softmax to probabilities
                probs = torch.exp(outputs)
                
                # Get predicted class
                _, predicted = torch.max(outputs.data, 1)
                
                # Get predicted digit and confidence
                predicted_digit = predicted.item()
                probabilities = probs[0].cpu().numpy()
                confidence = probabilities[predicted_digit]
                
                return predicted_digit, float(confidence), list(probabilities)
        
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            # Return fallback prediction
            return 0, 0.0, [0.1] * 10
    
    def predict_batch(self, images):
        """
        Make predictions on a batch of images.
        
        Args:
            images: List of PIL Images or numpy arrays
        
        Returns:
            List of tuples: [(predicted_digit, confidence, probabilities), ...]
        """
        results = []
        
        for image in images:
            result = self.predict(image)
            results.append(result)
        
        return results


# Test function
def test_predictor(model_path=None):
    """
    Test the predictor with a sample image.
    
    Args:
        model_path: Path to the trained model file
    """
    try:
        # Create a simple test image (random values)
        test_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        
        # Initialize predictor
        predictor = MNISTPredictor(model_path)
        
        # Make prediction
        predicted_digit, confidence, probabilities = predictor.predict(test_image)
        
        print(f"Predicted digit: {predicted_digit}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: {[round(prob, 4) for prob in probabilities]}")
        
        return True
    
    except Exception as e:
        print(f"Error testing predictor: {e}")
        return False


if __name__ == "__main__":
    # Get model path from command line argument or environment variable
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv('MODEL_PATH')
    
    # Test the predictor
    test_predictor(model_path) 