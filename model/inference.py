import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from .model import MNISTClassifier

class MNISTPredictor:
    """
    Utility class for making predictions with a trained MNIST classifier.
    
    This class handles loading the trained model and preprocessing input images
    to make them compatible with the model's expected input format.
    """
    def __init__(self, model_path='saved_models/mnist_classifier.pth'):
        """
        Initialise the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model weights
        """
        print(f"Loading model from {model_path}...")
        
        # Load the model architecture
        self.model = MNISTClassifier()
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the trained weights
        try:
            # Try to load directly to the target device
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            print(f"Warning: Error loading model with device mapping: {e}")
            print("Falling back to CPU loading...")
            # Fall back to CPU loading
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded and set to {self.device}")
        
        # Define the transformation pipeline for raw images (not already processed MNIST images)
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess an input image for the model.
        
        Args:
            image: PIL Image, numpy array, or PyTorch tensor
        
        Returns:
            Preprocessed tensor ready for model input
        """
        # If already a tensor, handle appropriately
        if isinstance(image, torch.Tensor):
            # If it's a 2D tensor (single image without batch dimension)
            if image.dim() == 2:
                # Add channel dimension (H, W) -> (1, H, W)
                tensor = image.unsqueeze(0)
            # If it's a 3D tensor with channel dimension (C, H, W)
            elif image.dim() == 3 and image.size(0) == 1:
                # Keep as is - already has channel dimension
                tensor = image
            # If it's a 3D tensor with batch dimension (N, H, W)
            elif image.dim() == 3 and image.size(0) != 1:
                # Add channel dimension (N, H, W) -> (N, 1, H, W)
                tensor = image.unsqueeze(1)
            else:
                # Already has batch and channel dimensions (N, C, H, W)
                tensor = image
                
            # Move to the same device as the model
            return tensor.to(self.device)
            
        # Convert numpy array to PIL Image if necessary
        if isinstance(image, np.ndarray):
            # If the image is a drawing from Streamlit (RGBA), convert to grayscale
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
        
        # Add batch dimension and move to device
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def predict(self, image):
        """
        Make a prediction for an input image.
        
        Args:
            image: PIL Image, numpy array, or PyTorch tensor
        
        Returns:
            Dictionary containing:
            - predicted_digit: The predicted digit (0-9)
            - confidence: Confidence score for the prediction (0-1)
            - probabilities: List of probabilities for all digits
        """
        # Preprocess the image
        tensor = self.preprocess_image(image)
        
        # make prediction
        with torch.no_grad():
            # Forward pass through the model
            output = self.model(tensor)
            
            # The model outputs log probabilities
            # Need to use exp to get actual probabilities
            # Get the first item if batch size is 1
            if output.size(0) == 1:
                probabilities = torch.exp(output[0])
            else:
                # Handle batch predictions if needed
                probabilities = torch.exp(output)
            
            # get the predicted digit and its probability
            predicted_digit = probabilities.argmax().item()
            confidence = probabilities[predicted_digit].item()
            
            # Convert probabilities to list
            all_probabilities = probabilities.cpu().tolist()
        
        return {
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'probabilities': all_probabilities
        } 