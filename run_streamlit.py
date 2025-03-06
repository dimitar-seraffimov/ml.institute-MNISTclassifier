#!/usr/bin/env python
"""
Script to run the Streamlit app for the MNIST Digit Classifier.
"""
import os
import subprocess
import sys
import glob
import shutil

# Try to import dotenv, install if not available
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    print("python-dotenv package not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv
    load_dotenv()

def check_dependencies():
    """Check and install required dependencies"""
    try:
        import streamlit
        import torch
        import psycopg2
        import numpy
        import PIL
        print("All dependencies are installed.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing missing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully.")

def ensure_saved_models_dir():
    """Ensure the saved_models directory exists"""
    os.makedirs('saved_models', exist_ok=True)
    print("Saved models directory exists.")

def find_model_file():
    """Find a suitable model file"""
    # Check if model path is set in environment variable
    model_path = os.getenv('MODEL_PATH')
    if model_path and os.path.exists(model_path):
        print(f"Using model from environment variable: {model_path}")
        return model_path
    
    # Check if model exists in saved_models directory
    saved_models = glob.glob('saved_models/*.pth')
    if saved_models:
        model_path = saved_models[0]
        print(f"Using model from saved_models directory: {model_path}")
        return model_path
    
    # Check if model exists in model directory
    model_dir_models = glob.glob('model/*.pth')
    if model_dir_models:
        model_path = model_dir_models[0]
        print(f"Using model from model directory: {model_path}")
        return model_path
    
    print("No model file found. Please train a model first using train_model.py.")
    return None

def main():
    """Main function to run the Streamlit app"""
    # Print current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    # Check dependencies
    check_dependencies()
    
    # Ensure saved_models directory exists
    ensure_saved_models_dir()
    
    # Find model file
    model_path = find_model_file()
    
    if model_path:
        # Set model path environment variable
        os.environ['MODEL_PATH'] = model_path
        print(f"Model path set to: {model_path}")
    else:
        print("WARNING: No model file found. The application may not work correctly.")
    
    # Print information about the application
    print("\n" + "="*50)
    print("MNIST Digit Classifier")
    print("="*50)
    print("This application allows you to draw digits and get predictions from a trained model.")
    print("\nInstructions:")
    print("1. Draw a digit on the canvas by clicking on the grid.")
    print("2. Set the true label of the image.")
    print("3. Click 'Predict' to get the model's prediction.")
    print("4. View statistics on the 'Statistics' tab.")

    print("="*50 + "\n")
    
    # Get the port from environment variable (for Cloud Run)
    port = int(os.environ.get("PORT", 8080))
    
    print(f"Starting Streamlit on port {port}")
    
    # Check if streamlit/app.py exists
    if not os.path.exists("streamlit/app.py"):
        print(f"ERROR: streamlit/app.py not found. Files in streamlit directory: {os.listdir('streamlit')}")
        sys.exit(1)
    
    # Run the Streamlit app with the correct host and port
    # Using --server.address=0.0.0.0 ensures it binds to all network interfaces
    # Using --server.port ensures it uses the port provided by Cloud Run
    cmd = [
        "streamlit", "run", "streamlit/app.py",
        "--server.address=0.0.0.0",
        f"--server.port={port}"
    ]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 