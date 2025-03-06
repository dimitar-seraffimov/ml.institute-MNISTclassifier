import numpy as np
import streamlit as st
from PIL import Image
import io
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import from model and database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.inference import MNISTPredictor

# Only import database if needed
try:
    from database.db_utils import DatabaseManager
    DATABASE_AVAILABLE = True
except Exception as e:
    print(f"Database not available: {e}")
    DATABASE_AVAILABLE = False

def load_predictor():
    """
    Load the MNIST predictor model.
    
    Returns:
        MNISTPredictor instance
    """
    # Check if predictor is already in session state
    if 'predictor' not in st.session_state:
        # Initialize predictor with model path
        model_path = os.getenv('MODEL_PATH', 'saved_models/mnist_classifier.pth')
        st.session_state.predictor = MNISTPredictor(model_path)
    
    return st.session_state.predictor

def load_db_manager():
    """
    Load the database manager.
    
    Returns:
        DatabaseManager instance or None if database not available
    """
    if not DATABASE_AVAILABLE:
        return None
        
    # Check if db_manager is already in session state
    if 'db_manager' not in st.session_state:
        try:
            # Initialize database manager
            st.session_state.db_manager = DatabaseManager()
        except Exception as e:
            print(f"Error initializing database: {e}")
            return None
    
    return st.session_state.db_manager

def create_canvas():
    """
    Create a simple canvas for drawing digits using Streamlit's built-in components.
    
    Returns:
        Canvas container
    """
    # Initialize the drawing state if it doesn't exist
    if 'drawing' not in st.session_state:
        # Create a blank canvas (black background)
        st.session_state.drawing = np.zeros((28, 28), dtype=np.uint8)
    
    canvas_container = st.container()
    
    with canvas_container:
        # Display instructions
        st.markdown("### Draw a digit by clicking on the cells below")
        
        # Display the current canvas image, scaled up for better visibility
        drawing_display = np.repeat(np.repeat(st.session_state.drawing, 10, axis=0), 10, axis=1)
        st.image(drawing_display, use_container_width=False)
        
        # Create a 7x4 grid of controls for drawing
        for i in range(7):
            cols = st.columns(4)
            for j in range(4):
                # Create a unique key for each cell
                cell_key = f"cell_{i}_{j}"
                
                # Calculate position in the 28x28 grid (each control represents a 4x4 area)
                start_x = j * 4
                start_y = i * 4
                
                # Add a button for each cell area
                with cols[j]:
                    if st.button("■", key=cell_key):
                        # Fill a 4x4 area with white
                        for y in range(start_y, start_y + 4):
                            for x in range(start_x, start_x + 4):
                                if 0 <= y < 28 and 0 <= x < 28:
                                    st.session_state.drawing[y, x] = 255
                        st.rerun()
    
    return canvas_container

def get_canvas_image():
    """
    Get the image data from the canvas.
    
    Returns:
        PIL Image object
    """
    if 'drawing' in st.session_state:
        # Convert numpy array to PIL Image
        image = Image.fromarray(st.session_state.drawing)
        return image
    
    return None

def clear_canvas():
    """
    Clear the canvas.
    """
    # Reset the drawing to a blank canvas (black background)
    if 'drawing' in st.session_state:
        st.session_state.drawing = np.zeros((28, 28), dtype=np.uint8)
    
    # Clear the prediction result
    if 'prediction_result' in st.session_state:
        del st.session_state.prediction_result
    
    # Clear the true label if it exists
    if 'true_label' in st.session_state:
        del st.session_state.true_label
    
    # Set a flag to indicate the canvas was cleared
    st.session_state.canvas_cleared = True
    
    # Trigger a rerun to refresh the canvas and UI
    st.rerun()

def plot_prediction_probabilities(probabilities):
    """
    Plot the prediction probabilities.
    
    Args:
        probabilities: List of probabilities for each digit
    
    Returns:
        Matplotlib figure
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot the probabilities as a bar chart
    digits = list(range(10))
    ax.bar(digits, probabilities, color='skyblue')
    
    # Add labels and title
    ax.set_xlabel('Digit')
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities')
    
    # Set x-axis ticks
    ax.set_xticks(digits)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add probability values above each bar
    for i, prob in enumerate(probabilities):
        ax.text(i, prob + 0.01, f'{prob:.2f}', ha='center')
    
    return fig

def display_statistics(db_manager):
    """
    Display statistics from the database.
    
    Args:
        db_manager: DatabaseManager instance
    """
    if db_manager is None:
        st.info("Database connection not available. Statistics cannot be displayed.")
        return
        
    try:
        # Get accuracy statistics
        accuracy_stats = db_manager.get_accuracy_statistics()
        
        # Get digit statistics
        digit_stats = db_manager.get_digit_statistics()
        
        # Display accuracy statistics
        st.subheader('Overall Accuracy Statistics')
        
        # Create columns for statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric('Total Predictions', accuracy_stats['total_predictions'])
        
        with col2:
            st.metric('Correct Predictions', accuracy_stats['correct_predictions'])
        
        with col3:
            st.metric('Accuracy', f"{accuracy_stats['accuracy_percentage']}%")
        
        with col4:
            st.metric('Avg. Confidence', f"{accuracy_stats['avg_confidence']:.2f}")
        
        # Display digit statistics
        st.subheader('Digit-Specific Statistics')
        
        # Create a table for digit statistics
        if digit_stats:
            # Convert to DataFrame for display
            import pandas as pd
            df = pd.DataFrame(digit_stats)
            df = df.rename(columns={
                'predicted_digit': 'Digit',
                'prediction_count': 'Count',
                'correct_predictions': 'Correct',
                'accuracy_percentage': 'Accuracy (%)',
                'avg_confidence': 'Avg. Confidence'
            })
            
            st.dataframe(df)
        else:
            st.info("No digit-specific statistics available yet.")
        
        # Display recent predictions
        st.subheader('Recent Predictions')
        
        # Get recent predictions
        recent_predictions = db_manager.get_recent_predictions(limit=10)
        
        if recent_predictions:
            # Convert to DataFrame for display
            import pandas as pd
            df = pd.DataFrame(recent_predictions)
            df = df.rename(columns={
                'id': 'ID',
                'timestamp': 'Timestamp',
                'predicted_digit': 'Predicted Digit',
                'confidence': 'Confidence',
                'true_label': 'True Label'
            })
            
            st.dataframe(df)
        else:
            st.info("No predictions have been made yet.")
    except Exception as e:
        st.error(f"Error displaying statistics: {e}")
        st.info("Database connection may not be available. Statistics will be displayed when the database is connected.") 