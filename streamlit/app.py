import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import io
import time
from dotenv import load_dotenv
from streamlit_drawable_canvas import st_canvas
import pandas as pd

# Add parent directory to path to import from model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.inference import MNISTPredictor
# Import from local directory, not from streamlit package
from db_utils import ensure_database_setup, log_prediction, get_statistics

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# Model path
MODEL_PATH = os.getenv('MODEL_PATH', 'saved_models/mnist_classifier.pth')

@st.cache_resource
def load_predictor():
    """Load the MNIST predictor"""
    try:
        # Initialize predictor with model path
        predictor = MNISTPredictor(MODEL_PATH)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def test_db_connection():
    """Test connection to PostgreSQL database"""
    try:
        import psycopg2
        from db_utils import get_db_params
        
        # Connect to database
        conn = psycopg2.connect(**get_db_params())
        cursor = conn.cursor()
        
        # Get PostgreSQL version
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return True, version
    except Exception as e:
        return False, str(e)

def main():
    """Main function to run the Streamlit app"""
    # Set title
    st.title("MNIST Digit Classifier")
    
    # Test connection button
    if st.sidebar.button("Test Connection"):
        success, message = test_db_connection()
        if success:
            st.sidebar.success(f"Connection successful!")
        else:
            st.sidebar.error(f"Connection failed.")
    
    # Check database setup
    db_available = ensure_database_setup()
    
    if not db_available:
        st.sidebar.warning("Database is not available or not set up correctly.")
        st.sidebar.info("The application will still work, but predictions won't be logged.")
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Draw & Predict", "Statistics"])
    
    with tab1:
        # Add a hint for using the application
        st.info("ℹ️ Draw a digit on the canvas, set the true label and click 'Predict' to see the model's prediction.")
        
        # Load predictor
        predictor = load_predictor()
        
        if predictor is None:
            st.error("Failed to load model. Please check if the model file exists.")
            return
        
        # Create two columns for drawing and results
        drawing_column, results_column = st.columns(2)
        
        with drawing_column:
            st.subheader("Draw a digit (0-9)")
            
            # Initialize canvas state if needed
            if 'canvas_key' not in st.session_state:
                st.session_state.canvas_key = "canvas_1"
            
            # Add a styled clear canvas button
            st.markdown(
                """
                <style>
                div.stButton > button {
                    width: 100%;
                    font-weight: bold;
                    margin-bottom: 10px;
                }
                </style>
                """, 
                unsafe_allow_html=True
            )
            
            if st.button("🗑️ Clear Canvas", key="clear_canvas", help="Clear the canvas"):
                # Clear any stored prediction
                if 'current_prediction' in st.session_state:
                    del st.session_state.current_prediction
                # Clear true label if it exists
                if 'true_label' in st.session_state:
                    del st.session_state.true_label
                # Change the canvas key to force a reset
                st.session_state.canvas_key = f"canvas_{int(time.time())}"
                st.rerun()
            
            # Create canvas for drawing
            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=20,
                stroke_color="white",
                background_color="black",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key=st.session_state.canvas_key,
                display_toolbar=False,
                update_streamlit=True
            )
            
            # More robust check for canvas data
            if canvas_result.image_data is not None:
                # Check if there are any non-black pixels (values > 0)
                # For RGB images, check if any channel has values > 0
                if len(canvas_result.image_data.shape) > 2:  # RGB image
                    # Sum across all channels
                    pixel_sum = np.sum(canvas_result.image_data[:, :, :3])
                else:  # Grayscale image
                    pixel_sum = np.sum(canvas_result.image_data)
                
                # Consider the canvas to have data if there are enough non-black pixels
                # This threshold can be adjusted as needed
                canvas_has_data = pixel_sum > 1000

            else:
                canvas_has_data = False
                st.sidebar.write("Canvas data is None")
            
            true_label_provided = 'true_label' in st.session_state
            
            # Initialize prediction_attempted flag if it doesn't exist
            if 'prediction_attempted' not in st.session_state:
                st.session_state.prediction_attempted = False
            
            # Add true label input and predict button
            true_label_input, predict_button = st.columns(2)
            
            with true_label_input:
                # Simple number input for true label
                st.subheader("True Label")
                true_label = st.number_input("Enter Digit (0-9):", 
                                          min_value=0, max_value=9, step=1,
                                          help="Enter the digit you drew before prediction",
                                          key="true_label_input")
                
                # Store true label in session state
                if true_label is not None:
                    st.session_state.true_label = true_label

            with predict_button:
                # Button is enabled only if canvas has data
                predict_enabled = canvas_has_data
                
                # Custom styling for the predict button based on its state
                button_style = """
                <style>
                .stButton button {
                    width: 100%;
                }
                .stButton button[disabled] {
                    opacity: 0.6;
                    cursor: not-allowed;
                    border: 1px solid #ccc;
                }
                </style>
                """
                st.markdown(button_style, unsafe_allow_html=True)
                
                st.subheader("Predict")
                
                # Add a visual indicator of button state
                if predict_enabled:
                    button_text = "✅ Run Prediction"
                else:
                    button_text = "❌ Run Prediction"
                
                predict_button_clicked = st.button(
                    button_text, 
                    disabled=not predict_enabled,
                    type="primary" if predict_enabled else "secondary",
                    key="predict_button"
                )
                
                # Set prediction_attempted to True if button is clicked
                if predict_button_clicked:
                    st.session_state.prediction_attempted = True
            
        
        with results_column:
            st.subheader("Prediction Results")
            
            # Handle predict button click
            if predict_button_clicked and canvas_has_data:
                # Get the drawn image
                image_data = canvas_result.image_data
                
                # Convert to grayscale and resize to 28x28
                image = Image.fromarray(image_data.astype(np.uint8)).convert('L')
                
                # Make prediction using the predictor
                predicted_digit, confidence, probabilities = predictor.predict(image)
                
                # Get true label from session state
                true_label = st.session_state.true_label
                
                # Log prediction to database
                log_success = log_prediction(predicted_digit, confidence, true_label, image)
                if log_success:
                    st.success(f"Prediction logged with true label: {true_label}")
                else:
                    st.error("Could not log to database. Please check your connection.")
                
                # Store in session state
                st.session_state.current_prediction = {
                    'image': image,
                    'predicted_digit': predicted_digit,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'true_label': true_label
                }
            
            # Display prediction if available
            if 'current_prediction' in st.session_state:
                # Get prediction from session state
                prediction = st.session_state.current_prediction
                image = prediction['image']
                predicted_digit = prediction['predicted_digit']
                confidence = prediction['confidence']
                probabilities = prediction['probabilities']
                true_label = prediction.get('true_label')
                
                # Display the processed image
                st.image(image.resize((140, 140)), caption="Processed Image")
                
                # Display prediction results
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### Predicted: {predicted_digit}")
                    st.markdown(f"### Confidence: {confidence:.2f}")
                
                with col2:
                    st.markdown(f"### True Label: {true_label}")
                    # Show if prediction was correct
                    if true_label is not None:
                        is_correct = predicted_digit == true_label
                        st.markdown(f"### Correct: {'✅' if is_correct else '❌'}")
                
                # Create a bar chart for probabilities
                prob_df = pd.DataFrame({
                    'Digit': [str(i) for i in range(10)],
                    'Probability': [float(p) for p in probabilities]
                })
                st.bar_chart(prob_df, x='Digit', y='Probability')
    
    with tab2:
        # Get statistics from database
        stats = get_statistics()
        
        if stats:
            # Create three columns for the top section
            side1, side2, side3 = st.columns(3)
            
            with side1:
                st.subheader("Prediction Statistics")
                # Display total predictions
                st.metric("Total Predictions", stats['total_predictions'])
                
                # Display accuracy statistics if available
                if stats.get('accuracy_stats'):
                    st.markdown("### Accuracy Statistics")
                    
                    # Check if we have any predictions with true labels
                    if stats['accuracy_stats'][1] > 0:  # predictions_with_true_label
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            # Convert to int or float to avoid Decimal type errors
                            correct_pred = int(stats['accuracy_stats'][2]) if isinstance(stats['accuracy_stats'][2], (int, float)) else float(stats['accuracy_stats'][2])
                            st.metric("Correct Predictions", correct_pred)
                        with col2:
                            # Convert to int or float to avoid Decimal type errors
                            incorrect_pred = int(stats['accuracy_stats'][3]) if isinstance(stats['accuracy_stats'][3], (int, float)) else float(stats['accuracy_stats'][3])
                            st.metric("Incorrect Predictions", incorrect_pred)
                        with col3:
                            # Convert to float and format as string with % symbol
                            accuracy = float(stats['accuracy_stats'][4]) if stats['accuracy_stats'][4] is not None else 0.0
                            st.metric("Accuracy", f"{accuracy:.2f}%")
                else:
                    st.info("No accuracy statistics available yet. Please provide feedback on your predictions.")
            
            with side2:
                # Display digit statistics
                st.subheader("Digit-specific Statistics")
                
                if stats['digit_stats']:
                    # Create a dataframe for digit statistics
                    digit_stats_data = {
                        "Digit": [],
                        "Count": [],
                        "Avg. Confidence": []
                    }
                    
                    for stat in stats['digit_stats']:
                        digit_stats_data["Digit"].append(str(stat[0]))  # Convert to string
                        digit_stats_data["Count"].append(int(stat[1]))  # Ensure it's an integer
                        digit_stats_data["Avg. Confidence"].append(f"{stat[2]:.2f}")
                    
                    # Convert to pandas DataFrame with explicit dtypes
                    digit_df = pd.DataFrame(digit_stats_data)
                    digit_df["Digit"] = digit_df["Digit"].astype(str)
                    digit_df["Count"] = digit_df["Count"].astype(int)
                    digit_df["Avg. Confidence"] = digit_df["Avg. Confidence"].astype(str)
                    st.dataframe(digit_df)
                else:
                    st.info("No digit statistics available yet. Make some predictions first.")
            
            with side3:
                # Display recent predictions
                st.subheader("Recent Predictions")
                
                if stats['recent_predictions']:
                    recent_data = {
                        "Time": [],
                        "Predicted": [],
                        "Confidence": [],
                        "True Label": []
                    }
                    
                    for pred in stats['recent_predictions']:
                        recent_data["Time"].append(pred[0])
                        recent_data["Predicted"].append(str(pred[1]))  # Convert to string
                        recent_data["Confidence"].append(f"{pred[2]:.2f}")
                        # Convert true_label to string to avoid type conversion issues
                        recent_data["True Label"].append(str(pred[3]) if pred[3] is not None else "Not provided")
                    
                    # Convert to pandas DataFrame with explicit dtypes
                    df = pd.DataFrame(recent_data)
                    # Ensure True Label column is treated as string
                    df["True Label"] = df["True Label"].astype(str)
                    st.dataframe(df)
                else:
                    st.info("No predictions have been made yet.")
            
            # Add a separator
            st.markdown("---")
            
            # Display digit distribution chart below the three columns
            if stats['digit_stats']:
                st.subheader("Digit Distribution")
                # Create a larger chart by using a container with custom CSS
                chart_container = st.container()
                with chart_container:
                    # Add some padding for better visual appearance
                    st.markdown(
                        """
                        <style>
                        .digit-distribution-chart {
                            padding: 20px 0;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    # Create the bar chart with the full width
                    st.markdown('<div class="digit-distribution-chart">', unsafe_allow_html=True)
                    st.bar_chart({str(digit): count for digit, count, _ in stats['digit_stats']})
        else:
            st.info("No statistics available. Make some predictions first or check database connection.")

if __name__ == "__main__":
    main() 
