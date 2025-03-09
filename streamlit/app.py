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

# add parent directory to path to import from model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.inference import MNISTPredictor
# import from local directory, not from streamlit package
from db_utils import log_prediction, get_statistics, ensure_database_setup

# load environment variables from .env file
load_dotenv()

# set page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# model path for the saved model file
MODEL_PATH = os.getenv('MODEL_PATH', 'saved_models/mnist_classifier.pth')

@st.cache_resource
def load_predictor():
    """Load the MNIST predictor"""
    try:
        # initialize predictor with model path
        predictor = MNISTPredictor(MODEL_PATH)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    """main function to run the streamlit app"""

    st.title("MNIST Digit Classifier")
    
    # check if database is available, if not - set up the database
    db_available = ensure_database_setup()
    
    if not db_available:
        st.warning("Database is not available or not set up correctly.")
        st.info("The application will still work, but predictions won't be logged.")

    # create tabs for different sections
    tab1, tab2 = st.tabs(["Draw & Predict", "Statistics"])
    
    with tab1:
        # add a hint for using the application
        st.info("ℹ️ Draw a digit on the canvas, set the true label and click 'Predict' to see the model's prediction.")
        
        # loading the predictor
        predictor = load_predictor()
        
        if predictor is None:
            st.error("Failed to load model. Please check if the model file exists.")
            return
        
        # create two columns for drawing and results
        drawing_column, results_column = st.columns(2)
        
        with drawing_column:
            st.subheader("Draw a digit (0-9)")
            
            # initialize canvas state if needed
            if 'canvas_key' not in st.session_state:
                st.session_state.canvas_key = "canvas_1"
            
            # add a styled clear canvas button
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
                # clear any stored prediction
                if 'current_prediction' in st.session_state:
                    del st.session_state.current_prediction
                # clear true label if it exists
                if 'true_label' in st.session_state:
                    del st.session_state.true_label
                # change the canvas key to force a reset
                st.session_state.canvas_key = f"canvas_{int(time.time())}"
                st.rerun()
            
            # create canvas for drawing
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
        
            # checking for canvas data, needed for enabling the predict button
            if canvas_result.image_data is not None:
                # check if there are any non-black pixels (values > 0)
                # for rgb images, check if any channel has values > 0
                if len(canvas_result.image_data.shape) > 2:  # rgb image
                    # sum across all channels
                    pixel_sum = np.sum(canvas_result.image_data[:, :, :3])
                else:  # grayscale image
                    pixel_sum = np.sum(canvas_result.image_data)                
                # consider the canvas to have data if there are enough non-black pixels
                canvas_has_data = pixel_sum > 1000

            else:
                canvas_has_data = False
                        
            # initialize prediction_attempted flag if it doesn't exist
            if 'prediction_attempted' not in st.session_state:
                st.session_state.prediction_attempted = False
            
            # create true label input and predict button
            true_label_input, predict_button = st.columns(2)
            
            with true_label_input:
                # number input for true label
                st.subheader("True Label")
                true_label = st.number_input("Enter Digit (0-9):", 
                                          min_value=0, max_value=9, step=1,
                                          help="Enter the digit you drew before prediction",
                                          key="true_label_input")
                
                # store true label in session state
                if true_label is not None:
                    st.session_state.true_label = true_label

            with predict_button:
                # button is enabled only if canvas has data
                predict_enabled = canvas_has_data
                
                # custom styling for the predict button based on its state
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
                
                # add a visual indicator of button state
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
                
                # set prediction_attempted to True if button is clicked
                if predict_button_clicked:
                    st.session_state.prediction_attempted = True
            
        
        with results_column:
            st.subheader("Prediction Results")
            
            # handle predict button click
            if predict_button_clicked and canvas_has_data:
                # get the drawn image
                image_data = canvas_result.image_data
                
                # convert to grayscale and resize to 28x28
                image = Image.fromarray(image_data.astype(np.uint8)).convert('L')
                
                # make prediction using the predictor
                predicted_digit, confidence, probabilities = predictor.predict(image)
                
                # get true label from session state
                true_label = st.session_state.true_label
                
                # log prediction to database
                log_success = log_prediction(predicted_digit, confidence, true_label, image)
                if log_success:
                    st.success(f"Prediction logged with true label: {true_label}")
                    st.info("To learn more about the overall model performance, check out the statistics tab.")
                else:
                    st.error("Could not log to database. Please check your connection.")
                
                # store in session state
                st.session_state.current_prediction = {
                    'image': image,
                    'predicted_digit': predicted_digit,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'true_label': true_label
                }
            
            # display prediction if available
            if 'current_prediction' in st.session_state:
                # get prediction from session state
                prediction = st.session_state.current_prediction
                image = prediction['image']
                predicted_digit = prediction['predicted_digit']
                confidence = prediction['confidence']
                probabilities = prediction['probabilities']
                true_label = prediction.get('true_label')
                
                # display the processed image
                st.image(image.resize((140, 140)), caption="Processed Image")
                
                # display prediction results
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### Predicted: {predicted_digit}")
                    # Ensure confidence is displayed as a percentage
                    confidence_pct = confidence * 100 if confidence <= 1.0 else confidence
                    # Cap at 100% if it's unreasonably high
                    if confidence_pct > 100:
                        confidence_pct = min(confidence_pct, 100)
                    st.markdown(f"### Confidence: {confidence_pct:.2f}%")
                
                with col2:
                    st.markdown(f"### True Label: {true_label}")
                    # show if prediction was correct
                    if true_label is not None:
                        is_correct = predicted_digit == true_label
                        st.markdown(f"### Correct: {'✅' if is_correct else '❌'}")
                
                # create a bar chart for probabilities
                prob_df = pd.DataFrame({
                    'Digit': [str(i) for i in range(10)],
                    'Probability': [min(float(p) * 100 if float(p) <= 1.0 else float(p), 100) for p in probabilities]
                })
                # Set y-axis label to indicate percentages
                st.bar_chart(prob_df, x='Digit', y='Probability')
                st.caption("Probability distribution across all digits (in %)")
    
    with tab2:
        # get statistics from database
        stats = get_statistics()
        
        if stats:
            # create three columns for the top section
            side1, side2, side3 = st.columns(3)
            
            with side1:
                st.subheader("Prediction Statistics")
                # display total predictions
                st.metric("Total Predictions", stats['total_predictions'])
                
                # display accuracy statistics if available
                if stats.get('accuracy_stats'):
                    st.markdown("### Accuracy Statistics")
                    
                    # check if we have any predictions with true labels
                    if stats['accuracy_stats'][1] > 0:  # predictions_with_true_label
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            # convert to int or float to avoid decimal type errors
                            correct_pred = int(stats['accuracy_stats'][2]) if isinstance(stats['accuracy_stats'][2], (int, float)) else float(stats['accuracy_stats'][2])
                            st.metric("Correct Predictions", correct_pred)
                        with col2:
                            # convert to int or float to avoid decimal type errors
                            incorrect_pred = int(stats['accuracy_stats'][3]) if isinstance(stats['accuracy_stats'][3], (int, float)) else float(stats['accuracy_stats'][3])
                            st.metric("Incorrect Predictions", incorrect_pred)
                        with col3:
                            # convert to float and format as string with % symbol
                            if stats['accuracy_stats'] and len(stats['accuracy_stats']) > 4:
                                accuracy = float(stats['accuracy_stats'][4]) if stats['accuracy_stats'][4] is not None else 0.0
                            else:
                                accuracy = 0.0
                            st.metric("Overall Model Accuracy", f"{accuracy:.2f}%")
                else:
                    st.info("No accuracy statistics available yet.")
            
            with side2:
                # display digit statistics
                st.subheader("Digit-specific Statistics")
                
                if stats['digit_stats']:
                    # create a dataframe for digit statistics
                    digit_stats_data = {
                        "Digit": [],
                        "Count": [],
                        "Avg. Confidence": []
                    }
                    
                    for stat in stats['digit_stats']:
                        digit_stats_data["Digit"].append(str(stat[0]))  # convert to string
                        digit_stats_data["Count"].append(int(stat[1]))  # ensure it's an integer
                        # Ensure confidence is displayed as a percentage
                        conf_val = stat[2] * 100 if stat[2] <= 1.0 else stat[2]
                        # Cap at 100% if it's unreasonably high
                        conf_val = min(conf_val, 100)
                        digit_stats_data["Avg. Confidence"].append(f"{conf_val:.2f}%")
                    
                    # convert to pandas dataframe with explicit dtypes
                    digit_df = pd.DataFrame(digit_stats_data)
                    digit_df["Digit"] = digit_df["Digit"].astype(str)
                    digit_df["Count"] = digit_df["Count"].astype(int)
                    digit_df["Avg. Confidence"] = digit_df["Avg. Confidence"].astype(str)
                    # Display dataframe without index
                    st.dataframe(digit_df, hide_index=True)
                else:
                    st.info("No digit statistics available yet. Make some predictions first.")
            
            with side3:
                # display recent predictions
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
                        recent_data["Predicted"].append(str(pred[1]))  # convert to string
                        # Ensure confidence is displayed as a percentage
                        conf_val = pred[2] * 100 if pred[2] <= 1.0 else pred[2]
                        # Cap at 100% if it's unreasonably high
                        conf_val = min(conf_val, 100)
                        recent_data["Confidence"].append(f"{conf_val:.2f}%")
                        # convert true_label to string to avoid type conversion issues
                        recent_data["True Label"].append(str(pred[3]) if pred[3] is not None else "Not provided")
                    
                    # convert to pandas DataFrame with explicit dtypes
                    df = pd.DataFrame(recent_data)
                    # ensure True Label column is treated as string
                    df["True Label"] = df["True Label"].astype(str)
                    # Display dataframe without index
                    st.dataframe(df, hide_index=True)
                else:
                    st.info("No predictions have been made yet.")
            
            # add a separator
            st.markdown("---")
            
            # display digit distribution chart below the three columns
            if stats['digit_stats']:
                st.subheader("Digit Distribution")
                # create a larger chart by using a container with custom CSS
                chart_container = st.container()
                with chart_container:
                    # add some padding for better visual appearance
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
                    # create the bar chart with the full width
                    st.markdown('<div class="digit-distribution-chart">', unsafe_allow_html=True)
                    st.bar_chart({str(row[0]): row[1] for row in stats['digit_stats']})
        else:
            st.info("No statistics available. Make some predictions first or check database connection.")

if __name__ == "__main__":
    main() 
