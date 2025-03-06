import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import psycopg2
import io
import base64
import time
import datetime
from dotenv import load_dotenv
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import json

# Add parent directory to path to import from model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.inference import MNISTPredictor

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# Database connection parameters
DB_PARAMS = {
    'host': 'localhost',
    'port': '5432',
    'database': 'mnist_db',
    'user': 'postgres',
    'password': 'master'
}

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

def log_prediction(predicted_digit, confidence, true_label=None, image=None):
    """Log prediction to PostgreSQL database"""
    try:
        # Convert image to binary data if provided
        image_data = None
        if image is not None:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            image_data = img_byte_arr.getvalue()
        
        # Connect to database
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        
        # Insert prediction into database
        query = """
            INSERT INTO predictions 
            (timestamp, predicted_digit, confidence, true_label, image_data)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        # Execute query with parameters
        cursor.execute(
            query, 
            (datetime.datetime.now(), predicted_digit, confidence, true_label,
             psycopg2.Binary(image_data) if image_data else None)
        )
        
        # Commit transaction
        conn.commit()
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return True
    except psycopg2.OperationalError as e:
        # Don't show warning for every prediction if we already know the database is unavailable
        if not hasattr(st.session_state, 'db_connection_failed'):
            st.warning(f"Could not log to database: {e}")
            st.session_state.db_connection_failed = True
        return False
    except Exception as e:
        st.warning(f"Could not log to database: {e}")
        return False

def get_statistics():
    """Get prediction statistics from the database"""
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        
        # Get total predictions count
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_count = cursor.fetchone()[0]
        
        # Get count of predictions with true labels
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE true_label IS NOT NULL")
        true_label_count = cursor.fetchone()[0]
        
        # Get accuracy statistics
        try:
            cursor.execute("""
                SELECT 
                    total_predictions,
                    correct_predictions,
                    incorrect_predictions,
                    accuracy_percentage,
                    avg_confidence
                FROM prediction_accuracy
            """)
            accuracy_stats = cursor.fetchone()
        except psycopg2.errors.UndefinedTable:
            # View doesn't exist yet
            accuracy_stats = None
        
        # If no accuracy stats yet (no true labels), create default values
        if accuracy_stats is None:
            accuracy_stats = (0, 0, 0, 0.0, 0.0)
        
        # Get digit-specific counts
        try:
            cursor.execute("""
                SELECT 
                    predicted_digit,
                    COUNT(*) as prediction_count,
                    CAST(AVG(confidence) AS NUMERIC(10,4)) as avg_confidence
                FROM predictions
                GROUP BY predicted_digit
                ORDER BY predicted_digit
            """)
            digit_stats = cursor.fetchall()
        except psycopg2.errors.UndefinedTable:
            # Table doesn't exist yet
            digit_stats = []
        
        # Get recent predictions
        try:
            cursor.execute("""
                SELECT timestamp, predicted_digit, confidence, true_label
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT 10
            """)
            recent_predictions = cursor.fetchall()
        except psycopg2.errors.UndefinedTable:
            # Table doesn't exist yet
            recent_predictions = []
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return {
            'total_count': total_count,
            'true_label_count': true_label_count,
            'accuracy_stats': accuracy_stats,
            'digit_stats': digit_stats,
            'recent_predictions': recent_predictions
        }
    except psycopg2.OperationalError as e:
        if not hasattr(st.session_state, 'db_connection_failed'):
            st.warning(f"Could not fetch statistics: {e}")
            st.session_state.db_connection_failed = True
        return None
    except Exception as e:
        st.warning(f"Could not fetch statistics: {e}")
        return None

def test_db_connection():
    """Test connection to PostgreSQL database"""
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_PARAMS)
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

def ensure_database_setup():
    """Ensure database is set up correctly"""
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        
        # Check if predictions table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'predictions'
            )
        """)
        table_exists = cursor.fetchone()[0]
        
        # Create tables if they don't exist
        if not table_exists:
            st.sidebar.info("Creating database tables...")
            
            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    predicted_digit INTEGER NOT NULL,
                    confidence NUMERIC(10, 4) NOT NULL,
                    true_label INTEGER,
                    image_data BYTEA
                )
            """)
            
            # Create views for statistics
            cursor.execute("""
                CREATE OR REPLACE VIEW prediction_accuracy AS
                SELECT
                    COUNT(*) AS total_predictions,
                    SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) AS correct_predictions,
                    SUM(CASE WHEN predicted_digit != true_label THEN 1 ELSE 0 END) AS incorrect_predictions,
                    CASE 
                        WHEN COUNT(*) > 0 THEN 
                            ROUND((SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)::NUMERIC) * 100, 2)
                        ELSE 0
                    END AS accuracy_percentage,
                    CAST(AVG(confidence) AS NUMERIC(10,4)) AS avg_confidence
                FROM predictions
                WHERE true_label IS NOT NULL
            """)
            
            # Create view for digit-specific statistics
            cursor.execute("""
                CREATE OR REPLACE VIEW digit_stats AS
                SELECT
                    predicted_digit,
                    COUNT(*) AS prediction_count,
                    SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) AS correct_predictions,
                    SUM(CASE WHEN predicted_digit != true_label THEN 1 ELSE 0 END) AS incorrect_predictions,
                    CASE 
                        WHEN COUNT(*) > 0 THEN 
                            ROUND((SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)::NUMERIC) * 100, 2)
                        ELSE 0
                    END AS accuracy_percentage,
                    CAST(AVG(confidence) AS NUMERIC(10,4)) AS avg_confidence
                FROM predictions
                WHERE true_label IS NOT NULL
                GROUP BY predicted_digit
                ORDER BY predicted_digit
            """)
            
            # Commit changes
            conn.commit()
            st.sidebar.success("Database tables created successfully!")
            table_exists = True
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return table_exists
    except Exception as e:
        if not hasattr(st.session_state, 'db_connection_failed'):
            st.warning(f"Database setup failed: {e}")
            st.session_state.db_connection_failed = True
        return False

def main():
    """Main function to run the Streamlit app"""
    # Set title
    st.title("MNIST Digit Classifier")
    
    # Test connection button
    if st.sidebar.button("Test Connection"):
        success, message = test_db_connection()
        if success:
            st.sidebar.success(f"Connection successful!")
            # Reset the connection failed flag if it was set
            if hasattr(st.session_state, 'db_connection_failed'):
                del st.session_state.db_connection_failed
        else:
            st.sidebar.error(f"Connection failed.")
            st.session_state.db_connection_failed = True
    
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
                st.metric("Total Predictions", stats['total_count'])
                
                # Display accuracy statistics if available
                if stats.get('accuracy_stats'):
                    st.markdown("### Accuracy Statistics")
                    
                    # Check if we have any predictions with true labels
                    if stats['accuracy_stats'][0] > 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Correct Predictions", stats['accuracy_stats'][1])
                        with col2:
                            st.metric("Incorrect Predictions", stats['accuracy_stats'][2])
                        with col3:
                            st.metric("Accuracy", f"{stats['accuracy_stats'][3]}%")
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
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No statistics available. Make some predictions first or check database connection.")

if __name__ == "__main__":
    main() 
