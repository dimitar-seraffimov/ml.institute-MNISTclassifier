import os
import sys
import numpy as np
import torch
import torch.nn as nn
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
import torchvision.transforms as transforms

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

# Uncomment to use environment variables
# DB_PARAMS = {
#     'host': os.getenv('POSTGRES_HOST', 'localhost'),
#     'port': os.getenv('POSTGRES_PORT', '5432'),
#     'database': os.getenv('POSTGRES_DB', 'mnist_db'),
#     'user': os.getenv('POSTGRES_USER', 'postgres'),
#     'password': os.getenv('POSTGRES_PASSWORD', 'master')
# }

# Model path
MODEL_PATH = os.getenv('MODEL_PATH', 'mnist_classifier.pth')

# Define the model architecture
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

@st.cache_resource
def load_model():
    """Load the trained MNIST model"""
    try:
        # Initialize model
        model = MNISTClassifier()
        
        # Load model weights
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        # Set model to evaluation mode
        model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image):
    """Preprocess image for model input"""
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Apply transformations
    tensor = transform(image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor

def predict(model, device, image):
    """Make prediction on an image"""
    # Preprocess image
    tensor = preprocess_image(image)
    
    # Move tensor to device
    tensor = tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(tensor)
        
        # Convert log softmax to probabilities
        probs = torch.exp(outputs)
        
        # Get predicted class
        _, predicted = torch.max(outputs.data, 1)
        
        # Get predicted digit and confidence
        predicted_digit = predicted.item()
        probabilities = probs[0].cpu().numpy()
        confidence = probabilities[predicted_digit]
        
        return predicted_digit, float(confidence), probabilities

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
            st.sidebar.success(f"Connection successful!\n{message}")
            # Reset the connection failed flag if it was set
            if hasattr(st.session_state, 'db_connection_failed'):
                del st.session_state.db_connection_failed
        else:
            st.sidebar.error(f"Connection failed: {message}")
            st.session_state.db_connection_failed = True
    
    # Check database setup
    db_available = ensure_database_setup()
    
    if not db_available:
        st.sidebar.warning("Database is not available or not set up correctly.")
        st.sidebar.info("The application will still work, but predictions won't be logged.")
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Draw & Predict", "Statistics"])
    
    with tab1:
        st.markdown("Draw a digit on the canvas, set the true label and click 'Predict' to see the model's prediction.")
        
        # Load model
        model, device = load_model()
        
        if model is None:
            st.error("Failed to load model. Please check if the model file exists.")
            return
        
        # Create two columns for drawing and results
        drawing_column, results_column = st.columns(2)
        
        with drawing_column:
            st.subheader("Draw a digit (0-9)")
            
            # Create canvas for drawing
            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=20,
                stroke_color="white",
                background_color="black",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            # Add buttons for actions
            clear_canvas_button, true_label_input, predict_button = st.columns(3)
            
            with clear_canvas_button:
                if st.button("Clear Canvas"):
                    # This will trigger a rerun and reset the canvas
                    st.session_state.canvas_cleared = True
                    # Clear any stored prediction
                    if 'current_prediction' in st.session_state:
                        del st.session_state.current_prediction
                    # Clear true label if it exists
                    if 'true_label' in st.session_state:
                        del st.session_state.true_label
                    st.rerun()
            
            with true_label_input:
                # Simple number input for true label (no form needed)
                st.subheader("True Label")
                true_label = st.number_input("Enter Digit (0-9):", 
                                          min_value=0, max_value=9, step=1,
                                          help="Enter the digit you drew before prediction",
                                          key="true_label_input")
                
                # Store true label in session state
                if true_label is not None:
                    st.session_state.true_label = true_label

            with predict_button:
                # Check if canvas has data and true label is provided
                canvas_has_data = canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0
                true_label_provided = 'true_label' in st.session_state
                
                # Button is enabled only if both conditions are met
                predict_enabled = canvas_has_data and true_label_provided
                
                predict_button_clicked = st.button(
                    "Predict", 
                    disabled=not predict_enabled,
                    type="primary" if predict_enabled else "secondary"
                )
                
                if not canvas_has_data and predict_button_clicked:
                    st.warning("Please draw a digit first.")
                elif not true_label_provided and predict_button_clicked:
                    st.warning("Please provide the true label first.")
        
        with results_column:
            st.subheader("Prediction Results")
            
            # Handle predict button click
            if predict_button_clicked and canvas_has_data and true_label_provided:
                # Get the drawn image
                image_data = canvas_result.image_data
                
                # Convert to grayscale and resize to 28x28
                image = Image.fromarray(image_data.astype(np.uint8)).convert('L')
                
                # Make prediction
                predicted_digit, confidence, probabilities = predict(model, device, image)
                
                # Get true label from session state
                true_label = st.session_state.true_label
                

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
        side1, side2 = st.columns(2)
        with side1:
            st.subheader("Prediction Statistics")
            # Get statistics from database
            stats = get_statistics()
            
            if stats:
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
                
                # Display digit statistics
                st.markdown("### Digit-specific Statistics")
                
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
                    
                    # Create a bar chart for digit distribution
                    st.markdown("### Digit Distribution")
                    st.bar_chart({str(digit): count for digit, count, _ in stats['digit_stats']})
                else:
                    st.info("No digit statistics available yet. Make some predictions first.")
            

        with side2:
            # Display recent predictions
            st.markdown("### Recent Predictions")
            
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
                st.info("No statistics available. Make some predictions first or check database connection.")
if __name__ == "__main__":
    main() 