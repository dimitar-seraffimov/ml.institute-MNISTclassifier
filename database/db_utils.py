import os
import io
import psycopg2
import psycopg2.extras
import numpy as np
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseManager:
    """
    Utility class for managing database operations for the MNIST classifier.
    
    This class handles connections to the PostgreSQL database and provides
    methods for logging predictions and retrieving statistics.
    """
    def __init__(self):
        """
        Initialize the database manager with connection parameters.
        """
        # Database connection parameters from environment variables with defaults
        self.db_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'mnist_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        }
        
        # Test connection during initialization
        self.test_connection()
    
    def test_connection(self):
        """
        Test the database connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            conn = self.get_connection()
            conn.close()
            print("Database connection successful")
            return True
        except Exception as e:
            print(f"Error connecting to PostgreSQL database: {e}")
            return False
    
    def get_connection(self):
        """
        Get a connection to the database.
        
        Returns:
            psycopg2.connection: Database connection
        
        Raises:
            Exception: If connection fails
        """
        try:
            # Connect to the database
            conn = psycopg2.connect(**self.db_params)
            return conn
        except Exception as e:
            print(f"Error connecting to PostgreSQL database: {e}")
            raise
    
    def log_prediction(self, predicted_digit, confidence, image=None, probabilities=None):
        """
        Log a prediction to the database.
        
        Args:
            predicted_digit: The predicted digit (0-9)
            confidence: The confidence of the prediction (0-1)
            image: The image data as a PIL Image or numpy array
            probabilities: List of probabilities for each digit
        
        Returns:
            ID of the inserted record or None if insertion fails
        """
        # Convert image to binary data if provided
        image_data = None
        if image is not None:
            if isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                image = Image.fromarray(image)
            
            # Convert PIL Image to binary data
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            image_data = img_byte_arr.getvalue()
        
        # Ensure probabilities is a list
        if probabilities is None:
            probabilities = [0.0] * 10
        
        conn = None
        cursor = None
        try:
            # Get database connection
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Insert prediction into database
            query = """
                INSERT INTO predictions 
                (timestamp, predicted_digit, confidence, image_data)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """
            
            # Execute query with parameters
            cursor.execute(
                query, 
                (datetime.now(), predicted_digit, confidence, 
                 psycopg2.Binary(image_data) if image_data else None)
            )
            
            # Get the ID of the inserted record
            prediction_id = cursor.fetchone()[0]
            
            # Commit the transaction
            conn.commit()
            
            return prediction_id
        except Exception as e:
            # Print error message
            print(f"Error logging prediction to database: {e}")
            
            # Rollback transaction if needed
            if conn:
                conn.rollback()
            
            return None
        finally:
            # Close cursor and connection
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def get_accuracy_statistics(self):
        """
        Get basic statistics from the database.
        
        Returns:
            Dictionary with total predictions and average confidence
        """
        # Default statistics in case of failure
        default_stats = {
            'total_predictions': 0,
            'avg_confidence': 0
        }
        
        conn = None
        cursor = None
        try:
            # Get database connection
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Query for basic statistics
            query = """
                SELECT 
                    COUNT(*) as total_predictions,
                    COALESCE(ROUND(AVG(confidence), 4), 0) as avg_confidence
                FROM predictions
            """
            
            # Execute query
            cursor.execute(query)
            
            # Fetch results
            result = cursor.fetchone()
            
            # Create statistics dictionary
            stats = {
                'total_predictions': result[0],
                'avg_confidence': result[1]
            }
            
            return stats
        except Exception as e:
            # Print error message
            print(f"Error getting basic statistics: {e}")
            
            # Return default statistics
            return default_stats
        finally:
            # Close cursor and connection
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def get_digit_statistics(self):
        """
        Get digit-specific statistics from the database.
        
        Returns:
            List of tuples with (digit, count, avg_confidence)
        """
        conn = None
        cursor = None
        try:
            # Get database connection
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Query digit statistics
            query = """
                SELECT 
                    predicted_digit,
                    COUNT(*) as prediction_count,
                    ROUND(AVG(confidence), 4) as avg_confidence
                FROM predictions
                GROUP BY predicted_digit
                ORDER BY predicted_digit
            """
            
            # Execute query
            cursor.execute(query)
            
            # Fetch all results
            results = cursor.fetchall()
            
            return results
        except Exception as e:
            print(f"Error getting digit statistics: {e}")
            return []
        finally:
            # Close cursor and connection
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def get_recent_predictions(self, limit=10):
        """
        Get recent predictions from the database.
        
        Args:
            limit: Maximum number of predictions to return
        
        Returns:
            List of dictionaries with recent predictions or empty list if database query fails
        """
        conn = None
        cursor = None
        try:
            # Get database connection
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Query for recent predictions
            query = """
                SELECT 
                    id,
                    timestamp,
                    predicted_digit,
                    confidence
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT %s
            """
            
            # Execute query with limit parameter
            cursor.execute(query, (limit,))
            
            # Fetch results
            results = cursor.fetchall()
            
            # Create list of recent predictions
            recent_predictions = []
            for row in results:
                prediction = {
                    'id': row[0],
                    'timestamp': row[1],
                    'predicted_digit': row[2],
                    'confidence': row[3]
                }
                recent_predictions.append(prediction)
            
            return recent_predictions
        except Exception as e:
            # Print error message
            print(f"Error getting recent predictions: {e}")
            
            # Return empty list
            return []
        finally:
            # Close cursor and connection
            if cursor:
                cursor.close()
            if conn:
                conn.close() 