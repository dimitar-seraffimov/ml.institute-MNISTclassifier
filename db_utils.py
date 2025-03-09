"""
Database utilities for the MNIST Classifier application.
Handles connections to PostgreSQL, including Cloud SQL.
"""
import os
import psycopg2
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_db_params():
    """Get database connection parameters with Cloud SQL support"""
    # Get database connection parameters from environment variables
    db_params = {
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }
    
    # Check if running on Google Cloud Run
    running_in_cloud_run = os.environ.get('K_SERVICE') is not None
    
    # If running on Cloud Run, and a Cloud SQL instance connection name is provided,
    # use the Cloud SQL Proxy unix socket to connect
    db_instance_connection_name = os.getenv('DB_INSTANCE_CONNECTION_NAME')
    if running_in_cloud_run and db_instance_connection_name:
        db_params['host'] = f"/cloudsql/{db_instance_connection_name}"
    
    return db_params

def connect_to_database():
    """Connect to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(**get_db_params())
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def ensure_database_setup():
    """Ensure database is set up correctly"""
    try:
        conn = connect_to_database()
        if not conn:
            print("Could not connect to database.")
            return False
            
        cursor = conn.cursor()
        # check if predictions table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'predictions'
            )
        """)
        table_exists = cursor.fetchone()[0]

        # create tables if they don't exist
        if not table_exists:
            print("Creating database tables...")
            
            # create predictions table
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
            
            # create views for statistics
            cursor.execute("""
                CREATE OR REPLACE VIEW accuracy_stats AS
                SELECT 
                    COUNT(*) AS total_predictions,
                    COUNT(true_label) AS predictions_with_true_label,
                    SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) AS correct_predictions,
                    COUNT(true_label) - SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) AS incorrect_predictions,
                    CASE 
                        WHEN COUNT(true_label) > 0 THEN 
                            CAST(SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) AS NUMERIC) / COUNT(true_label) * 100
                        ELSE 0 
                    END AS accuracy,
                    CAST(AVG(confidence) AS NUMERIC(10,4)) AS avg_confidence
                FROM predictions;
            """)
            
            cursor.execute("""
                CREATE OR REPLACE VIEW digit_stats AS
                SELECT 
                    predicted_digit AS digit,
                    COUNT(*) AS count,
                    CAST(AVG(confidence) AS NUMERIC(10,4)) AS avg_confidence,
                    SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) AS correct_count,
                    SUM(CASE WHEN true_label IS NOT NULL THEN 1 ELSE 0 END) - SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) AS incorrect_count
                FROM predictions
                GROUP BY predicted_digit
                ORDER BY predicted_digit;
            """)
            
            # commit changes
            conn.commit()
            print("Database tables created successfully!")
        else:
            # update views to ensure they have the correct logic
            print("Updating database views...")
            
            cursor.execute("""
                CREATE OR REPLACE VIEW accuracy_stats AS
                SELECT 
                    COUNT(*) AS total_predictions,
                    COUNT(true_label) AS predictions_with_true_label,
                    SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) AS correct_predictions,
                    COUNT(true_label) - SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) AS incorrect_predictions,
                    CASE 
                        WHEN COUNT(true_label) > 0 THEN 
                            CAST(SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) AS NUMERIC) / COUNT(true_label) * 100
                        ELSE 0 
                    END AS accuracy,
                    CAST(AVG(confidence) AS NUMERIC(10,4)) AS avg_confidence
                FROM predictions;
            """)
            
            cursor.execute("""
                CREATE OR REPLACE VIEW digit_stats AS
                SELECT 
                    predicted_digit AS digit,
                    COUNT(*) AS count,
                    CAST(AVG(confidence) AS NUMERIC(10,4)) AS avg_confidence,
                    SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) AS correct_count,
                    SUM(CASE WHEN true_label IS NOT NULL THEN 1 ELSE 0 END) - SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) AS incorrect_count
                FROM predictions
                GROUP BY predicted_digit
                ORDER BY predicted_digit;
            """)
            
            conn.commit()
            print("Database views updated successfully!")
        
        # close cursor and connection
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Database setup error: {e}")
        return False

def log_prediction(predicted_digit, confidence, true_label=None, image=None):
    """Log prediction to PostgreSQL database"""
    try:
        # convert image to binary data if provided
        image_data = None
        if image is not None:
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            image_data = img_byte_arr.getvalue()
        
        # convert numpy types to Python native types
        if hasattr(predicted_digit, 'item'):
            predicted_digit = predicted_digit.item()
        if hasattr(confidence, 'item'):
            confidence = confidence.item()
        # ensure confidence is a probability between 0 and 1
        if confidence > 1.0:
            confidence = confidence / 100.0 if confidence <= 100.0 else 1.0
        if true_label is not None and hasattr(true_label, 'item'):
            true_label = true_label.item()
        
        conn = connect_to_database()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # insert prediction into database
        cursor.execute("""
            INSERT INTO predictions (timestamp, predicted_digit, confidence, true_label, image_data)
            VALUES (NOW(), %s, %s, %s, %s)
        """, (predicted_digit, confidence, true_label, image_data))
        
        # commit changes
        conn.commit()
        
        # close cursor and connection
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        # log error but don't crash the application
        print(f"Error logging prediction to database: {e}")
        return False

def get_statistics():
    """Get statistics from the database"""
    try:
        conn = connect_to_database()
        if not conn:
            return None
            
        cursor = conn.cursor()
        stats = {}
        
        # get total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        stats['total_predictions'] = cursor.fetchone()[0]
        
        # get count of predictions with true labels
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE true_label IS NOT NULL")
        stats['true_label_count'] = cursor.fetchone()[0]
        
        # get accuracy statistics
        try:
            cursor.execute("SELECT * FROM accuracy_stats")
            raw_stats = cursor.fetchone()
            
            # convert Decimal values to Python native types
            if raw_stats:
                stats['accuracy_stats'] = tuple(
                    float(val) if isinstance(val, (float, int)) or hasattr(val, 'as_integer_ratio') else val 
                    for val in raw_stats
                )
            else:
                stats['accuracy_stats'] = None
        except Exception as e:
            # if the view doesn't exist yet, set to None
            print(f"Error getting accuracy stats: {e}")
            stats['accuracy_stats'] = None
        
        # if no true labels, set default values
        if not stats['true_label_count'] or not stats['accuracy_stats']:
            # default values for: total_predictions, predictions_with_true_label, correct_predictions, incorrect_predictions, accuracy, avg_confidence
            stats['accuracy_stats'] = (stats['total_predictions'], 0, 0, 0, 0.0, 0.0)
        elif len(stats['accuracy_stats']) < 6:
            # if accuracy_stats exists but doesn't have all 6 elements, pad it with zeros
            existing_stats = list(stats['accuracy_stats'])
            while len(existing_stats) < 6:
                existing_stats.append(0.0)
            stats['accuracy_stats'] = tuple(existing_stats)
        else:
            # ensure the values are correct
            accuracy_stats = list(stats['accuracy_stats'])
            # ensure incorrect_predictions is calculated correctly
            if stats['true_label_count'] > 0:
                # recalculate incorrect predictions as total predictions with true labels minus correct predictions
                accuracy_stats[3] = accuracy_stats[1] - accuracy_stats[2]
            stats['accuracy_stats'] = tuple(accuracy_stats)
        
        # get digit-specific statistics
        try:
            cursor.execute("SELECT * FROM digit_stats")
            raw_digit_stats = cursor.fetchall()
            
            # convert Decimal values to Python native types
            stats['digit_stats'] = [
                tuple(float(val) if isinstance(val, (float, int)) or hasattr(val, 'as_integer_ratio') else val 
                      for val in row)
                for row in raw_digit_stats
            ]
        except Exception as e:
            # if the view doesn't exist yet, set to empty list
            print(f"Error getting digit stats: {e}")
            stats['digit_stats'] = []
        
        # get recent predictions
        cursor.execute("""
            SELECT timestamp, predicted_digit, confidence, true_label 
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT 40
        """)
        raw_recent = cursor.fetchall()
        
        # convert Decimal values to Python native types
        stats['recent_predictions'] = [
            tuple(float(val) if isinstance(val, (float, int)) or hasattr(val, 'as_integer_ratio') else val 
                  for val in row)
            for row in raw_recent
        ]
        
        # close cursor and connection
        cursor.close()
        conn.close()
        
        return stats
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return None 