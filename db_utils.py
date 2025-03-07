"""
Database utilities for the MNIST Classifier application.
Handles connections to PostgreSQL, including Cloud SQL.
"""
import os
import psycopg2
import streamlit as st

def get_db_params():
    """Get database connection parameters with Cloud SQL support"""
    # Database connection parameters
    db_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'mnist_db'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    # Check if we're running on Google Cloud Run
    running_in_cloud_run = os.environ.get('K_SERVICE') is not None
    
    # If we're running on Cloud Run, and a Cloud SQL instance connection name is provided,
    # use the Cloud SQL Proxy unix socket to connect
    db_instance_connection_name = os.getenv('DB_INSTANCE_CONNECTION_NAME', '')
    if running_in_cloud_run and db_instance_connection_name:
        db_params['host'] = f"/cloudsql/{db_instance_connection_name}"
    
    # Support for DATABASE_URL format (used by many cloud providers)
    database_url = os.getenv('DATABASE_URL', '')
    if database_url:
        # Parse DATABASE_URL and override db_params
        try:
            # Format: postgres://username:password@hostname:port/database
            import re
            pattern = r'postgres://(?P<user>[^:]+):(?P<password>[^@]+)@(?P<host>[^:]+):(?P<port>\d+)/(?P<database>.+)'
            match = re.match(pattern, database_url)
            if match:
                db_params = match.groupdict()
        except Exception as e:
            print(f"Failed to parse DATABASE_URL: {e}")
    
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
        # Connect to database
        conn = connect_to_database()
        if not conn:
            print("Could not connect to database.")
            return False
            
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
            print("Creating database tables...")
            
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
                CREATE OR REPLACE VIEW accuracy_stats AS
                SELECT 
                    COUNT(*) AS total_predictions,
                    COUNT(true_label) AS predictions_with_true_label,
                    CASE 
                        WHEN COUNT(true_label) > 0 THEN 
                            CAST(SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) AS NUMERIC) / COUNT(true_label) * 100
                        ELSE 0 
                    END AS accuracy,
                    CAST(AVG(confidence) AS NUMERIC(10,4)) AS avg_confidence
                FROM predictions
            """)
            
            cursor.execute("""
                CREATE OR REPLACE VIEW digit_stats AS
                SELECT 
                    predicted_digit AS digit,
                    COUNT(*) AS count,
                    CAST(AVG(confidence) AS NUMERIC(10,4)) AS avg_confidence
                FROM predictions
                GROUP BY predicted_digit
                ORDER BY predicted_digit
            """)
            
            # Commit changes
            conn.commit()
            print("Database tables created successfully!")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Database setup error: {e}")
        return False

def log_prediction(predicted_digit, confidence, true_label=None, image=None):
    """Log prediction to PostgreSQL database"""
    try:
        # Convert image to binary data if provided
        image_data = None
        if image is not None:
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            image_data = img_byte_arr.getvalue()
        
        # Convert numpy types to Python native types
        if hasattr(predicted_digit, 'item'):
            predicted_digit = predicted_digit.item()
        if hasattr(confidence, 'item'):
            confidence = confidence.item()
        if true_label is not None and hasattr(true_label, 'item'):
            true_label = true_label.item()
        
        # Connect to database
        conn = connect_to_database()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Insert prediction into database
        cursor.execute("""
            INSERT INTO predictions (timestamp, predicted_digit, confidence, true_label, image_data)
            VALUES (NOW(), %s, %s, %s, %s)
        """, (predicted_digit, confidence, true_label, image_data))
        
        # Commit changes
        conn.commit()
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        # Log error but don't crash the application
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
        
        # Get total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        stats['total_predictions'] = cursor.fetchone()[0]
        
        # Get count of predictions with true labels
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE true_label IS NOT NULL")
        stats['true_label_count'] = cursor.fetchone()[0]
        
        # Get accuracy statistics
        try:
            cursor.execute("SELECT * FROM accuracy_stats")
            raw_stats = cursor.fetchone()
            
            # Convert Decimal values to Python native types
            if raw_stats:
                stats['accuracy_stats'] = tuple(
                    float(val) if isinstance(val, (float, int)) or hasattr(val, 'as_integer_ratio') else val 
                    for val in raw_stats
                )
            else:
                stats['accuracy_stats'] = None
        except Exception as e:
            # If the view doesn't exist yet, set to None
            print(f"Error getting accuracy stats: {e}")
            stats['accuracy_stats'] = None
        
        # If no true labels, set default values
        if not stats['true_label_count'] or not stats['accuracy_stats']:
            stats['accuracy_stats'] = (stats['total_predictions'], 0, 0, 0)
        
        # Get digit-specific statistics
        try:
            cursor.execute("SELECT * FROM digit_stats")
            raw_digit_stats = cursor.fetchall()
            
            # Convert Decimal values to Python native types
            stats['digit_stats'] = [
                tuple(float(val) if isinstance(val, (float, int)) or hasattr(val, 'as_integer_ratio') else val 
                      for val in row)
                for row in raw_digit_stats
            ]
        except Exception as e:
            # If the view doesn't exist yet, set to empty list
            print(f"Error getting digit stats: {e}")
            stats['digit_stats'] = []
        
        # Get recent predictions
        cursor.execute("""
            SELECT id, predicted_digit, confidence, true_label 
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT 5
        """)
        raw_recent = cursor.fetchall()
        
        # Convert Decimal values to Python native types
        stats['recent_predictions'] = [
            tuple(float(val) if isinstance(val, (float, int)) or hasattr(val, 'as_integer_ratio') else val 
                  for val in row)
            for row in raw_recent
        ]
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return stats
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return None 