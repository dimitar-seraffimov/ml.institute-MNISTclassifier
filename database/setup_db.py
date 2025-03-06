import os
import psycopg2
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

def setup_database():
    """
    Set up the database for the MNIST classifier application.
    This script will:
    1. Connect to the PostgreSQL server
    2. Create the database if it doesn't exist
    3. Create the tables and views
    """
    # Get connection parameters from environment variables
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    user = os.getenv('POSTGRES_USER', 'postgres')
    password = os.getenv('POSTGRES_PASSWORD', 'master')
    dbname = os.getenv('POSTGRES_DB', 'mnist_db')
    
    print(f"Setting up database '{dbname}' on {host}:{port} as user '{user}'")
    
    # First, connect to the 'postgres' database to create our database
    try:
        # Connect to the postgres database
        print("\nConnecting to 'postgres' database to create our application database...")
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname="postgres",
            user=user,
            password=password
        )
        conn.autocommit = True  # Needed for CREATE DATABASE
        cursor = conn.cursor()
        
        # Check if our database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}'")
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Creating database '{dbname}'...")
            cursor.execute(f"CREATE DATABASE {dbname}")
            print(f"Database '{dbname}' created successfully!")
        else:
            print(f"Database '{dbname}' already exists.")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error connecting to PostgreSQL server: {e}")
        print("\nPlease check your PostgreSQL server is running and your credentials are correct.")
        print("You may need to modify the .env file with the correct credentials.")
        return False
    
    # Now connect to our database and create tables
    try:
        # Connect to our database
        print(f"\nConnecting to '{dbname}' database to create tables...")
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        cursor = conn.cursor()
        
        # Check if predictions table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'predictions'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            print("Predictions table already exists. Checking for required columns...")
            
            # Check if true_label column exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'predictions' AND column_name = 'true_label'
                );
            """)
            true_label_exists = cursor.fetchone()[0]
            
            if not true_label_exists:
                print("Adding true_label column to predictions table...")
                cursor.execute("ALTER TABLE predictions ADD COLUMN true_label INTEGER;")
                conn.commit()
                print("Column added successfully!")
            else:
                print("true_label column already exists.")
            
            # Now create or replace the views
            print("Creating or replacing views...")
            
            # Create digit statistics view
            cursor.execute("""
                CREATE OR REPLACE VIEW digit_statistics AS
                SELECT 
                    predicted_digit,
                    COUNT(*) as prediction_count,
                    CAST(AVG(confidence) AS NUMERIC(10,4)) as avg_confidence
                FROM predictions
                GROUP BY predicted_digit
                ORDER BY predicted_digit;
            """)
            
            # Create accuracy statistics view
            cursor.execute("""
                CREATE OR REPLACE VIEW prediction_accuracy AS
                SELECT
                    COUNT(*) as total_predictions,
                    COUNT(CASE WHEN predicted_digit = true_label THEN 1 END) as correct_predictions,
                    COUNT(CASE WHEN predicted_digit != true_label AND true_label IS NOT NULL THEN 1 END) as incorrect_predictions,
                    COALESCE(
                        CAST(
                            100.0 * COUNT(CASE WHEN predicted_digit = true_label THEN 1 END) / 
                            NULLIF(COUNT(CASE WHEN true_label IS NOT NULL THEN 1 END), 0)
                        AS NUMERIC(10,2)), 
                        0
                    ) as accuracy_percentage,
                    CAST(AVG(confidence) AS NUMERIC(10,4)) as avg_confidence
                FROM predictions
                WHERE true_label IS NOT NULL;
            """)
            
            conn.commit()
            print("Views created successfully!")
        else:
            # Read SQL script
            with open('database/init.sql', 'r') as f:
                sql_script = f.read()
            
            # Execute SQL script
            print("Creating tables and views...")
            cursor.execute(sql_script)
            
            # Commit changes
            conn.commit()
            
            print("Tables and views created successfully!")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        print("\nDatabase setup completed successfully!")
        print(f"You can now connect to the database '{dbname}' on {host}:{port} as user '{user}'")
        return True
    except Exception as e:
        print(f"Error setting up database: {e}")
        return False

if __name__ == "__main__":
    setup_database() 