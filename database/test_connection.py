import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    """Test the database connection"""
    # Database connection parameters
    db_params = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'mnist_db'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'master')
    }
    
    # Print connection parameters (without password)
    print("Connection parameters:")
    for key, value in db_params.items():
        if key != 'password':
            print(f"  {key}: {value}")
    
    # Connect to database
    try:
        print(f"\nAttempting to connect to PostgreSQL database...")
        conn = psycopg2.connect(**db_params)
        
        # Get server version
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"Connection successful! PostgreSQL version: {version}")
        
        # Check if predictions table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'predictions'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            print("Predictions table exists.")
            
            # Get count of predictions
            cursor.execute("SELECT COUNT(*) FROM predictions;")
            count = cursor.fetchone()[0]
            print(f"Number of predictions in database: {count}")
        else:
            print("Predictions table does not exist.")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return False

if __name__ == "__main__":
    test_connection() 