import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_database():
    """Initialize the database with required tables and views"""
    # Database connection parameters
    db_params = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'mnist_db'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'master')
    }
    
    # Read SQL script
    with open('database/init.sql', 'r') as f:
        sql_script = f.read()
    
    # Connect to database
    try:
        print(f"Connecting to PostgreSQL database at {db_params['host']}:{db_params['port']}...")
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        # Execute SQL script
        print("Executing SQL script...")
        cursor.execute(sql_script)
        
        # Commit changes
        conn.commit()
        
        print("Database initialized successfully!")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    init_database() 