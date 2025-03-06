import psycopg2
import sys

def test_connection(host, port, dbname, user, password):
    """Test a specific database connection"""
    # Print connection parameters (without password)
    print(f"Testing connection to {dbname} on {host}:{port} as {user}")
    
    # Connect to database
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        
        # Get server version
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"Connection successful! PostgreSQL version: {version}")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return False

if __name__ == "__main__":
    # Default connection parameters
    host = "localhost"
    port = "5432"
    dbname = "mnist_db"
    user = "postgres"
    password = "master"
    
    # Try different connection parameters
    print("\n=== Testing with default parameters ===")
    test_connection(host, port, dbname, user, password)
    
    print("\n=== Testing with postgres database ===")
    test_connection(host, port, "postgres", user, password)
    
    print("\n=== Testing with hyphenated database name ===")
    test_connection(host, port, "mnist-db", user, password)
    
    # Ask for custom parameters
    print("\n=== Enter custom connection parameters ===")
    custom_host = input(f"Host [{host}]: ") or host
    custom_port = input(f"Port [{port}]: ") or port
    custom_dbname = input(f"Database name [{dbname}]: ") or dbname
    custom_user = input(f"User [{user}]: ") or user
    custom_password = input(f"Password [hidden]: ") or password
    
    print("\n=== Testing with custom parameters ===")
    test_connection(custom_host, custom_port, custom_dbname, custom_user, custom_password) 