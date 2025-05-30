#!/usr/bin/env python3
"""
Test database connection before running the main application
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

def test_database_connection():
    """Test if we can connect to the database."""
    load_dotenv()
    
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/fake_news_db')
    
    try:
        # Create engine
        engine = create_engine(DATABASE_URL)
        
        # Test connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("✓ Database connection successful!")
            return True
            
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("\nPlease check:")
        print("1. PostgreSQL is running")
        print("2. Database exists")
        print("3. Credentials are correct in .env file")
        print("4. DATABASE_URL format: postgresql://user:password@host:port/dbname")
        return False

if __name__ == "__main__":
    test_database_connection()