#!/usr/bin/env python3
"""
Database Initialization Script for Fake News Detection System
============================================================
This script initializes the PostgreSQL database, creates all necessary tables,
and optionally seeds the database with sample data.

Usage:
    python init_db.py [options]

Options:
    --drop-tables    Drop existing tables before creating new ones
    --sample-data    Insert sample data after creating tables
    --reset          Drop tables, recreate, and add sample data
    --test-only      Only test database connection without making changes
    --fix-permissions Fix user permissions for the database
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
import traceback
import getpass

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Import our database models and repository
    from database.models import Base, GNewsArticle, ArticleHash
    from database.repository import SessionLocal, engine
    
    print("✅ Successfully imported all required modules")
    
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("\nPlease install missing dependencies:")
    print("pip install psycopg2-binary sqlalchemy python-dotenv")
    sys.exit(1)

# Database configuration
DEFAULT_DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'fake_news_user',
    'password': 'your_secure_password',
    'database': 'fake_news_db'
}

def get_db_config():
    """Get database configuration from environment variables or defaults"""
    database_url = os.getenv('DATABASE_URL')
    
    if database_url:
        # Parse DATABASE_URL if provided
        # Format: postgresql://user:password@host:port/database
        try:
            import re
            match = re.match(r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', database_url)
            if match:
                user, password, host, port, database = match.groups()
                return {
                    'host': host,
                    'port': int(port),
                    'user': user,
                    'password': password,
                    'database': database
                }
        except Exception as e:
            print(f"⚠ Warning: Could not parse DATABASE_URL: {e}")
    
    # Use individual environment variables or defaults
    return {
        'host': os.getenv('DB_HOST', DEFAULT_DB_CONFIG['host']),
        'port': int(os.getenv('DB_PORT', DEFAULT_DB_CONFIG['port'])),
        'user': os.getenv('DB_USER', DEFAULT_DB_CONFIG['user']),
        'password': os.getenv('DB_PASSWORD', DEFAULT_DB_CONFIG['password']),
        'database': os.getenv('DB_NAME', DEFAULT_DB_CONFIG['database'])
    }

def test_postgresql_connection():
    """Test if PostgreSQL server is running and accessible"""
    config = get_db_config()
    
    try:
        # Connect to PostgreSQL server (not to specific database)
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            database='postgres'  # Connect to default postgres database
        )
        conn.close()
        print("✅ PostgreSQL server connection successful")
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("\nTroubleshooting steps:")
        print("1. Ensure PostgreSQL is installed and running")
        print("2. Check if the user and password are correct")
        print("3. Verify the host and port are accessible")
        print(f"4. Current config: {config['user']}@{config['host']}:{config['port']}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error testing PostgreSQL: {e}")
        return False

def get_superuser_connection():
    """Try to get a superuser connection for administrative tasks"""
    config = get_db_config()
    
    # Try different superuser approaches
    superuser_options = [
        {'user': 'postgres', 'password': None},  # Default postgres user
        {'user': 'postgres', 'password': ''},     # Empty password
        {'user': 'postgres', 'password': 'postgres'},  # Common default
    ]
    
    # Ask user for postgres password if needed
    print("\n🔑 To fix permissions, we need superuser access.")
    print("Trying common postgres superuser configurations...")
    
    for option in superuser_options:
        try:
            conn = psycopg2.connect(
                host=config['host'],
                port=config['port'],
                user=option['user'],
                password=option['password'],
                database='postgres'
            )
            print(f"✅ Connected as superuser: {option['user']}")
            return conn
        except:
            continue
    
    # If all defaults failed, ask user for postgres password
    for attempt in range(3):
        try:
            postgres_password = getpass.getpass("Enter postgres superuser password (or press Enter to skip): ")
            if not postgres_password.strip():
                print("⚠ Skipping superuser connection")
                return None
                
            conn = psycopg2.connect(
                host=config['host'],
                port=config['port'],
                user='postgres',
                password=postgres_password,
                database='postgres'
            )
            print("✅ Connected as postgres superuser")
            return conn
        except psycopg2.OperationalError as e:
            print(f"❌ Failed to connect as postgres: {e}")
            if attempt < 2:
                print("Please try again...")
    
    return None

def fix_user_permissions():
    """Fix user permissions to allow table creation"""
    config = get_db_config()
    
    print(f"🔧 Attempting to fix permissions for user '{config['user']}'...")
    
    # Try to get superuser connection
    superuser_conn = get_superuser_connection()
    if not superuser_conn:
        print("❌ Cannot fix permissions without superuser access")
        print("\nManual fix options:")
        print("1. Connect to PostgreSQL as postgres user:")
        print(f"   psql -U postgres -d {config['database']}")
        print("2. Run these commands:")
        print(f"   GRANT CREATE ON SCHEMA public TO {config['user']};")
        print(f"   GRANT USAGE ON SCHEMA public TO {config['user']};")
        print(f"   GRANT ALL PRIVILEGES ON DATABASE {config['database']} TO {config['user']};")
        print(f"   ALTER USER {config['user']} CREATEDB;")
        return False
    
    try:
        superuser_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = superuser_conn.cursor()
        
        # Create user if not exists
        try:
            cursor.execute(
                sql.SQL("CREATE USER {} WITH PASSWORD %s").format(
                    sql.Identifier(config['user'])
                ),
                (config['password'],)
            )
            print(f"✅ User '{config['user']}' created")
        except psycopg2.errors.DuplicateObject:
            print(f"ℹ User '{config['user']}' already exists")
        
        # Grant database privileges
        cursor.execute(
            sql.SQL("GRANT ALL PRIVILEGES ON DATABASE {} TO {}").format(
                sql.Identifier(config['database']),
                sql.Identifier(config['user'])
            )
        )
        print(f"✅ Database privileges granted to '{config['user']}'")
        
        # Grant schema privileges
        cursor.execute(
            sql.SQL("GRANT CREATE ON SCHEMA public TO {}").format(
                sql.Identifier(config['user'])
            )
        )
        print(f"✅ CREATE privilege on public schema granted")
        
        cursor.execute(
            sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(
                sql.Identifier(config['user'])
            )
        )
        print(f"✅ USAGE privilege on public schema granted")
        
        # Allow user to create databases (useful for testing)
        cursor.execute(
            sql.SQL("ALTER USER {} CREATEDB").format(
                sql.Identifier(config['user'])
            )
        )
        print(f"✅ CREATEDB privilege granted to '{config['user']}'")
        
        # Grant default privileges for future tables
        cursor.execute(
            sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {}").format(
                sql.Identifier(config['user'])
            )
        )
        print(f"✅ Default table privileges granted")
        
        cursor.close()
        superuser_conn.close()
        
        print("🎉 Permissions fixed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing permissions: {e}")
        return False

def create_database_if_not_exists():
    """Create the database if it doesn't exist"""
    config = get_db_config()
    
    try:
        # Try with regular user first
        try:
            conn = psycopg2.connect(
                host=config['host'],
                port=config['port'],
                user=config['user'],
                password=config['password'],
                database='postgres'
            )
        except:
            # If that fails, try superuser
            conn = get_superuser_connection()
            if not conn:
                print("❌ Cannot connect to create database")
                return False
        
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
            (config['database'],)
        )
        exists = cursor.fetchone()
        
        if not exists:
            # Create database
            cursor.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(config['database']),
                    sql.Identifier(config['user'])
                )
            )
            print(f"✅ Database '{config['database']}' created successfully")
        else:
            print(f"ℹ Database '{config['database']}' already exists")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False

def test_database_connection():
    """Test connection to the specific database"""
    try:
        # Test SQLAlchemy connection
        session = SessionLocal()
        result = session.execute(text("SELECT version()")).fetchone()
        session.close()
        
        print("✅ Database connection successful")
        print(f"📊 PostgreSQL version: {result[0].split(',')[0] if result else 'Unknown'}")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_table_creation_permissions():
    """Test if user can create tables"""
    try:
        session = SessionLocal()
        
        # Try to create a simple test table
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS permission_test (
                id SERIAL PRIMARY KEY,
                test_column VARCHAR(50)
            )
        """))
        
        # Drop the test table
        session.execute(text("DROP TABLE IF EXISTS permission_test"))
        session.commit()
        session.close()
        
        print("✅ Table creation permissions verified")
        return True
        
    except Exception as e:
        print(f"❌ Cannot create tables: {e}")
        print("🔧 Use --fix-permissions to attempt automatic fix")
        return False

def drop_all_tables():
    """Drop all existing tables"""
    try:
        print("🗑 Dropping all existing tables...")
        Base.metadata.drop_all(engine)
        print("✅ All tables dropped successfully")
        return True
    except Exception as e:
        print(f"❌ Error dropping tables: {e}")
        return False

def create_all_tables():
    """Create all database tables"""
    try:
        print("🏗 Creating database tables...")
        Base.metadata.create_all(engine)
        print("✅ All tables created successfully")
        
        # Verify tables were created
        session = SessionLocal()
        
        # Check if tables exist
        tables_query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = session.execute(tables_query).fetchall()
        session.close()
        
        if tables:
            print("📋 Created tables:")
            for table in tables:
                print(f"   - {table[0]}")
        else:
            print("⚠ Warning: No tables found after creation")
            
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        print(traceback.format_exc())
        return False

def insert_sample_data():
    """Insert sample data for testing"""
    try:
        print("📝 Inserting sample data...")
        session = SessionLocal()
        
        # Sample articles
        sample_articles = [
            {
                'title': 'Climate Change Report Shows Rising Global Temperatures',
                'description': 'New scientific report confirms accelerating climate change with rising global temperatures and extreme weather events.',
                'content': 'A comprehensive climate report released today by international scientists confirms that global temperatures continue to rise at an accelerating pace. The report, based on data from over 100 research institutions worldwide, shows that average temperatures have increased by 1.2 degrees Celsius since pre-industrial times.',
                'url': 'https://example-news.com/climate-report-2024',
                'source_name': 'Science Daily',
                'published_at': datetime.now() - timedelta(days=1)
            },
            {
                'title': 'New Technology Breakthrough in Renewable Energy',
                'description': 'Scientists develop more efficient solar panels that could revolutionize renewable energy production.',
                'content': 'Researchers at the International Energy Institute have announced a breakthrough in solar panel technology that could dramatically improve renewable energy efficiency. The new panels use advanced materials to achieve 45% efficiency rates, compared to 20% for traditional panels.',
                'url': 'https://example-tech.com/solar-breakthrough',
                'source_name': 'Tech Innovation Weekly',
                'published_at': datetime.now() - timedelta(days=2)
            },
            {
                'title': 'Economic Markets Show Signs of Recovery',
                'description': 'Global markets demonstrate positive trends as economic indicators improve across major economies.',
                'content': 'Financial markets around the world are showing encouraging signs of recovery as key economic indicators point to sustained growth. Major stock indices have gained 15% over the past quarter, while unemployment rates continue to decline in developed nations.',
                'url': 'https://example-finance.com/market-recovery',
                'source_name': 'Financial Times',
                'published_at': datetime.now() - timedelta(days=3)
            }
        ]
        
        # Insert articles
        for article_data in sample_articles:
            article = GNewsArticle(**article_data)
            session.add(article)
            
            # Also create hash entries
            content_hash = f"sample_hash_{hash(article_data['title'])}"
            hash_entry = ArticleHash(hash=content_hash)
            session.add(hash_entry)
        
        session.commit()
        
        # Verify insertion
        article_count = session.query(GNewsArticle).count()
        hash_count = session.query(ArticleHash).count()
        
        session.close()
        
        print(f"✅ Sample data inserted successfully")
        print(f"   - {article_count} articles")
        print(f"   - {hash_count} hash entries")
        
        return True
        
    except Exception as e:
        print(f"❌ Error inserting sample data: {e}")
        print(traceback.format_exc())
        return False

def get_database_statistics():
    """Get and display database statistics"""
    try:
        session = SessionLocal()
        
        # Get table statistics
        article_count = session.query(GNewsArticle).count()
        hash_count = session.query(ArticleHash).count()
        
        # Get database size
        db_size_query = text("SELECT pg_size_pretty(pg_database_size(current_database()))")
        db_size = session.execute(db_size_query).fetchone()[0]
        
        session.close()
        
        print("\n📊 Database Statistics:")
        print(f"   Articles stored: {article_count}")
        print(f"   Hash entries: {hash_count}")
        print(f"   Database size: {db_size}")
        
        return True
        
    except Exception as e:
        print(f"⚠ Could not retrieve database statistics: {e}")
        return False

def main():
    """Main initialization function"""
    parser = argparse.ArgumentParser(description='Initialize Fake News Detection Database')
    parser.add_argument('--drop-tables', action='store_true', 
                       help='Drop existing tables before creating new ones')
    parser.add_argument('--sample-data', action='store_true',
                       help='Insert sample data after creating tables')
    parser.add_argument('--reset', action='store_true',
                       help='Complete reset: drop tables, recreate, and add sample data')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test database connection without making changes')
    parser.add_argument('--fix-permissions', action='store_true',
                       help='Fix user permissions for the database')
    
    args = parser.parse_args()
    
    # Handle reset option
    if args.reset:
        args.drop_tables = True
        args.sample_data = True
    
    print("🚀 Fake News Detection Database Initialization")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display configuration
    config = get_db_config()
    print(f"\n📋 Database Configuration:")
    print(f"   Host: {config['host']}:{config['port']}")
    print(f"   User: {config['user']}")
    print(f"   Database: {config['database']}")
    
    # Step 1: Test PostgreSQL connection
    print(f"\n🔍 Step 1: Testing PostgreSQL connection...")
    if not test_postgresql_connection():
        print("❌ Cannot proceed without PostgreSQL connection")
        return False
    
    # Step 2: Fix permissions if requested
    if args.fix_permissions:
        print(f"\n🔧 Step 2: Fixing user permissions...")
        if not fix_user_permissions():
            print("❌ Failed to fix permissions")
            return False
    
    # Step 3: Create database if needed
    print(f"\n🏗 Step 3: Creating database if needed...")
    if not create_database_if_not_exists():
        print("❌ Cannot proceed without database")
        return False
    
    # Step 4: Test specific database connection
    print(f"\n🔗 Step 4: Testing database connection...")
    if not test_database_connection():
        print("❌ Cannot proceed without database connection")
        return False
    
    # Step 5: Test table creation permissions
    print(f"\n🔑 Step 5: Testing table creation permissions...")
    if not test_table_creation_permissions():
        print("❌ Insufficient permissions to create tables")
        print("💡 Try running with --fix-permissions flag")
        return False
    
    # If test-only, stop here
    if args.test_only:
        print("\n✅ Database connection and permissions test completed successfully!")
        get_database_statistics()
        return True
    
    # Step 6: Drop tables if requested
    if args.drop_tables:
        print(f"\n🗑 Step 6: Dropping existing tables...")
        if not drop_all_tables():
            print("⚠ Warning: Could not drop tables, continuing...")
    
    # Step 7: Create tables
    print(f"\n🏗 Step 7: Creating database tables...")
    if not create_all_tables():
        print("❌ Failed to create tables")
        return False
    
    # Step 8: Insert sample data if requested
    if args.sample_data:
        print(f"\n📝 Step 8: Inserting sample data...")
        if not insert_sample_data():
            print("⚠ Warning: Could not insert sample data")
    
    # Step 9: Show final statistics
    print(f"\n📊 Step 9: Database statistics...")
    get_database_statistics()
    
    print("\n" + "=" * 50)
    print("🎉 Database initialization completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Start the Flask application: python app.py")
    print("2. Test the API endpoints")
    print("3. Load your Chrome extension")
    print("\n💡 Use --help to see all available options")
    
    return True

if __name__ == "_main_":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Initialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during initialization: {e}")
        print(traceback.format_exc())
        sys.exit(1)