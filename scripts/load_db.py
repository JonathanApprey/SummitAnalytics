#!/usr/bin/env python3
"""
Summit Analytics - Data Loader
Demonstrates: SQL skills, data cleaning, and schema design.

This script loads the website_Traffic.csv into a SQLite database
with proper schema design and data validation.
"""

import sqlite3
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def create_database_schema(conn: sqlite3.Connection) -> None:
    """
    Create the database schema using raw SQL.
    Demonstrates SQL DDL skills.
    """
    cursor = conn.cursor()
    
    # Drop existing tables if they exist (for clean reload)
    cursor.execute("DROP TABLE IF EXISTS website_sessions")
    cursor.execute("DROP TABLE IF EXISTS traffic_sources")
    cursor.execute("DROP TABLE IF EXISTS session_metrics")
    
    # Create normalized tables with proper relationships
    # Table 1: Traffic Sources (dimension table)
    cursor.execute("""
        CREATE TABLE traffic_sources (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Table 2: Website Sessions (fact table)
    cursor.execute("""
        CREATE TABLE website_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            page_views INTEGER NOT NULL,
            session_duration REAL NOT NULL,
            bounce_rate REAL NOT NULL,
            time_on_page REAL NOT NULL,
            previous_visits INTEGER NOT NULL,
            conversion_rate REAL NOT NULL,
            -- Derived fields for analysis
            is_converted BOOLEAN GENERATED ALWAYS AS (conversion_rate >= 0.5) STORED,
            engagement_score REAL GENERATED ALWAYS AS (
                (page_views * 0.3) + (session_duration * 0.3) + 
                (time_on_page * 0.2) + ((1 - bounce_rate) * 0.2)
            ) STORED,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_id) REFERENCES traffic_sources(source_id)
        )
    """)
    
    # Create indexes for common query patterns
    cursor.execute("""
        CREATE INDEX idx_sessions_source ON website_sessions(source_id)
    """)
    cursor.execute("""
        CREATE INDEX idx_sessions_converted ON website_sessions(is_converted)
    """)
    cursor.execute("""
        CREATE INDEX idx_sessions_engagement ON website_sessions(engagement_score)
    """)
    
    conn.commit()
    logger.info("Database schema created successfully")


def load_traffic_sources(conn: sqlite3.Connection, df: pd.DataFrame) -> dict:
    """
    Load unique traffic sources and return mapping.
    """
    cursor = conn.cursor()
    
    unique_sources = df['Traffic Source'].unique()
    source_mapping = {}
    
    for source in unique_sources:
        cursor.execute(
            "INSERT INTO traffic_sources (source_name) VALUES (?)",
            (source,)
        )
        source_mapping[source] = cursor.lastrowid
    
    conn.commit()
    logger.info(f"Loaded {len(source_mapping)} traffic sources")
    return source_mapping


def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the dataset.
    Demonstrates data preparation skills.
    """
    logger.info(f"Starting data cleaning. Initial rows: {len(df)}")
    
    # Remove any rows with missing values
    initial_count = len(df)
    df = df.dropna()
    removed = initial_count - len(df)
    if removed > 0:
        logger.warning(f"Removed {removed} rows with missing values")
    
    # Validate numeric ranges
    # Page Views should be non-negative
    df = df[df['Page Views'] >= 0]
    
    # Bounce rate should be between 0 and 1
    df = df[(df['Bounce Rate'] >= 0) & (df['Bounce Rate'] <= 1)]
    
    # Conversion rate should be between 0 and 1
    df = df[(df['Conversion Rate'] >= 0) & (df['Conversion Rate'] <= 1)]
    
    # Session duration and time on page should be non-negative
    df = df[(df['Session Duration'] >= 0) & (df['Time on Page'] >= 0)]
    
    logger.info(f"Data cleaning complete. Final rows: {len(df)}")
    
    return df


def load_sessions(conn: sqlite3.Connection, df: pd.DataFrame, source_mapping: dict) -> None:
    """
    Load session data into the database.
    Uses batch inserts for performance.
    """
    cursor = conn.cursor()
    
    # Prepare data for bulk insert
    records = []
    for _, row in df.iterrows():
        records.append((
            source_mapping[row['Traffic Source']],
            int(row['Page Views']),
            float(row['Session Duration']),
            float(row['Bounce Rate']),
            float(row['Time on Page']),
            int(row['Previous Visits']),
            float(row['Conversion Rate'])
        ))
    
    # Bulk insert using executemany
    cursor.executemany("""
        INSERT INTO website_sessions (
            source_id, page_views, session_duration, bounce_rate,
            time_on_page, previous_visits, conversion_rate
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)
    
    conn.commit()
    logger.info(f"Loaded {len(records)} session records")


def create_analysis_views(conn: sqlite3.Connection) -> None:
    """
    Create SQL views for common analysis patterns.
    Demonstrates advanced SQL skills.
    """
    cursor = conn.cursor()
    
    # View 1: Traffic Source Performance Summary
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS vw_source_performance AS
        SELECT 
            ts.source_name,
            COUNT(*) as total_sessions,
            AVG(ws.page_views) as avg_page_views,
            AVG(ws.session_duration) as avg_session_duration,
            AVG(ws.bounce_rate) as avg_bounce_rate,
            AVG(ws.conversion_rate) as avg_conversion_rate,
            SUM(CASE WHEN ws.is_converted THEN 1 ELSE 0 END) as total_conversions,
            ROUND(
                100.0 * SUM(CASE WHEN ws.is_converted THEN 1 ELSE 0 END) / COUNT(*), 
                2
            ) as conversion_pct
        FROM website_sessions ws
        JOIN traffic_sources ts ON ws.source_id = ts.source_id
        GROUP BY ts.source_name
        ORDER BY avg_conversion_rate DESC
    """)
    
    # View 2: Engagement Quartiles
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS vw_engagement_segments AS
        SELECT 
            session_id,
            engagement_score,
            CASE 
                WHEN engagement_score >= (SELECT AVG(engagement_score) + STDEV(engagement_score) FROM website_sessions)
                    THEN 'High Engagement'
                WHEN engagement_score >= (SELECT AVG(engagement_score) FROM website_sessions)
                    THEN 'Medium Engagement'
                WHEN engagement_score >= (SELECT AVG(engagement_score) - STDEV(engagement_score) FROM website_sessions)
                    THEN 'Low Engagement'
                ELSE 'At Risk'
            END as engagement_segment
        FROM website_sessions
    """)
    
    # View 3: Visitor Loyalty Analysis
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS vw_visitor_loyalty AS
        SELECT 
            CASE 
                WHEN previous_visits = 0 THEN 'New Visitor'
                WHEN previous_visits BETWEEN 1 AND 2 THEN 'Returning'
                WHEN previous_visits BETWEEN 3 AND 5 THEN 'Frequent'
                ELSE 'Loyal'
            END as visitor_type,
            COUNT(*) as session_count,
            AVG(page_views) as avg_page_views,
            AVG(conversion_rate) as avg_conversion_rate,
            AVG(engagement_score) as avg_engagement_score
        FROM website_sessions
        GROUP BY 
            CASE 
                WHEN previous_visits = 0 THEN 'New Visitor'
                WHEN previous_visits BETWEEN 1 AND 2 THEN 'Returning'
                WHEN previous_visits BETWEEN 3 AND 5 THEN 'Frequent'
                ELSE 'Loyal'
            END
    """)
    
    conn.commit()
    logger.info("Analysis views created successfully")


def print_summary_stats(conn: sqlite3.Connection) -> None:
    """Print summary statistics from the loaded data."""
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("📊 SUMMIT ANALYTICS - DATABASE LOAD SUMMARY")
    print("="*60)
    
    # Total records
    cursor.execute("SELECT COUNT(*) FROM website_sessions")
    total = cursor.fetchone()[0]
    print(f"\n📈 Total Sessions Loaded: {total:,}")
    
    # Traffic source breakdown
    print("\n📡 Traffic Source Distribution:")
    cursor.execute("""
        SELECT source_name, total_sessions, avg_conversion_rate
        FROM vw_source_performance
        ORDER BY total_sessions DESC
    """)
    for row in cursor.fetchall():
        print(f"   • {row[0]}: {row[1]:,} sessions (Conv. Rate: {row[2]:.2%})")
    
    # Visitor loyalty
    print("\n👥 Visitor Loyalty Breakdown:")
    cursor.execute("SELECT * FROM vw_visitor_loyalty ORDER BY session_count DESC")
    for row in cursor.fetchall():
        print(f"   • {row[0]}: {row[1]:,} sessions (Avg Engagement: {row[4]:.2f})")
    
    print("\n" + "="*60)
    print("✅ Database ready for analysis!")
    print("="*60 + "\n")


def main():
    """Main entry point for the data loader."""
    project_root = get_project_root()
    data_file = project_root / "data" / "website_Traffic.csv"
    db_file = project_root / "data" / "summit.db"
    
    # Verify data file exists
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        sys.exit(1)
    
    logger.info(f"Loading data from: {data_file}")
    
    # Read CSV
    df = pd.read_csv(data_file)
    logger.info(f"Read {len(df)} records from CSV")
    
    # Clean data
    df = clean_and_validate_data(df)
    
    # Connect to database
    conn = sqlite3.connect(db_file)
    logger.info(f"Connected to database: {db_file}")
    
    try:
        # Create schema
        create_database_schema(conn)
        
        # Load traffic sources
        source_mapping = load_traffic_sources(conn, df)
        
        # Load sessions
        load_sessions(conn, df, source_mapping)
        
        # Create analysis views
        create_analysis_views(conn)
        
        # Print summary
        print_summary_stats(conn)
        
    finally:
        conn.close()
    
    logger.info("Data loading complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
