#!/usr/bin/env python3
"""
Summit Analytics - Analysis Engine
Demonstrates: Statistical techniques, clustering, segmentation strategies.

This module provides the core analytics functionality including:
- RFM-style analysis adapted for website traffic (Recency → Engagement, Frequency → Visits, Monetary → Conversion)
- K-Means clustering for visitor segmentation
- Predictive scoring for conversion likelihood
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path
from typing import Tuple, Dict, List
import logging
from dataclasses import dataclass
from src.predictive_model import ConversionPredictor
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SegmentProfile:
    """Data class for segment profiles."""
    name: str
    count: int
    avg_page_views: float
    avg_session_duration: float
    avg_bounce_rate: float
    avg_conversion_rate: float
    avg_engagement: float
    recommendation: str


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_data_from_db() -> pd.DataFrame:
    """
    Load session data from SQLite database.
    """
    db_path = get_project_root() / "data" / "summit.db"
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run load_db.py first.")
    
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT 
            ws.session_id,
            ts.source_name as traffic_source,
            ws.page_views,
            ws.session_duration,
            ws.bounce_rate,
            ws.time_on_page,
            ws.previous_visits,
            ws.conversion_rate,
            ws.is_converted,
            ws.engagement_score
        FROM website_sessions ws
        JOIN traffic_sources ts ON ws.source_id = ts.source_id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} sessions from database")
    return df


def calculate_efm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate EFM (Engagement, Frequency, Monetary) scores.
    This is an adaptation of RFM analysis for website traffic:
    - E (Engagement): Based on page views, session duration, time on page
    - F (Frequency): Based on previous visits
    - M (Monetary): Based on conversion rate
    
    Returns DataFrame with EFM scores (1-5 scale, 5 being best)
    """
    logger.info("Calculating EFM scores...")
    
    # Create a copy to avoid modifying original
    result = df.copy()
    
    # Calculate composite engagement score
    result['e_raw'] = (
        result['page_views'].rank(pct=True) * 0.3 +
        result['session_duration'].rank(pct=True) * 0.3 +
        result['time_on_page'].rank(pct=True) * 0.2 +
        (1 - result['bounce_rate'].rank(pct=True)) * 0.2
    )
    
    # Frequency score based on previous visits
    result['f_raw'] = result['previous_visits'].rank(pct=True)
    
    # Monetary score based on conversion rate
    result['m_raw'] = result['conversion_rate'].rank(pct=True)
    
    # Helper function to safely bin scores into 1-5
    def safe_score(series, name):
        """Convert percentile ranks to 1-5 scores safely handling duplicates."""
        try:
            # Try qcut first (equal frequency bins)
            return pd.qcut(series, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(float).fillna(3).astype(int)
        except ValueError:
            # Fall back to cut (equal width bins) if too many duplicates
            try:
                return pd.cut(series, bins=5, labels=[1, 2, 3, 4, 5]).astype(float).fillna(3).astype(int)
            except ValueError:
                # Last resort: use percentile-based manual binning
                logger.warning(f"Using fallback binning for {name}")
                conditions = [
                    series <= series.quantile(0.2),
                    series <= series.quantile(0.4),
                    series <= series.quantile(0.6),
                    series <= series.quantile(0.8),
                    series <= series.quantile(1.0)
                ]
                return np.select(conditions, [1, 2, 3, 4, 5], default=3)
    
    # Apply scoring with fallback handling
    result['E_Score'] = safe_score(result['e_raw'], 'Engagement')
    result['F_Score'] = safe_score(result['f_raw'], 'Frequency')
    result['M_Score'] = safe_score(result['m_raw'], 'Monetary')
    
    # Calculate composite EFM score
    result['EFM_Score'] = result['E_Score'] + result['F_Score'] + result['M_Score']
    
    # Clean up temporary columns
    result = result.drop(columns=['e_raw', 'f_raw', 'm_raw'])
    
    logger.info("EFM scores calculated successfully")
    return result


def find_optimal_clusters(X: np.ndarray, max_k: int = 10) -> int:
    """
    Find optimal number of clusters using silhouette score.
    """
    silhouette_scores = []
    K_range = range(2, min(max_k + 1, len(X)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    optimal_k = list(K_range)[np.argmax(silhouette_scores)]
    logger.info(f"Optimal number of clusters: {optimal_k} (silhouette: {max(silhouette_scores):.3f})")
    
    return optimal_k


def perform_clustering(df: pd.DataFrame, n_clusters: int = None) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Perform K-Means clustering on visitor behavior.
    
    Features used:
    - Page views
    - Session duration
    - Bounce rate
    - Time on page
    - Previous visits
    - Conversion rate
    """
    logger.info("Performing visitor segmentation via K-Means clustering...")
    
    # Select features for clustering
    feature_cols = [
        'page_views', 'session_duration', 'bounce_rate',
        'time_on_page', 'previous_visits', 'conversion_rate'
    ]
    
    X = df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal clusters if not specified
    if n_clusters is None:
        n_clusters = find_optimal_clusters(X_scaled)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate cluster distances (for confidence scoring)
    distances = kmeans.transform(X_scaled)
    df['cluster_distance'] = np.min(distances, axis=1)
    
    logger.info(f"Clustering complete. Created {n_clusters} segments.")
    return df, kmeans, scaler


def generate_segment_profiles(df: pd.DataFrame) -> List[SegmentProfile]:
    """
    Generate human-readable profiles for each segment.
    """
    logger.info("Generating segment profiles...")
    
    profiles = []
    
    # Define segment naming based on characteristics
    segment_names = {
        'high_value': 'High-Value Converters',
        'engaged_browsers': 'Engaged Browsers',
        'returning_visitors': 'Loyal Returners',
        'at_risk': 'At-Risk Visitors',
        'new_potential': 'New with Potential'
    }
    
    recommendations = {
        'high_value': 'Focus on retention and upselling. These visitors are your champions.',
        'engaged_browsers': 'Implement targeted CTAs. They browse but need nudging to convert.',
        'returning_visitors': 'Reward loyalty with exclusive offers. Build on existing relationship.',
        'at_risk': 'Re-engagement campaigns needed. Consider exit-intent popups.',
        'new_potential': 'Nurture with educational content. Build trust before asking for conversion.'
    }
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        
        # Calculate averages
        avg_pv = cluster_data['page_views'].mean()
        avg_sd = cluster_data['session_duration'].mean()
        avg_br = cluster_data['bounce_rate'].mean()
        avg_cr = cluster_data['conversion_rate'].mean()
        avg_eng = cluster_data['engagement_score'].mean()
        avg_visits = cluster_data['previous_visits'].mean()
        
        # Determine segment type based on characteristics
        if avg_cr > 0.8 and avg_eng > df['engagement_score'].median():
            segment_type = 'high_value'
        elif avg_eng > df['engagement_score'].mean() and avg_cr < 0.7:
            segment_type = 'engaged_browsers'
        elif avg_visits > df['previous_visits'].mean():
            segment_type = 'returning_visitors'
        elif avg_br > df['bounce_rate'].mean() or avg_cr < 0.5:
            segment_type = 'at_risk'
        else:
            segment_type = 'new_potential'
        
        profile = SegmentProfile(
            name=f"Segment {cluster_id + 1}: {segment_names[segment_type]}",
            count=len(cluster_data),
            avg_page_views=round(avg_pv, 2),
            avg_session_duration=round(avg_sd, 2),
            avg_bounce_rate=round(avg_br, 3),
            avg_conversion_rate=round(avg_cr, 3),
            avg_engagement=round(avg_eng, 2),
            recommendation=recommendations[segment_type]
        )
        profiles.append(profile)
    
    return profiles


def calculate_conversion_probability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a simple conversion probability score based on behavioral patterns.
    This is a heuristic model demonstrating predictive thinking.
    """
    logger.info("Calculating conversion probability scores...")
    
    df = df.copy()
    
    # Normalize features to 0-1 scale
    df['norm_pv'] = (df['page_views'] - df['page_views'].min()) / (df['page_views'].max() - df['page_views'].min())
    df['norm_sd'] = (df['session_duration'] - df['session_duration'].min()) / (df['session_duration'].max() - df['session_duration'].min())
    df['norm_visits'] = (df['previous_visits'] - df['previous_visits'].min()) / (df['previous_visits'].max() - df['previous_visits'].min())
    
    # Inverse of bounce rate (lower is better)
    df['norm_br_inv'] = 1 - df['bounce_rate']
    
    # Weighted probability score
    df['conversion_probability'] = (
        df['norm_pv'] * 0.20 +
        df['norm_sd'] * 0.25 +
        df['norm_visits'] * 0.20 +
        df['norm_br_inv'] * 0.15 +
        df['conversion_rate'] * 0.20  # Historical conversion rate
    )
    
    # Clean up temp columns
    df = df.drop(columns=['norm_pv', 'norm_sd', 'norm_visits', 'norm_br_inv'])
    
    logger.info("Conversion probability calculated")
    return df


def get_traffic_source_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance by traffic source.
    """
    logger.info("Analyzing traffic sources...")
    
    source_analysis = df.groupby('traffic_source').agg({
        'session_id': 'count',
        'page_views': 'mean',
        'session_duration': 'mean',
        'bounce_rate': 'mean',
        'conversion_rate': 'mean',
        'engagement_score': 'mean',
        'is_converted': 'sum'
    }).rename(columns={
        'session_id': 'total_sessions',
        'page_views': 'avg_page_views',
        'session_duration': 'avg_session_duration',
        'bounce_rate': 'avg_bounce_rate',
        'conversion_rate': 'avg_conversion_rate',
        'engagement_score': 'avg_engagement',
        'is_converted': 'total_conversions'
    })
    
    # Calculate conversion percentage
    source_analysis['conversion_pct'] = (
        source_analysis['total_conversions'] / source_analysis['total_sessions'] * 100
    ).round(2)
    
    # Rank sources by effectiveness (composite score)
    source_analysis['effectiveness_score'] = (
        source_analysis['avg_engagement'].rank(pct=True) * 0.4 +
        source_analysis['avg_conversion_rate'].rank(pct=True) * 0.4 +
        (1 - source_analysis['avg_bounce_rate'].rank(pct=True)) * 0.2
    )
    
    return source_analysis.sort_values('effectiveness_score', ascending=False)


def run_full_analysis() -> Dict:
    """
    Run the complete analysis pipeline and return results.
    """
    logger.info("Starting full analysis pipeline...")
    
    # Load data
    df = load_data_from_db()
    
    # Calculate EFM scores
    df = calculate_efm_scores(df)
    
    # Perform clustering
    df, kmeans_model, scaler = perform_clustering(df)
    
    # Generate segment profiles
    profiles = generate_segment_profiles(df)
    
    # Calculate conversion probability
    df = calculate_conversion_probability(df)
    
    # Traffic source analysis
    source_analysis = get_traffic_source_analysis(df)
    
    # Train Predictive Model
    logger.info("Training predictive model...")
    predictor = ConversionPredictor()
    model_metrics = predictor.train(df)
    logger.info(f"Model trained. Accuracy: {model_metrics['accuracy']:.2f}, AUC: {model_metrics['roc_auc']:.2f}")
    
    # Compile results
    results = {
        'data': df,
        'segment_profiles': profiles,
        'source_analysis': source_analysis,
        'kmeans_model': kmeans_model,
        'scaler': scaler,
        'predictor': predictor,
        'model_metrics': model_metrics,
        'summary': {
            'total_sessions': len(df),
            'total_conversions': df['is_converted'].sum(),
            'overall_conversion_rate': df['conversion_rate'].mean(),
            'avg_engagement': df['engagement_score'].mean(),
            'num_segments': len(profiles),
            'top_traffic_source': source_analysis.index[0]
        }
    }
    
    logger.info("Analysis pipeline complete!")
    return results


def save_results_to_db(results: Dict) -> None:
    """
    Save analysis results back to the database for dashboard consumption.
    """
    db_path = get_project_root() / "data" / "summit.db"
    conn = sqlite3.connect(db_path)
    
    # Save enriched session data
    df = results['data']
    df.to_sql('analyzed_sessions', conn, if_exists='replace', index=False)
    
    # Save source analysis
    source_df = results['source_analysis'].reset_index()
    source_df.to_sql('source_analysis', conn, if_exists='replace', index=False)
    
    # Save predictive model
    model_path = get_project_root() / "data" / "conversion_model.joblib"
    results['predictor'].save_model(model_path)
    
    conn.close()
    logger.info("Analysis results saved to database")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("🔬 SUMMIT ANALYTICS - ANALYSIS ENGINE")
    print("="*60 + "\n")
    
    # Run analysis
    results = run_full_analysis()
    
    # Save results
    save_results_to_db(results)
    
    # Print summary
    print("\n📊 ANALYSIS SUMMARY")
    print("-" * 40)
    summary = results['summary']
    print(f"Total Sessions Analyzed: {summary['total_sessions']:,}")
    print(f"Total Conversions: {summary['total_conversions']:,}")
    print(f"Overall Conversion Rate: {summary['overall_conversion_rate']:.2%}")
    print(f"Average Engagement Score: {summary['avg_engagement']:.2f}")
    print(f"Number of Segments Created: {summary['num_segments']}")
    print(f"Top Performing Source: {summary['top_traffic_source']}")
    
    print("\n👥 SEGMENT PROFILES")
    print("-" * 40)
    for profile in results['segment_profiles']:
        print(f"\n{profile.name}")
        print(f"  • Sessions: {profile.count:,}")
        print(f"  • Avg Page Views: {profile.avg_page_views}")
        print(f"  • Avg Conversion Rate: {profile.avg_conversion_rate:.2%}")
        print(f"  • Recommendation: {profile.recommendation}")
    
    print("\n📡 TRAFFIC SOURCE EFFECTIVENESS")
    print("-" * 40)
    for source, row in results['source_analysis'].iterrows():
        print(f"{source}: {row['total_sessions']:,.0f} sessions, "
              f"{row['avg_conversion_rate']:.2%} conv. rate, "
              f"Effectiveness: {row['effectiveness_score']:.2f}")
    
    print("\n" + "="*60)
    print("✅ Analysis complete! Results saved to database.")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
