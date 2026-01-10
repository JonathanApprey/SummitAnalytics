#!/usr/bin/env python3
"""
Summit Analytics - Executive Dashboard
Demonstrates: Data visualization, KPI tracking, strategic insights communication.

A Streamlit-based interactive dashboard for website traffic analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.analysis_engine import run_full_analysis, load_data_from_db

# Page configuration
st.set_page_config(
    page_title="Summit Analytics | Website Traffic Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.2);
    }
    
    [data-testid="metric-container"] label {
        color: #e94560 !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e94560 !important;
    }
    
    /* Sidebar - Updated for better visibility */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2a4a 0%, #2d3b5e 100%) !important;
    }
    
    /* Sidebar text styling */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #00d4ff !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    [data-testid="stSidebar"] h3 {
        color: #4ade80 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: #e8e8e8 !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }
    
    [data-testid="stSidebar"] strong {
        color: #00d4ff !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Tab styling - Make tab labels clearer */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%);
        border-radius: 12px;
        padding: 8px;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #a0aec0 !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 212, 255, 0.2) !important;
        color: #00d4ff !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4) !important;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    
    /* Cards/containers */
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(15, 52, 96, 0.5);
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Insight boxes - Enhanced for clarity */
    .insight-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%);
        border-left: 5px solid #e94560;
        border: 2px solid rgba(233, 69, 96, 0.3);
        border-left: 5px solid #e94560;
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        color: #ffffff !important;
    }
    
    .insight-box strong {
        color: #00d4ff !important;
        font-size: 1.1rem !important;
    }
    
    .insight-box ul {
        margin: 15px 0;
        padding-left: 20px;
    }
    
    .insight-box li {
        color: #e8e8e8 !important;
        font-size: 1rem !important;
        line-height: 1.8 !important;
        margin-bottom: 8px;
    }
    
    .insight-box li strong {
        color: #4ade80 !important;
        font-weight: 700 !important;
    }
    
    /* Recommendation boxes - Enhanced for clarity */
    .recommendation-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-left: 5px solid #00d4ff;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        color: #ffffff !important;
    }
    
    .recommendation-box strong {
        color: #00d4ff !important;
        font-size: 1.05rem !important;
    }
    
    /* Main Header Section - Summit Analytics */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 50%, #1e3a5f 100%);
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 20px;
        border: 2px solid #00d4ff;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.3);
    }
    
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        margin-bottom: 10px !important;
    }
    
    .main-header .subtitle {
        color: #00d4ff !important;
        font-size: 1.2rem !important;
        font-weight: 500 !important;
    }
    
    .main-header .badge {
        display: inline-block;
        background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
        color: white !important;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 10px;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(74, 222, 128, 0.2);
        border: 1px solid #4ade80;
        color: #4ade80 !important;
        padding: 8px 16px;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .status-badge::before {
        content: "●";
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
</style>
""", unsafe_allow_html=True)


def get_db_path() -> Path:
    """Get database path."""
    return Path(__file__).parent / "data" / "summit.db"


@st.cache_data(ttl=300)
def load_analyzed_data():
    """Load analyzed data from database."""
    db_path = get_db_path()
    
    if not db_path.exists():
        return None, None
    
    conn = sqlite3.connect(db_path)
    
    try:
        df = pd.read_sql_query("SELECT * FROM analyzed_sessions", conn)
        source_df = pd.read_sql_query("SELECT * FROM source_analysis", conn)
    except:
        conn.close()
        return None, None
    
    conn.close()
    return df, source_df


def render_header():
    """Render the dashboard header."""
    st.markdown("""
    <div class='main-header'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <h1>📊 Summit Analytics</h1>
                <p class='subtitle'>Website Traffic Intelligence Dashboard</p>
                <span class='badge'>Entry-Level Data Analyst Portfolio Project</span>
            </div>
            <div style='text-align: right;'>
                <div class='status-badge'>Live Data</div>
                <p style='color: #a0aec0; font-size: 12px; margin-top: 10px;'>Real-time Analysis</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_section(df: pd.DataFrame):
    """Render the KPI metrics section."""
    st.markdown("### 📈 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_sessions = len(df)
        st.metric(
            label="Total Sessions",
            value=f"{total_sessions:,}",
            delta="Live Data"
        )
    
    with col2:
        conversion_rate = df['conversion_rate'].mean() * 100
        st.metric(
            label="Avg. Conversion Rate",
            value=f"{conversion_rate:.1f}%",
            delta=f"{(conversion_rate - 50):.1f}% vs benchmark"
        )
    
    with col3:
        avg_page_views = df['page_views'].mean()
        st.metric(
            label="Avg. Page Views",
            value=f"{avg_page_views:.1f}",
            delta="per session"
        )
    
    with col4:
        bounce_rate = df['bounce_rate'].mean() * 100
        st.metric(
            label="Avg. Bounce Rate",
            value=f"{bounce_rate:.1f}%",
            delta=f"{(30 - bounce_rate):.1f}% vs target" if bounce_rate < 30 else f"+{(bounce_rate - 30):.1f}%"
        )
    
    with col5:
        avg_engagement = df['engagement_score'].mean()
        st.metric(
            label="Engagement Score",
            value=f"{avg_engagement:.2f}",
            delta="composite index"
        )


def render_traffic_source_analysis(df: pd.DataFrame, source_df: pd.DataFrame):
    """Render traffic source analysis section."""
    st.markdown("### 📡 Traffic Source Performance")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Pie chart of traffic distribution
        source_counts = df['traffic_source'].value_counts()
        fig_pie = px.pie(
            values=source_counts.values,
            names=source_counts.index,
            title="Traffic Distribution by Source",
            color_discrete_sequence=px.colors.sequential.RdBu,
            hole=0.4
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title=dict(font=dict(size=18, color='#00d4ff')),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=14, color='white')
            )
        )
        fig_pie.update_traces(
            textfont=dict(size=14, color='white'),
            textinfo='percent+label'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart of conversion rates by source
        source_conv = df.groupby('traffic_source')['conversion_rate'].mean().sort_values(ascending=True)
        fig_bar = px.bar(
            x=source_conv.values * 100,
            y=source_conv.index,
            orientation='h',
            title="Conversion Rate by Source (%)",
            color=source_conv.values,
            color_continuous_scale='RdBu'
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title=dict(font=dict(size=18, color='#00d4ff')),
            showlegend=False,
            xaxis=dict(
                title=dict(text='Conversion Rate (%)', font=dict(size=16, color='#4ade80')),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title=dict(text='', font=dict(size=16, color='#4ade80')),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            coloraxis_colorbar=dict(
                title=dict(text='Rate', font=dict(size=14, color='white')),
                tickfont=dict(size=12, color='white')
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Effectiveness ranking table
    st.markdown("#### 🏆 Source Effectiveness Ranking")
    
    if source_df is not None:
        display_df = source_df[['traffic_source', 'total_sessions', 'avg_conversion_rate', 
                                 'avg_bounce_rate', 'effectiveness_score']].copy()
        display_df.columns = ['Source', 'Sessions', 'Conv. Rate', 'Bounce Rate', 'Effectiveness']
        display_df['Conv. Rate'] = (display_df['Conv. Rate'] * 100).round(1).astype(str) + '%'
        display_df['Bounce Rate'] = (display_df['Bounce Rate'] * 100).round(1).astype(str) + '%'
        display_df['Effectiveness'] = display_df['Effectiveness'].round(2)
        display_df['Sessions'] = display_df['Sessions'].astype(int)
        
        st.dataframe(
            display_df.sort_values('Effectiveness', ascending=False),
            hide_index=True,
            use_container_width=True
        )


def render_segment_analysis(df: pd.DataFrame):
    """Render visitor segmentation analysis."""
    st.markdown("### 👥 Visitor Segmentation Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Scatter plot of segments
        fig_scatter = px.scatter(
            df,
            x='engagement_score',
            y='conversion_rate',
            color='cluster',
            size='page_views',
            title="Visitor Segments (Engagement vs Conversion)",
            color_continuous_scale='RdBu',
            opacity=0.6
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title=dict(font=dict(size=18, color='#00d4ff')),
            xaxis=dict(
                title=dict(text='Engagement Score', font=dict(size=16, color='#4ade80')),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title=dict(text='Conversion Rate', font=dict(size=16, color='#4ade80')),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            coloraxis_colorbar=dict(
                title=dict(text='Cluster', font=dict(size=14, color='white')),
                tickfont=dict(size=12, color='white')
            )
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Segment distribution
        segment_counts = df['cluster'].value_counts().sort_index()
        fig_seg = px.bar(
            x=[f"Segment {i+1}" for i in segment_counts.index],
            y=segment_counts.values,
            title="Sessions by Segment",
            color=segment_counts.values,
            color_continuous_scale='RdBu'
        )
        fig_seg.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title=dict(font=dict(size=18, color='#00d4ff')),
            showlegend=False,
            xaxis=dict(
                title=dict(text='Segment', font=dict(size=16, color='#4ade80')),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title=dict(text='Number of Sessions', font=dict(size=16, color='#4ade80')),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            coloraxis_colorbar=dict(
                title=dict(text='Sessions', font=dict(size=14, color='white')),
                tickfont=dict(size=12, color='white')
            )
        )
        st.plotly_chart(fig_seg, use_container_width=True)
    
    # Segment profiles
    st.markdown("#### 📋 Segment Profiles")
    
    segment_profiles = df.groupby('cluster').agg({
        'session_id': 'count',
        'page_views': 'mean',
        'session_duration': 'mean',
        'bounce_rate': 'mean',
        'conversion_rate': 'mean',
        'engagement_score': 'mean'
    }).round(2)
    
    segment_profiles.columns = ['Sessions', 'Avg Page Views', 'Avg Duration', 
                                 'Bounce Rate', 'Conv. Rate', 'Engagement']
    segment_profiles.index = [f"Segment {i+1}" for i in segment_profiles.index]
    
    st.dataframe(segment_profiles, use_container_width=True)


def render_efm_analysis(df: pd.DataFrame):
    """Render EFM (Engagement, Frequency, Monetary) analysis."""
    st.markdown("### 🎯 EFM Score Analysis")
    
    st.markdown("""
    <div class='insight-box'>
        <strong>EFM Analysis</strong> is our adaptation of RFM (Recency, Frequency, Monetary) for website traffic:
        <ul>
            <li><strong>E (Engagement):</strong> Based on page views, session duration, and time on page</li>
            <li><strong>F (Frequency):</strong> Based on previous visits to the site</li>
            <li><strong>M (Monetary):</strong> Based on conversion rate (value delivered)</li>
        </ul>
        Scores range from 1-5, with 5 being the highest.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_e = px.histogram(
            df, x='E_Score', nbins=5,
            title="Engagement Score Distribution",
            color_discrete_sequence=['#e94560']
        )
        fig_e.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title=dict(font=dict(size=18, color='#00d4ff')),
            xaxis=dict(
                title=dict(text='E Score', font=dict(size=16, color='#4ade80')),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title=dict(text='Count', font=dict(size=16, color='#4ade80')),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            )
        )
        st.plotly_chart(fig_e, use_container_width=True)
    
    with col2:
        fig_f = px.histogram(
            df, x='F_Score', nbins=5,
            title="Frequency Score Distribution",
            color_discrete_sequence=['#00d4ff']
        )
        fig_f.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title=dict(font=dict(size=18, color='#00d4ff')),
            xaxis=dict(
                title=dict(text='F Score', font=dict(size=16, color='#4ade80')),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title=dict(text='Count', font=dict(size=16, color='#4ade80')),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            )
        )
        st.plotly_chart(fig_f, use_container_width=True)
    
    with col3:
        fig_m = px.histogram(
            df, x='M_Score', nbins=5,
            title="Monetary Score Distribution",
            color_discrete_sequence=['#4ade80']
        )
        fig_m.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title=dict(font=dict(size=18, color='#00d4ff')),
            xaxis=dict(
                title=dict(text='M Score', font=dict(size=16, color='#4ade80')),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title=dict(text='Count', font=dict(size=16, color='#4ade80')),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            )
        )
        st.plotly_chart(fig_m, use_container_width=True)
    
    # EFM heatmap
    efm_pivot = df.groupby(['E_Score', 'M_Score']).size().unstack(fill_value=0)
    fig_heatmap = px.imshow(
        efm_pivot,
        title="EFM Score Heatmap (Engagement vs Monetary)",
        color_continuous_scale='RdBu',
        labels=dict(x="Monetary Score", y="Engagement Score", color="Sessions")
    )
    fig_heatmap.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=14),
        title=dict(font=dict(size=20, color='#00d4ff')),
        xaxis=dict(
            title=dict(text='Monetary Score', font=dict(size=16, color='#4ade80')),
            tickfont=dict(size=14, color='white')
        ),
        yaxis=dict(
            title=dict(text='Engagement Score', font=dict(size=16, color='#4ade80')),
            tickfont=dict(size=14, color='white')
        ),
        coloraxis_colorbar=dict(
            title=dict(text='Sessions', font=dict(size=14, color='white')),
            tickfont=dict(size=12, color='white')
        )
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)


def render_insights(df: pd.DataFrame, source_df: pd.DataFrame):
    """Render strategic insights and recommendations."""
    st.markdown("### 💡 Strategic Insights & Recommendations")
    
    # Calculate insights
    best_source = source_df.loc[source_df['effectiveness_score'].idxmax(), 'traffic_source']
    worst_source = source_df.loc[source_df['effectiveness_score'].idxmin(), 'traffic_source']
    
    high_value_pct = (df[df['EFM_Score'] >= 12].shape[0] / len(df)) * 100
    at_risk_pct = (df[df['EFM_Score'] <= 6].shape[0] / len(df)) * 100
    
    avg_conv = df['conversion_rate'].mean()
    best_segment_conv = df.groupby('cluster')['conversion_rate'].mean().max()
    
    # Display insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Key Findings")
        
        st.markdown(f"""
        <div class='insight-box'>
            <strong>1. Top Performing Channel</strong><br>
            <span style='color: #4ade80; font-size: 18px;'>{best_source}</span> is your most effective traffic source.
            Focus marketing budget here for best ROI.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='insight-box'>
            <strong>2. High-Value Visitors</strong><br>
            <span style='color: #4ade80; font-size: 18px;'>{high_value_pct:.1f}%</span> of visitors are high-value 
            (EFM Score ≥ 12). These are your champions—prioritize retention.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='insight-box'>
            <strong>3. At-Risk Segment</strong><br>
            <span style='color: #e94560; font-size: 18px;'>{at_risk_pct:.1f}%</span> of visitors are at-risk 
            (EFM Score ≤ 6). Implement re-engagement campaigns immediately.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🚀 Recommended Actions")
        
        st.markdown("""
        <div class='recommendation-box'>
            <strong>📈 Growth Strategy</strong><br>
            1. Double down on <strong>""" + best_source + """</strong> channel<br>
            2. A/B test landing pages for <strong>""" + worst_source + """</strong> traffic<br>
            3. Implement exit-intent popups for high-bounce segments
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='recommendation-box'>
            <strong>👥 Segmentation Strategy</strong><br>
            1. Create personalized nurture flows for each segment<br>
            2. Offer loyalty rewards to Frequent & Loyal visitors<br>
            3. Fast-track high-engagement visitors with targeted CTAs
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='recommendation-box'>
            <strong>📊 Measurement Plan</strong><br>
            1. Track EFM score trends weekly<br>
            2. Monitor segment migration (are visitors improving?)<br>
            3. Set up alerts for bounce rate spikes by source
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with filters and info."""
    with st.sidebar:
        st.markdown("## 🎛️ Dashboard Controls")
        
        st.markdown("---")
        
        st.markdown("### 📋 About This Project")
        st.markdown("""
        **Summit Analytics** is a portfolio project demonstrating:
        
        ✅ **Data Management** - SQL schema design, ETL  
        ✅ **Analysis** - Statistical techniques, clustering  
        ✅ **Visualization** - Interactive dashboards  
        ✅ **Strategy** - Actionable business insights  
        """)
        
        st.markdown("---")
        
        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
        - **Python** - Core analysis
        - **SQL/SQLite** - Data storage
        - **Pandas/NumPy** - Data manipulation
        - **Scikit-learn** - ML/Clustering
        - **Streamlit** - Dashboard
        - **Plotly** - Visualizations
        """)
        
        st.markdown("---")
        
        st.markdown("### 📧 Contact")
        st.markdown("Built for Entry-Level Data Analyst role @ Camping World")


def main():
    """Main dashboard entry point."""
    # Render sidebar
    render_sidebar()
    
    # Render header
    render_header()
    
    st.markdown("---")
    
    # Load data
    df, source_df = load_analyzed_data()
    
    if df is None:
        st.error("⚠️ No analyzed data found. Please run the analysis pipeline first:")
        st.code("""
# Step 1: Load data into database
python scripts/load_db.py

# Step 2: Run analysis engine
python src/analysis_engine.py

# Step 3: Launch dashboard
streamlit run app.py
        """)
        return
    
    # Render KPI section
    render_kpi_section(df)
    
    st.markdown("---")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📡 Traffic Sources", 
        "👥 Visitor Segments", 
        "🎯 EFM Analysis",
        "💡 Strategic Insights"
    ])
    
    with tab1:
        render_traffic_source_analysis(df, source_df)
    
    with tab2:
        render_segment_analysis(df)
    
    with tab3:
        render_efm_analysis(df)
    
    with tab4:
        render_insights(df, source_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>📊 Summit Analytics | Website Traffic Intelligence Dashboard</p>
        <p style='font-size: 12px;'>Portfolio Project for Entry-Level Data Analyst Position</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
