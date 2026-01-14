import pytest
from src.ab_testing import ABTestAnalyzer

def test_ab_analysis_significant():
    analyzer = ABTestAnalyzer()
    
    # Clearly significant difference
    result = analyzer.analyze_conversion(
        conversions_a=100, total_a=1000,  # 10%
        conversions_b=200, total_b=1000   # 20%
    )
    
    assert result.is_significant
    assert result.winner == "Variant"
    assert result.uplift == 1.0  # 100% uplift

def test_ab_analysis_not_significant():
    analyzer = ABTestAnalyzer()
    
    # Likely not significant
    result = analyzer.analyze_conversion(
        conversions_a=100, total_a=1000,
        conversions_b=105, total_b=1000
    )
    
    assert not result.is_significant
    assert result.winner is None

def test_simulation_data():
    analyzer = ABTestAnalyzer()
    df = analyzer.simulate_test_data(n=100, base_rate=0.5, uplift=0.0)
    
    assert len(df) == 200
    assert 'group' in df.columns
    assert 'converted' in df.columns
    assert set(df['group'].unique()) == {'Control', 'Variant'}
