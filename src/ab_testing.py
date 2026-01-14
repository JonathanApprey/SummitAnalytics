import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class TestResult:
    variant_a_name: str
    variant_b_name: str
    metric_name: str
    stat_score: float
    p_value: float
    is_significant: bool
    winner: Optional[str]
    uplift: float
    confidence_interval: Tuple[float, float]

class ABTestAnalyzer:
    """
    Analyzer for A/B test results using statistical methods.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def analyze_conversion(self, 
                          conversions_a: int, total_a: int,
                          conversions_b: int, total_b: int,
                          name_a: str = "Control", name_b: str = "Variant") -> TestResult:
        """
        Analyze binary conversion metrics using Chi-Square or Z-test.
        Here we use simple proportion Z-test approximation.
        """
        
        # Calculate conversion rates
        rate_a = conversions_a / total_a if total_a > 0 else 0
        rate_b = conversions_b / total_b if total_b > 0 else 0
        
        # Uplift
        uplift = (rate_b - rate_a) / rate_a if rate_a > 0 else 0
        
        # Pooled probability
        p_pool = (conversions_a + conversions_b) / (total_a + total_b)
        se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/total_a + 1/total_b))
        
        # Z-score
        z_score = (rate_b - rate_a) / se_pool if se_pool > 0 else 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        is_significant = p_value < self.alpha
        
        winner = None
        if is_significant:
            winner = name_b if rate_b > rate_a else name_a
            
        # Confidence Interval for difference
        margin_error = stats.norm.ppf(1 - self.alpha/2) * se_pool
        diff = rate_b - rate_a
        ci = (diff - margin_error, diff + margin_error)

        return TestResult(
            variant_a_name=name_a,
            variant_b_name=name_b,
            metric_name="Conversion Rate",
            stat_score=z_score,
            p_value=p_value,
            is_significant=is_significant,
            winner=winner,
            uplift=uplift,
            confidence_interval=ci
        )

    def simulate_test_data(self, n: int = 1000, 
                          base_rate: float = 0.10, 
                          uplift: float = 0.0) -> pd.DataFrame:
        """
        Generate mock data for demonstration.
        """
        # Control group
        control_conversions = np.random.binomial(1, base_rate, n)
        control_df = pd.DataFrame({'group': 'Control', 'converted': control_conversions})
        
        # Variant group with uplift
        variant_rate = base_rate * (1 + uplift)
        variant_conversions = np.random.binomial(1, variant_rate, n)
        variant_df = pd.DataFrame({'group': 'Variant', 'converted': variant_conversions})
        
        return pd.concat([control_df, variant_df], ignore_index=True)
