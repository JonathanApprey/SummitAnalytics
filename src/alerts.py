import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    operator: str
    severity: str  # 'warning' or 'critical'
    message: str

class KPIAlertSystem:
    """
    System for evaluating KPI metrics against defined thresholds.
    """
    
    def __init__(self):
        # default thresholds
        # Format: metric_name: {'operator': 'lt'/'gt', 'value': X, 'level': 'warning'/'critical'}
        self.thresholds = {
            'conversion_rate': [
                {'operator': 'lt', 'value': 0.50, 'level': 'warning', 'msg': 'Conversion rate below 50% target'},
                {'operator': 'lt', 'value': 0.30, 'level': 'critical', 'msg': 'Conversion rate critically low (<30%)'}
            ],
            'bounce_rate': [
                {'operator': 'gt', 'value': 0.30, 'level': 'warning', 'msg': 'Bounce rate above 30% target'},
                {'operator': 'gt', 'value': 0.50, 'level': 'critical', 'msg': 'Bounce rate critically high (>50%)'}
            ],
            'avg_session_duration': [
                {'operator': 'lt', 'value': 2.0, 'level': 'warning', 'msg': 'Avg session duration below 2 mins'}
            ]
        }

    def check_alerts(self, metrics: Dict[str, float]) -> List[Alert]:
        """
        Evaluate current metrics against thresholds and return checking results.
        
        Args:
            metrics: Dictionary of metric names and their current values
            
        Returns:
            List of Alert objects for any violated thresholds
        """
        triggered_alerts = []
        
        for metric, value in metrics.items():
            if metric in self.thresholds:
                for rule in self.thresholds[metric]:
                    is_violation = False
                    if rule['operator'] == 'lt':
                        is_violation = value < rule['value']
                    elif rule['operator'] == 'gt':
                        is_violation = value > rule['value']
                        
                    if is_violation:
                        triggered_alerts.append(Alert(
                            metric=metric,
                            value=value,
                            threshold=rule['value'],
                            operator=rule['operator'],
                            severity=rule['level'],
                            message=rule['msg']
                        ))
                        
        return triggered_alerts
