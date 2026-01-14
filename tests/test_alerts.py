import pytest
from src.alerts import KPIAlertSystem

def test_alerts_triggered():
    system = KPIAlertSystem()
    metrics = {
        'conversion_rate': 0.25,  # Trigger critical low
        'bounce_rate': 0.40,      # Trigger warning high
        'avg_session_duration': 1.5 # Trigger warning low
    }
    
    alerts = system.check_alerts(metrics)
    
    assert len(alerts) >= 3
    
    # Check specific alerts
    critical_conv = next(a for a in alerts if a.metric == 'conversion_rate' and a.severity == 'critical')
    assert critical_conv.value == 0.25
    
    warning_bounce = next(a for a in alerts if a.metric == 'bounce_rate' and a.severity == 'warning')
    assert warning_bounce.value == 0.40

def test_no_alerts():
    system = KPIAlertSystem()
    metrics = {
        'conversion_rate': 0.60,
        'bounce_rate': 0.20,
        'avg_session_duration': 3.0
    }
    
    alerts = system.check_alerts(metrics)
    assert len(alerts) == 0
