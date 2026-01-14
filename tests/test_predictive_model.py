import pytest
import pandas as pd
import numpy as np
from src.predictive_model import ConversionPredictor
from pathlib import Path

def test_model_training():
    # Create mock data
    df = pd.DataFrame({
        'page_views': np.random.randint(1, 20, 100),
        'session_duration': np.random.uniform(10, 600, 100),
        'bounce_rate': np.random.uniform(0, 1, 100),
        'previous_visits': np.random.randint(0, 10, 100),
        'is_converted': np.random.randint(0, 2, 100)
    })
    
    predictor = ConversionPredictor()
    metrics = predictor.train(df)
    
    assert 'accuracy' in metrics
    assert 'roc_auc' in metrics
    assert 0 <= metrics['accuracy'] <= 1.0

def test_model_prediction():
    # Create mock data for training first (needed for scaler fitting)
    df = pd.DataFrame({
        'page_views': np.random.randint(1, 20, 20),
        'session_duration': np.random.uniform(10, 600, 20),
        'bounce_rate': np.random.uniform(0, 1, 20),
        'previous_visits': np.random.randint(0, 10, 20),
        'is_converted': np.random.randint(0, 2, 20)
    })
    
    # Ensure optimal diversity
    df.loc[0:9, 'is_converted'] = 0
    df.loc[10:, 'is_converted'] = 1
    
    predictor = ConversionPredictor()
    predictor.train(df)
    
    # Test single prediction
    input_data = {
        'page_views': 8,
        'session_duration': 250,
        'bounce_rate': 0.3,
        'previous_visits': 3
    }
    
    prob = predictor.predict(input_data)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

def test_model_persistence(tmp_path):
    # Train
    df = pd.DataFrame({
        'page_views': np.random.randint(1, 20, 20),
        'session_duration': np.random.uniform(10, 600, 20),
        'bounce_rate': np.random.uniform(0, 1, 20),
        'previous_visits': np.random.randint(0, 10, 20),
        'is_converted': np.random.randint(0, 2, 20)
    })
    # Ensure diversity
    df.loc[0:9, 'is_converted'] = 0
    df.loc[10:, 'is_converted'] = 1
    predictor = ConversionPredictor()
    predictor.train(df)
    
    # Save
    model_path = tmp_path / "test_model.joblib"
    predictor.save_model(model_path)
    
    # Load into new instance
    new_predictor = ConversionPredictor()
    new_predictor.load_model(model_path)
    
    # Verify input data produces same output
    input_data = {'page_views': 8, 'session_duration': 250, 'bounce_rate': 0.3, 'previous_visits': 3}
    assert predictor.predict(input_data) == new_predictor.predict(input_data)
