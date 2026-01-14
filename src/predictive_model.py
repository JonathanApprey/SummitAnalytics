import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Dict, Tuple, Any

class ConversionPredictor:
    """
    Predictive model for session conversion probability using Logistic Regression.
    """
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.feature_cols = ['page_views', 'session_duration', 'bounce_rate', 'previous_visits']
        self.metrics = {}

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the model on historical session data.
        
        Args:
            df: DataFrame containing session data with features and 'is_converted' target.
            
        Returns:
            Dictionary containing model performance metrics (accuracy, roc_auc).
        """
        X = df[self.feature_cols]
        y = df['is_converted']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5  # Default/neutral score if undefined

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': auc
        }
        
        return self.metrics

    def predict(self, input_data: Dict[str, float]) -> float:
        """
        Predict conversion probability for a single session.
        
        Args:
            input_data: Dictionary containing feature values.
            
        Returns:
            Probability of conversion (0.0 to 1.0).
        """
        # Create DataFrame for single sample
        df = pd.DataFrame([input_data])
        
        # Ensure all features exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        # Scale features
        X_scaled = self.scaler.transform(df[self.feature_cols])
        
        # Predict probability
        prob = self.model.predict_proba(X_scaled)[0, 1]
        return float(prob)

    def save_model(self, filepath: Path):
        """Save trained model and scaler to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'feature_cols': self.feature_cols
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath: Path):
        """Load trained model and scaler from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.metrics = model_data['metrics']
        self.feature_cols = model_data['feature_cols']
