"""
Ensemble Anomaly Detector
Combines multiple detection algorithms for robust fraud detection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.knn import KNN


class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detection algorithms
    Combines unsupervised and supervised methods for robust detection
    """
    
    def __init__(self, contamination: float = 0.05, voting: str = 'soft'):
        """
        Initialize ensemble detector
        
        Args:
            contamination: Expected proportion of anomalies
            voting: 'hard' for majority vote, 'soft' for weighted average
        """
        self.contamination = contamination
        self.voting = voting
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        self.fitted = False
        
    def _init_models(self):
        """Initialize all detection models"""
        # 1. Isolation Forest - Fast and effective
        self.models['isolation_forest'] = IForest(
            n_estimators=100,
            max_samples=256,
            contamination=self.contamination,
            random_state=42
        )
        self.weights['isolation_forest'] = 0.3
        
        # 2. Local Outlier Factor - Detects local anomalies
        self.models['lof'] = LOF(
            n_neighbors=20,
            contamination=self.contamination
        )
        self.weights['lof'] = 0.25
        
        # 3. K-Nearest Neighbors - Distance based
        self.models['knn'] = KNN(
            n_neighbors=10,
            contamination=self.contamination
        )
        self.weights['knn'] = 0.20
        
        # 4. Statistical - Z-score based
        self.models['statistical'] = StatisticalDetector(
            threshold=3.0
        )
        self.weights['statistical'] = 0.25
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """
        Train all models in the ensemble
        
        Args:
            X: Training features
            y: Optional labels (for semi-supervised models)
        """
        print(f"Training ensemble with {X.shape[0]} samples...")
        
        # Initialize models
        self._init_models()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                if hasattr(model, 'fit'):
                    model.fit(X_scaled, y)
            except Exception as e:
                print(f"Warning: Failed to train {name}: {e}")
                
        self.fitted = True
        print("Ensemble training complete!")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly labels (0=normal, 1=anomaly)
        
        Args:
            X: Features to predict
            
        Returns:
            Binary predictions
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        scores = self.predict_proba(X)
        
        # Threshold at contamination percentile
        threshold = np.percentile(scores, (1 - self.contamination) * 100)
        predictions = (scores >= threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly scores (0-1, higher = more anomalous)
        
        Args:
            X: Features to score
            
        Returns:
            Anomaly scores
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Get scores from each model
        model_scores = {}
        for name, model in self.models.items():
            try:
                if hasattr(model, 'decision_function'):
                    # Get raw scores and normalize to [0, 1]
                    scores = model.decision_function(X_scaled)
                    scores = self._normalize_scores(scores)
                elif hasattr(model, 'predict_proba'):
                    # Already probabilities
                    scores = model.predict_proba(X_scaled)[:, 1]
                else:
                    # Binary predictions, convert to scores
                    scores = model.predict(X_scaled).astype(float)
                
                model_scores[name] = scores
            except Exception as e:
                print(f"Warning: Failed to score with {name}: {e}")
                model_scores[name] = np.zeros(len(X))
        
        # Combine scores based on voting method
        if self.voting == 'hard':
            # Majority vote
            threshold = 0.5
            votes = np.array([scores > threshold for scores in model_scores.values()])
            ensemble_scores = votes.mean(axis=0)
        else:
            # Weighted average
            weighted_scores = []
            for name, scores in model_scores.items():
                weight = self.weights.get(name, 1.0 / len(self.models))
                weighted_scores.append(scores * weight)
            
            ensemble_scores = np.sum(weighted_scores, axis=0)
        
        return ensemble_scores
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Get aggregated feature importance from models that support it
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importances
        """
        importances = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                for i, feat in enumerate(feature_names):
                    importances[feat] = importances.get(feat, 0) + model.feature_importances_[i]
        
        # Normalize
        total = sum(importances.values())
        if total > 0:
            importances = {k: v/total for k, v in importances.items()}
        
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score == 0:
            return np.ones_like(scores) * 0.5
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized
    
    def save(self, path: str):
        """Save ensemble model to disk"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'voting': self.voting,
            'fitted': self.fitted
        }
        
        joblib.dump(model_data, save_path)
        print(f"Model saved to {save_path}")
    
    # @classmethod
    # def load(cls, path: str):
    #     """Load ensemble model from disk"""
    #     import joblib
    #     model_data = joblib.load(path)
        
    #     detector = cls(
    #         contamination=model_data['contamination'],
    #         voting=model_data['voting']
    #     )
        
    #     detector.models = model_data['models']
    #     detector.weights = model_data['weights']
    #     detector.scaler = model_data['scaler']
    #     detector.fitted = model_data['fitted']
        
    #     print(f"Model loaded from {path}")
    #     return detector

    @classmethod
    def load(cls, path: str):
        """Load ensemble model from disk using joblib"""
        import joblib
        # Since we saved the whole object, joblib.load returns the 
        # already-constructed EnsembleAnomalyDetector instance.
        detector = joblib.load(path)
        
        print(f"Model loaded successfully from {path}")
        return detector


class StatisticalDetector:
    """Simple statistical anomaly detector using Z-scores"""
    
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X, y=None):
        """Fit statistical parameters"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-10  # Avoid division by zero
        return self
        
    def decision_function(self, X):
        """Calculate anomaly scores based on Z-scores"""
        z_scores = np.abs((X - self.mean_) / self.std_)
        # Max Z-score across features
        anomaly_scores = np.max(z_scores, axis=1)
        return anomaly_scores
        
    def predict(self, X):
        """Predict anomalies (1) vs normal (0)"""
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)


class SupervisedDetector:
    """
    Supervised fraud detector using XGBoost
    Used when labeled fraud data is available
    """
    
    def __init__(self, **xgb_params):
        """Initialize XGBoost classifier"""
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': 20,
            'random_state': 42
        }
        default_params.update(xgb_params)
        
        self.model = xgb.XGBClassifier(**default_params)
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Train supervised model"""
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(
            X_scaled, y,
            eval_set=[(X_scaled, y)],
            verbose=False
        )
        
        self.fitted = True
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud labels"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importances"""
        importances = dict(zip(feature_names, self.model.feature_importances_))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))