import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


class AnomalyScorer:
    """
    Anomaly scoring and risk assessment system
    """
    
    def __init__(
        self, 
        high_threshold: float = 0.98,
        medium_threshold: float = 0.90,
        low_threshold: float = 0.75
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold
        
    def calculate_risk_level(
        self, 
        anomaly_score: float
    ) -> Tuple[str, float]:
        """
        Calculate risk level and confidence
        
        Args:
            anomaly_score: Anomaly score from detector (0-1)
            
        Returns:
            Tuple of (risk_level, confidence)
        """
        if anomaly_score >= self.high_threshold:
            risk_level = "CRITICAL"
            confidence = min(1.0, (anomaly_score - self.high_threshold) / 
                           (1.0 - self.high_threshold))
        elif anomaly_score >= self.medium_threshold:
            risk_level = "HIGH"
            confidence = (anomaly_score - self.medium_threshold) / \
                        (self.high_threshold - self.medium_threshold)
        elif anomaly_score >= self.low_threshold:
            risk_level = "MEDIUM"
            confidence = (anomaly_score - self.low_threshold) / \
                        (self.medium_threshold - self.low_threshold)
        else:
            risk_level = "LOW"
            confidence = anomaly_score / self.low_threshold
        
        return risk_level, float(confidence)
    
    def identify_contributing_factors(
        self, 
        transaction_features: pd.Series,
        feature_names: List[str],
        top_n: int = 5
    ) -> List[str]:
        """
        Identify features contributing to anomaly
        
        Args:
            transaction_features: Feature values for transaction
            feature_names: List of feature names
            top_n: Number of top contributors to return
            
        Returns:
            List of contributing factor descriptions
        """
        contributors = []
        
        # Check various risk indicators
        if 'is_unusual_amount' in transaction_features and \
           transaction_features['is_unusual_amount'] == 1:
            contributors.append("Unusual transaction amount")
        
        if 'is_rapid_succession' in transaction_features and \
           transaction_features['is_rapid_succession'] == 1:
            contributors.append("Rapid succession of transactions")
        
        if 'is_high_velocity' in transaction_features and \
           transaction_features['is_high_velocity'] == 1:
            contributors.append("High transaction velocity")
        
        if 'is_far_from_home' in transaction_features and \
           transaction_features['is_far_from_home'] == 1:
            contributors.append("Transaction far from usual location")
        
        if 'is_first_txn_with_merchant' in transaction_features and \
           transaction_features['is_first_txn_with_merchant'] == 1:
            contributors.append("First transaction with this merchant")
        
        if 'is_night' in transaction_features and \
           transaction_features['is_night'] == 1:
            contributors.append("Transaction during unusual hours")
        
        # Add high deviation scores
        if 'amount_dev_from_user_mean' in transaction_features:
            dev = abs(transaction_features['amount_dev_from_user_mean'])
            if dev > 3:
                contributors.append(f"Amount deviates {dev:.1f}σ from user average")
        
        return contributors[:top_n] if contributors else ["No specific factors identified"]
