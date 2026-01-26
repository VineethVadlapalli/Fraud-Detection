"""
Feature Engineering Pipeline for Fraud Detection
Transforms raw transaction data into ML-ready features
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Comprehensive feature engineering for transaction anomaly detection
    """
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.scaler = RobustScaler()
        self.fitted = False
        
    def engineer_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        
        Args:
            df: DataFrame with columns [transaction_id, user_id, merchant_id, 
                amount, timestamp, location_lat, location_lon]
            fit: Whether to fit scalers (True for training, False for inference)
        
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 1. Temporal Features
        df = self._add_temporal_features(df)
        
        # 2. Transaction Amount Features
        df = self._add_amount_features(df)
        
        # 3. User Behavioral Features
        df = self._add_user_features(df)
        
        # 4. Merchant Features
        df = self._add_merchant_features(df)
        
        # 5. Velocity Features
        df = self._add_velocity_features(df)
        
        # 6. Geographic Features
        df = self._add_geographic_features(df)
        
        # 7. Deviation Features
        df = self._add_deviation_features(df)
        
        # 8. Risk Indicator Features
        df = self._add_risk_indicators(df)
        
        # 9. Scale numerical features
        df = self._scale_features(df, fit=fit)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal patterns"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Cyclical encoding for hour and day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transaction amount based features"""
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_sqrt'] = np.sqrt(df['amount'])
        
        # Round amount bins
        df['amount_bin'] = pd.cut(df['amount'], 
                                   bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                                   labels=['tiny', 'small', 'medium', 'large', 'very_large', 'huge'])
        df['is_round_amount'] = (df['amount'] % 10 == 0).astype(int)
        df['is_very_round_amount'] = (df['amount'] % 100 == 0).astype(int)
        
        return df
    
    def _add_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """User behavioral features"""
        # User transaction statistics
        user_stats = df.groupby('user_id')['amount'].agg([
            ('user_mean_amount', 'mean'),
            ('user_std_amount', 'std'),
            ('user_median_amount', 'median'),
            ('user_min_amount', 'min'),
            ('user_max_amount', 'max'),
            ('user_transaction_count', 'count')
        ]).reset_index()
        
        df = df.merge(user_stats, on='user_id', how='left')
        
        # Handle users with single transaction
        df['user_std_amount'] = df['user_std_amount'].fillna(0)
        
        # User diversity metrics
        user_merchants = df.groupby('user_id')['merchant_id'].nunique().reset_index()
        user_merchants.columns = ['user_id', 'user_distinct_merchants']
        df = df.merge(user_merchants, on='user_id', how='left')
        
        # User temporal patterns
        user_hours = df.groupby('user_id')['hour'].agg([
            ('user_most_common_hour', lambda x: x.mode()[0] if len(x.mode()) > 0 else 12)
        ]).reset_index()
        df = df.merge(user_hours, on='user_id', how='left')
        
        return df
    
    def _add_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merchant-based features"""
        merchant_stats = df.groupby('merchant_id')['amount'].agg([
            ('merchant_mean_amount', 'mean'),
            ('merchant_std_amount', 'std'),
            ('merchant_transaction_count', 'count')
        ]).reset_index()
        
        df = df.merge(merchant_stats, on='merchant_id', how='left')
        df['merchant_std_amount'] = df['merchant_std_amount'].fillna(0)
        
        # Merchant popularity
        total_transactions = len(df)
        df['merchant_popularity'] = df['merchant_transaction_count'] / total_transactions
        
        return df
    
    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transaction velocity features"""
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Time since last transaction (in seconds)
        df['time_since_last_txn'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        df['time_since_last_txn'] = df['time_since_last_txn'].fillna(86400)  # 24 hours default
        
        # Transactions in last 1 hour, 6 hours, 24 hours
        for hours in [1, 6, 24]:
            df[f'txn_count_{hours}h'] = df.groupby('user_id')['timestamp'].transform(
                lambda x: x.rolling(f'{hours}H', on=df.loc[x.index, 'timestamp']).count()
            )
        
        # Amount velocity (total spent in time windows)
        for hours in [1, 6, 24]:
            df[f'amount_sum_{hours}h'] = df.groupby('user_id')['amount'].transform(
                lambda x: x.rolling(window=hours).sum()
            )
        
        # Fill NaN values
        velocity_cols = [c for c in df.columns if 'txn_count' in c or 'amount_sum' in c]
        df[velocity_cols] = df[velocity_cols].fillna(0)
        
        return df
    
    def _add_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Geographic and location-based features"""
        if 'location_lat' not in df.columns or 'location_lon' not in df.columns:
            return df
        
        # User's home location (most frequent location)
        user_home = df.groupby('user_id')[['location_lat', 'location_lon']].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean()
        ).reset_index()
        user_home.columns = ['user_id', 'home_lat', 'home_lon']
        
        df = df.merge(user_home, on='user_id', how='left')
        
        # Distance from home (Haversine formula)
        df['distance_from_home'] = self._haversine_distance(
            df['location_lat'], df['location_lon'],
            df['home_lat'], df['home_lon']
        )
        
        # Location change velocity
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        df['prev_lat'] = df.groupby('user_id')['location_lat'].shift(1)
        df['prev_lon'] = df.groupby('user_id')['location_lon'].shift(1)
        
        df['location_change'] = self._haversine_distance(
            df['location_lat'], df['location_lon'],
            df['prev_lat'], df['prev_lon']
        )
        df['location_change'] = df['location_change'].fillna(0)
        
        # Distinct locations for user
        user_locations = df.groupby('user_id').apply(
            lambda x: len(set(zip(x['location_lat'], x['location_lon'])))
        ).reset_index()
        user_locations.columns = ['user_id', 'user_distinct_locations']
        df = df.merge(user_locations, on='user_id', how='left')
        
        return df
    
    def _add_deviation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deviation from normal behavior"""
        # Amount deviation from user average
        df['amount_dev_from_user_mean'] = (df['amount'] - df['user_mean_amount']) / (df['user_std_amount'] + 1e-5)
        df['amount_dev_from_merchant_mean'] = (df['amount'] - df['merchant_mean_amount']) / (df['merchant_std_amount'] + 1e-5)
        
        # Z-score for amount
        df['amount_zscore'] = (df['amount'] - df['user_mean_amount']) / (df['user_std_amount'] + 1e-5)
        
        # Ratio features
        df['amount_to_user_max_ratio'] = df['amount'] / (df['user_max_amount'] + 1)
        df['amount_to_user_median_ratio'] = df['amount'] / (df['user_median_amount'] + 1)
        
        return df
    
    def _add_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """High-level risk indicators"""
        # First transaction with merchant
        df['is_first_txn_with_merchant'] = (~df.duplicated(subset=['user_id', 'merchant_id'])).astype(int)
        
        # Unusual amount (3x std from mean)
        df['is_unusual_amount'] = (abs(df['amount_zscore']) > 3).astype(int)
        
        # Rapid succession (< 1 minute since last)
        df['is_rapid_succession'] = (df['time_since_last_txn'] < 60).astype(int)
        
        # High velocity
        df['is_high_velocity'] = (df['txn_count_1h'] > 5).astype(int)
        
        # Far from home (> 100 km)
        if 'distance_from_home' in df.columns:
            df['is_far_from_home'] = (df['distance_from_home'] > 100).astype(int)
        
        # Large location change
        if 'location_change' in df.columns:
            df['is_large_location_change'] = (df['location_change'] > 500).astype(int)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale numerical features"""
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude IDs and target if present
        exclude = ['transaction_id', 'user_id', 'merchant_id', 'is_fraud']
        numerical_features = [f for f in numerical_features if f not in exclude]
        
        if fit:
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
            self.fitted = True
        elif self.fitted:
            df[numerical_features] = self.scaler.transform(df[numerical_features])
        
        return df
    
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two points in kilometers"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature names for modeling"""
        exclude = ['transaction_id', 'user_id', 'merchant_id', 'timestamp', 
                   'is_fraud', 'amount_bin', 'prev_lat', 'prev_lon', 
                   'home_lat', 'home_lon']
        
        features = [col for col in df.columns if col not in exclude]
        return features