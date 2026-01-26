"""
Central configuration management for Anomaly Detection System
"""
import os
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # Application
    APP_NAME: str = "Anomaly Detection System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # API Settings
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_WORKERS: int = Field(default=4, env="API_WORKERS")
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://postgres:password@localhost:5432/anomaly_db",
        env="DATABASE_URL"
    )
    DB_POOL_SIZE: int = Field(default=10, env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default=20, env="DB_MAX_OVERFLOW")
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = DATA_DIR / "models"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    # Model Settings
    MODEL_RETRAIN_DAYS: int = Field(default=7, env="MODEL_RETRAIN_DAYS")
    ANOMALY_THRESHOLD: float = Field(default=0.95, env="ANOMALY_THRESHOLD")
    CONTAMINATION: float = Field(default=0.05, env="CONTAMINATION")
    
    # Alert Settings
    ALERT_EMAIL_ENABLED: bool = Field(default=False, env="ALERT_EMAIL_ENABLED")
    ALERT_SLACK_ENABLED: bool = Field(default=False, env="ALERT_SLACK_ENABLED")
    HIGH_PRIORITY_THRESHOLD: float = Field(default=0.98, env="HIGH_PRIORITY_THRESHOLD")
    MEDIUM_PRIORITY_THRESHOLD: float = Field(default=0.90, env="MEDIUM_PRIORITY_THRESHOLD")
    
    # Monitoring
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    DRIFT_CHECK_INTERVAL: int = Field(default=3600, env="DRIFT_CHECK_INTERVAL")  # seconds
    
    # Feature Engineering
    FEATURE_WINDOW_DAYS: int = Field(default=30, env="FEATURE_WINDOW_DAYS")
    MIN_TRANSACTIONS_FOR_PROFILE: int = Field(default=10, env="MIN_TRANSACTIONS_FOR_PROFILE")
    
    # Performance
    BATCH_SIZE: int = Field(default=1000, env="BATCH_SIZE")
    MAX_CONCURRENT_REQUESTS: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    PREDICTION_TIMEOUT: int = Field(default=5, env="PREDICTION_TIMEOUT")  # seconds
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = "json"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class ModelConfig:
    """Model-specific configurations"""
    
    ISOLATION_FOREST = {
        "n_estimators": 100,
        "max_samples": 256,
        "contamination": 0.05,
        "random_state": 42,
        "n_jobs": -1
    }
    
    LOF = {
        "n_neighbors": 20,
        "contamination": 0.05,
        "novelty": True,
        "n_jobs": -1
    }
    
    XGBOOST = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": 20,  # Handle imbalance
        "random_state": 42
    }
    
    AUTOENCODER = {
        "encoding_dim": 32,
        "hidden_layers": [64, 32, 16],
        "activation": "relu",
        "output_activation": "sigmoid",
        "loss": "mse",
        "optimizer": "adam",
        "epochs": 50,
        "batch_size": 128,
        "validation_split": 0.2
    }
    
    ENSEMBLE = {
        "models": ["isolation_forest", "lof", "autoencoder"],
        "voting": "soft",
        "weights": [0.4, 0.3, 0.3]
    }


class FeatureConfig:
    """Feature engineering configuration"""
    
    TRANSACTION_FEATURES = [
        "amount",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "time_since_last_transaction",
        "transaction_count_24h",
        "amount_zscore",
        "velocity_1h",
        "velocity_24h"
    ]
    
    USER_FEATURES = [
        "user_avg_amount",
        "user_std_amount",
        "user_transaction_count",
        "account_age_days",
        "user_distinct_merchants",
        "user_distinct_locations"
    ]
    
    MERCHANT_FEATURES = [
        "merchant_avg_amount",
        "merchant_transaction_count",
        "merchant_fraud_rate"
    ]
    
    DERIVED_FEATURES = [
        "amount_deviation_from_user_avg",
        "amount_deviation_from_merchant_avg",
        "is_new_merchant_for_user",
        "distance_from_home",
        "unusual_hour_for_user"
    ]


# Singleton instance
settings = Settings()
model_config = ModelConfig()
feature_config = FeatureConfig()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return model_config


def get_feature_config() -> FeatureConfig:
    """Get feature configuration"""
    return feature_config