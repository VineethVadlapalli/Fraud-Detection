import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_transactions():
    """Generate sample transaction data for testing"""
    n = 100
    
    data = {
        'transaction_id': [f'txn_{i:05d}' for i in range(n)],
        'user_id': [f'user_{i%10:03d}' for i in range(n)],
        'merchant_id': [f'merchant_{i%20:03d}' for i in range(n)],
        'amount': np.random.lognormal(3.5, 1.0, n),
        'timestamp': [
            datetime.now() - timedelta(hours=i) for i in range(n)
        ],
        'location_lat': np.random.uniform(25, 48, n),
        'location_lon': np.random.uniform(-125, -65, n),
        'is_fraud': np.random.choice([0, 1], n, p=[0.95, 0.05])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def trained_detector():
    """Fixture for trained detector"""
    from src.models.ensemble import EnsembleAnomalyDetector
    
    detector = EnsembleAnomalyDetector(contamination=0.05)
    # Fit with dummy data
    X = np.random.randn(1000, 20)
    detector.fit(pd.DataFrame(X))
    
    return detector