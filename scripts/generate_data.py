"""
Synthetic Transaction Data Generator
Generates realistic transaction data with fraud patterns
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Tuple

from config.settings import get_settings


class TransactionGenerator:
    """Generate synthetic transaction data with fraud patterns"""
    
    def __init__(self, n_users: int = 1000, n_merchants: int = 500):
        self.n_users = n_users
        self.n_merchants = n_merchants
        self.fraud_rate = 0.05  # 5% fraud rate
        
        # User profiles
        self.user_profiles = self._create_user_profiles()
        
        # Merchant profiles
        self.merchant_profiles = self._create_merchant_profiles()
        
    def _create_user_profiles(self) -> pd.DataFrame:
        """Create user profiles with typical behaviors"""
        profiles = []
        
        for user_id in range(self.n_users):
            profile = {
                'user_id': f'user_{user_id:05d}',
                'avg_amount': np.random.lognormal(mean=3.5, sigma=1.0),  # ~$50-200
                'std_amount': np.random.uniform(10, 50),
                'txn_frequency_per_day': np.random.uniform(0.5, 5.0),
                'preferred_hour': np.random.randint(8, 22),
                'home_lat': np.random.uniform(25, 48),  # US latitude range
                'home_lon': np.random.uniform(-125, -65),  # US longitude range
                'typical_merchants': random.sample(
                    [f'merchant_{i:05d}' for i in range(self.n_merchants)],
                    k=random.randint(5, 20)
                )
            }
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def _create_merchant_profiles(self) -> pd.DataFrame:
        """Create merchant profiles"""
        categories = ['retail', 'dining', 'gas', 'grocery', 'entertainment', 
                     'travel', 'healthcare', 'utilities']
        
        profiles = []
        
        for merchant_id in range(self.n_merchants):
            profile = {
                'merchant_id': f'merchant_{merchant_id:05d}',
                'category': random.choice(categories),
                'avg_amount': np.random.lognormal(mean=3.0, sigma=1.2),
                'location_lat': np.random.uniform(25, 48),
                'location_lon': np.random.uniform(-125, -65),
                'risk_level': random.choices(
                    ['low', 'medium', 'high'],
                    weights=[0.7, 0.25, 0.05]
                )[0]
            }
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def generate_normal_transaction(
        self, 
        user_profile: dict, 
        timestamp: datetime
    ) -> dict:
        """Generate a normal transaction"""
        # Select merchant (prefer typical merchants)
        if random.random() < 0.8 and len(user_profile['typical_merchants']) > 0:
            merchant_id = random.choice(user_profile['typical_merchants'])
        else:
            merchant_id = f'merchant_{random.randint(0, self.n_merchants-1):05d}'
        
        merchant = self.merchant_profiles[
            self.merchant_profiles['merchant_id'] == merchant_id
        ].iloc[0]
        
        # Amount based on user and merchant
        amount = np.random.normal(
            loc=user_profile['avg_amount'],
            scale=user_profile['std_amount']
        )
        amount = max(1.0, amount)  # Minimum $1
        amount = round(amount, 2)
        
        # Location (near user's home)
        location_lat = user_profile['home_lat'] + np.random.normal(0, 0.5)
        location_lon = user_profile['home_lon'] + np.random.normal(0, 0.5)
        
        return {
            'user_id': user_profile['user_id'],
            'merchant_id': merchant_id,
            'amount': amount,
            'timestamp': timestamp,
            'location_lat': round(location_lat, 6),
            'location_lon': round(location_lon, 6),
            'is_fraud': 0
        }
    
    def generate_fraud_transaction(
        self, 
        user_profile: dict, 
        timestamp: datetime
    ) -> dict:
        """Generate a fraudulent transaction"""
        fraud_type = random.choice([
            'high_amount',
            'unusual_merchant',
            'unusual_location',
            'rapid_succession',
            'unusual_time'
        ])
        
        # Base transaction
        txn = self.generate_normal_transaction(user_profile, timestamp)
        
        # Apply fraud pattern
        if fraud_type == 'high_amount':
            # Unusually high amount
            txn['amount'] = user_profile['avg_amount'] * random.uniform(5, 15)
            txn['amount'] = round(txn['amount'], 2)
        
        elif fraud_type == 'unusual_merchant':
            # New merchant, potentially risky
            high_risk_merchants = self.merchant_profiles[
                self.merchant_profiles['risk_level'] == 'high'
            ]
            if len(high_risk_merchants) > 0:
                merchant = high_risk_merchants.sample(1).iloc[0]
                txn['merchant_id'] = merchant['merchant_id']
        
        elif fraud_type == 'unusual_location':
            # Far from home
            txn['location_lat'] = user_profile['home_lat'] + random.uniform(5, 20)
            txn['location_lon'] = user_profile['home_lon'] + random.uniform(5, 20)
            txn['location_lat'] = round(txn['location_lat'], 6)
            txn['location_lon'] = round(txn['location_lon'], 6)
        
        elif fraud_type == 'rapid_succession':
            # This would be multiple transactions - handled at generation level
            pass
        
        elif fraud_type == 'unusual_time':
            # Late night transaction
            hour = random.choice([1, 2, 3, 4, 5])
            txn['timestamp'] = txn['timestamp'].replace(hour=hour)
        
        txn['is_fraud'] = 1
        return txn
    
    def generate_dataset(
        self, 
        n_transactions: int = 100000,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """Generate complete dataset"""
        print(f"Generating {n_transactions:,} transactions...")
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)
        if end_date is None:
            end_date = datetime.now()
        
        transactions = []
        n_fraud = int(n_transactions * self.fraud_rate)
        n_normal = n_transactions - n_fraud
        
        print(f"Normal: {n_normal:,}, Fraud: {n_fraud:,} ({self.fraud_rate:.1%})")
        
        # Generate normal transactions
        for i in range(n_normal):
            if i % 10000 == 0:
                print(f"Generated {i:,} normal transactions...")
            
            # Random user
            user = self.user_profiles.sample(1).iloc[0]
            
            # Random timestamp
            timestamp = start_date + timedelta(
                seconds=random.randint(0, int((end_date - start_date).total_seconds()))
            )
            
            # Adjust hour to user's typical pattern
            hour_offset = np.random.normal(0, 3)
            hour = int(user['preferred_hour'] + hour_offset)
            hour = max(0, min(23, hour))
            timestamp = timestamp.replace(hour=hour, minute=random.randint(0, 59))
            
            txn = self.generate_normal_transaction(user.to_dict(), timestamp)
            transactions.append(txn)
        
        # Generate fraud transactions
        for i in range(n_fraud):
            if i % 1000 == 0:
                print(f"Generated {i:,} fraud transactions...")
            
            user = self.user_profiles.sample(1).iloc[0]
            timestamp = start_date + timedelta(
                seconds=random.randint(0, int((end_date - start_date).total_seconds()))
            )
            
            txn = self.generate_fraud_transaction(user.to_dict(), timestamp)
            transactions.append(txn)
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        
        # Add transaction IDs
        df['transaction_id'] = [f'txn_{i:010d}' for i in range(len(df))]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Reorder columns
        df = df[[
            'transaction_id', 'user_id', 'merchant_id', 'amount',
            'timestamp', 'location_lat', 'location_lon', 'is_fraud'
        ]]
        
        print(f"\n✓ Generated {len(df):,} total transactions")
        print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
        
        return df


def main():
    """Generate and save synthetic data"""
    print("\n" + "="*70)
    print("SYNTHETIC TRANSACTION DATA GENERATOR")
    print("="*70 + "\n")
    
    settings = get_settings()
    
    # Parameters
    n_transactions = 100000
    n_users = 1000
    n_merchants = 500
    
    # Generate data
    generator = TransactionGenerator(
        n_users=n_users,
        n_merchants=n_merchants
    )
    
    df = generator.generate_dataset(
        n_transactions=n_transactions,
        start_date=datetime.now() - timedelta(days=90),
        end_date=datetime.now()
    )
    
    # Statistics
    print("\nDataset Statistics:")
    print(f"Total Transactions: {len(df):,}")
    print(f"Unique Users: {df['user_id'].nunique():,}")
    print(f"Unique Merchants: {df['merchant_id'].nunique():,}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nAmount Statistics:")
    print(df['amount'].describe())
    print(f"\nFraud Distribution:")
    print(df['is_fraud'].value_counts())
    print(f"Fraud Rate: {df['is_fraud'].mean():.2%}")
    
    # Save data
    output_dir = settings.DATA_DIR / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "transactions.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Data saved to: {output_path}")
    
    # Save sample for testing
    sample = df.sample(n=1000, random_state=42)
    sample_path = settings.DATA_DIR / "sample_data" / "sample_transactions.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False)
    
    print(f"✓ Sample data saved to: {sample_path}")
    
    print("\n" + "="*70)
    print("DATA GENERATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Train models: python scripts/train_models.py")
    print("2. Start API: python src/api/main.py")


if __name__ == "__main__":
    main()