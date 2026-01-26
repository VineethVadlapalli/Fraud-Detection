"""
Model Training Script
Train and evaluate anomaly detection models
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config.settings import get_settings
from src.data.feature_engineer import FeatureEngineer
from src.models.ensemble import EnsembleAnomalyDetector, SupervisedDetector


def load_data(data_path: str) -> pd.DataFrame:
    """Load transaction data"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} transactions")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Split data into train and test sets"""
    print("Splitting data...")
    
    # Stratified split to maintain fraud ratio
    if 'is_fraud' in df.columns:
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['is_fraud'],
            random_state=42
        )
    else:
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size,
            random_state=42
        )
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Apply feature engineering"""
    print("Engineering features...")
    
    engineer = FeatureEngineer()
    
    # Fit on training data
    train_features = engineer.engineer_features(train_df, fit=True)
    
    # Transform test data
    test_features = engineer.engineer_features(test_df, fit=False)
    
    # Get feature columns
    feature_cols = engineer.get_feature_names(train_features)
    
    print(f"Generated {len(feature_cols)} features")
    
    return train_features, test_features, feature_cols, engineer


def train_unsupervised_model(X_train, contamination: float = 0.05):
    """Train unsupervised ensemble detector"""
    print("\n" + "="*70)
    print("Training Unsupervised Ensemble Detector")
    print("="*70)
    
    detector = EnsembleAnomalyDetector(
        contamination=contamination,
        voting='soft'
    )
    
    detector.fit(X_train)
    
    return detector


def train_supervised_model(X_train, y_train):
    """Train supervised XGBoost detector"""
    print("\n" + "="*70)
    print("Training Supervised XGBoost Detector")
    print("="*70)
    
    detector = SupervisedDetector(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        scale_pos_weight=int(sum(y_train==0) / sum(y_train==1))
    )
    
    detector.fit(X_train, y_train)
    
    return detector


def evaluate_model(detector, X_test, y_test, model_name: str):
    """Evaluate model performance"""
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name}")
    print(f"{'='*70}")
    
    # Get predictions
    y_pred_proba = detector.predict_proba(X_test)
    
    if hasattr(detector, 'contamination'):
        threshold = np.percentile(y_pred_proba, (1 - detector.contamination) * 100)
    else:
        threshold = 0.5
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC Score: {pr_auc:.4f}")
    
    # Business Metrics
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\nBusiness Metrics:")
    print(f"Detection Rate (Recall): {detection_rate:.2%}")
    print(f"Precision: {precision_score:.2%}")
    print(f"False Positive Rate: {false_positive_rate:.2%}")
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'detection_rate': detection_rate,
        'precision': precision_score,
        'fpr': false_positive_rate,
        'predictions': y_pred_proba
    }


def plot_results(results: dict, y_test, save_dir: Path):
    """Create visualization plots"""
    print("\nGenerating plots...")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=(15, 5))
    
    for i, (name, result) in enumerate(results.items(), 1):
        plt.subplot(1, 3, i)
        
        fpr, tpr, _ = roc_curve(y_test, result['predictions'])
        plt.plot(fpr, tpr, label=f'ROC (AUC = {result["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} - ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved ROC curves to {save_dir / 'roc_curves.png'}")
    
    # Metrics Comparison
    plt.figure(figsize=(10, 6))
    metrics_df = pd.DataFrame({
        name: {
            'ROC-AUC': result['roc_auc'],
            'PR-AUC': result['pr_auc'],
            'Detection Rate': result['detection_rate'],
            'Precision': result['precision']
        }
        for name, result in results.items()
    }).T
    
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.legend(loc='lower right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison to {save_dir / 'metrics_comparison.png'}")


def save_models(models: dict, save_dir: Path):
    """Save trained models"""
    print("\nSaving models...")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        save_path = save_dir / f"{name}.joblib"
        model.save(str(save_path))
        print(f"Saved {name} to {save_path}")


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("ANOMALY DETECTION MODEL TRAINING PIPELINE")
    print("="*70 + "\n")
    
    settings = get_settings()
    
    # 1. Load Data
    data_path = settings.DATA_DIR / "raw" / "transactions.csv"
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please generate data first using: python scripts/generate_data.py")
        return
    
    df = load_data(data_path)
    
    # 2. Split Data
    train_df, test_df = split_data(df, test_size=0.2)
    
    # 3. Engineer Features
    train_features, test_features, feature_cols, engineer = engineer_features(
        train_df, test_df
    )
    
    X_train = train_features[feature_cols]
    X_test = test_features[feature_cols]
    
    has_labels = 'is_fraud' in train_features.columns
    
    if has_labels:
        y_train = train_features['is_fraud'].values
        y_test = test_features['is_fraud'].values
        fraud_rate = y_train.mean()
        print(f"\nFraud rate in training data: {fraud_rate:.2%}")
    
    # 4. Train Models
    models = {}
    results = {}
    
    # Unsupervised Model
    unsupervised_detector = train_unsupervised_model(
        X_train, 
        contamination=fraud_rate if has_labels else 0.05
    )
    models['ensemble_detector'] = unsupervised_detector
    
    if has_labels:
        # Supervised Model
        supervised_detector = train_supervised_model(X_train, y_train)
        models['supervised_detector'] = supervised_detector
        
        # 5. Evaluate Models
        results['Unsupervised Ensemble'] = evaluate_model(
            unsupervised_detector, X_test, y_test, "Unsupervised Ensemble"
        )
        
        results['Supervised XGBoost'] = evaluate_model(
            supervised_detector, X_test, y_test, "Supervised XGBoost"
        )
        
        # 6. Visualize Results
        plot_results(results, y_test, settings.BASE_DIR / "results")
    else:
        print("\nNo labels found - skipping supervised training and evaluation")
    
    # 7. Save Models
    save_models(models, settings.MODEL_DIR)
    
    # Save feature engineer
    import joblib
    engineer_path = settings.MODEL_DIR / "feature_engineer.joblib"
    joblib.dump(engineer, engineer_path)
    print(f"\nSaved feature engineer to {engineer_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModels saved to: {settings.MODEL_DIR}")
    print("You can now start the API server with: python src/api/main.py")


if __name__ == "__main__":
    main()