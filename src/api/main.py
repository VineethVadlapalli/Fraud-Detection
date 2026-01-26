"""
FastAPI Application for Anomaly Detection System
Provides REST API for real-time fraud detection
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import uvicorn

import sys
import os
sys.path.append(os.getcwd())

from config.settings import get_settings
from src.models.ensemble import EnsembleAnomalyDetector
from src.data.feature_engineer import FeatureEngineer
from src.scoring.anomaly_scorer import AnomalyScorer
from src.alerts.alert_generator import AlertGenerator

settings = get_settings()

# Initialize FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Real-time fraud detection and anomaly scoring API"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (loaded on startup)
detector: Optional[EnsembleAnomalyDetector] = None
feature_engineer: Optional[FeatureEngineer] = None
scorer: Optional[AnomalyScorer] = None
alert_generator: Optional[AlertGenerator] = None


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class Transaction(BaseModel):
    """Transaction input schema"""
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp: datetime
    location_lat: Optional[float] = Field(None, ge=-90, le=90)
    location_lon: Optional[float] = Field(None, ge=-180, le=180)
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_12345",
                "user_id": "user_789",
                "merchant_id": "merchant_456",
                "amount": 125.50,
                "timestamp": "2024-01-26T14:30:00",
                "location_lat": 40.7128,
                "location_lon": -74.0060
            }
        }


class DetectionResult(BaseModel):
    """Detection result schema"""
    transaction_id: str
    is_anomaly: bool
    anomaly_score: float = Field(..., ge=0, le=1)
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float = Field(..., ge=0, le=1)
    contributing_factors: List[str]
    alert_generated: bool
    alert_id: Optional[str] = None
    processing_time_ms: float


class BatchDetectionRequest(BaseModel):
    """Batch detection request"""
    transactions: List[Transaction]


class BatchDetectionResponse(BaseModel):
    """Batch detection response"""
    results: List[DetectionResult]
    summary: Dict[str, Any]


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: datetime
    version: str


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    total_predictions: int
    anomaly_rate: float
    avg_score: float
    high_risk_count: int
    last_retrain: Optional[datetime]


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models and initialize components on startup"""
    global detector, feature_engineer, scorer, alert_generator
    
    print("Initializing Anomaly Detection System...")
    
    try:
        # Load trained model
        model_path = settings.MODEL_DIR / "ensemble_detector.joblib"
        if model_path.exists():
            detector = EnsembleAnomalyDetector.load(str(model_path))
            print("✓ Detector loaded")
        else:
            print("⚠ No trained model found. Please train a model first.")
            detector = EnsembleAnomalyDetector()
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer(
            lookback_days=settings.FEATURE_WINDOW_DAYS
        )
        print("✓ Feature engineer initialized")
        
        # Initialize scorer
        scorer = AnomalyScorer(
            high_threshold=settings.HIGH_PRIORITY_THRESHOLD,
            medium_threshold=settings.MEDIUM_PRIORITY_THRESHOLD
        )
        print("✓ Anomaly scorer initialized")
        
        # Initialize alert generator
        alert_generator = AlertGenerator()
        print("✓ Alert generator initialized")
        
        print("🚀 System ready!")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down Anomaly Detection System...")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=HealthCheck)
async def root():
    """Root endpoint - health check"""
    return HealthCheck(
        status="healthy",
        model_loaded=detector is not None and detector.fitted,
        timestamp=datetime.now(),
        version=settings.APP_VERSION
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Detailed health check"""
    return HealthCheck(
        status="healthy" if detector and detector.fitted else "degraded",
        model_loaded=detector is not None and detector.fitted,
        timestamp=datetime.now(),
        version=settings.APP_VERSION
    )


@app.post("/detect", response_model=DetectionResult)
async def detect_anomaly(
    transaction: Transaction,
    background_tasks: BackgroundTasks
):
    """
    Detect fraud in a single transaction
    
    Args:
        transaction: Transaction data
        
    Returns:
        Detection result with anomaly score and risk level
    """
    if not detector or not detector.fitted:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([transaction.dict()])
        
        # Engineer features
        df_features = feature_engineer.engineer_features(df, fit=False)
        feature_cols = feature_engineer.get_feature_names(df_features)
        X = df_features[feature_cols]
        
        # Get anomaly score
        anomaly_score = detector.predict_proba(X)[0]
        is_anomaly = detector.predict(X)[0]
        
        # Calculate risk level and confidence
        risk_level, confidence = scorer.calculate_risk_level(anomaly_score)
        
        # Identify contributing factors
        contributing_factors = scorer.identify_contributing_factors(
            df_features.iloc[0],
            feature_cols
        )
        
        # Generate alert if needed
        alert_id = None
        alert_generated = False
        
        if risk_level in ['HIGH', 'CRITICAL']:
            alert_id = alert_generator.generate_alert(
                transaction_id=transaction.transaction_id,
                anomaly_score=anomaly_score,
                risk_level=risk_level,
                details=contributing_factors
            )
            alert_generated = True
            
            # Send notification in background
            background_tasks.add_task(
                alert_generator.send_notification,
                alert_id
            )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return DetectionResult(
            transaction_id=transaction.transaction_id,
            is_anomaly=bool(is_anomaly),
            anomaly_score=float(anomaly_score),
            risk_level=risk_level,
            confidence=float(confidence),
            contributing_factors=contributing_factors,
            alert_generated=alert_generated,
            alert_id=alert_id,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    request: BatchDetectionRequest,
    background_tasks: BackgroundTasks
):
    """
    Detect fraud in multiple transactions (batch processing)
    
    Args:
        request: Batch of transactions
        
    Returns:
        Batch detection results with summary statistics
    """
    if not detector or not detector.fitted:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    results = []
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([t.dict() for t in request.transactions])
        
        # Engineer features
        df_features = feature_engineer.engineer_features(df, fit=False)
        feature_cols = feature_engineer.get_feature_names(df_features)
        X = df_features[feature_cols]
        
        # Batch prediction
        anomaly_scores = detector.predict_proba(X)
        is_anomalies = detector.predict(X)
        
        # Process each result
        for idx, transaction in enumerate(request.transactions):
            score = anomaly_scores[idx]
            is_anomaly = is_anomalies[idx]
            
            risk_level, confidence = scorer.calculate_risk_level(score)
            contributing_factors = scorer.identify_contributing_factors(
                df_features.iloc[idx],
                feature_cols
            )
            
            # Generate alert if needed
            alert_id = None
            alert_generated = False
            
            if risk_level in ['HIGH', 'CRITICAL']:
                alert_id = alert_generator.generate_alert(
                    transaction_id=transaction.transaction_id,
                    anomaly_score=score,
                    risk_level=risk_level,
                    details=contributing_factors
                )
                alert_generated = True
            
            results.append(DetectionResult(
                transaction_id=transaction.transaction_id,
                is_anomaly=bool(is_anomaly),
                anomaly_score=float(score),
                risk_level=risk_level,
                confidence=float(confidence),
                contributing_factors=contributing_factors,
                alert_generated=alert_generated,
                alert_id=alert_id,
                processing_time_ms=0  # Calculated at batch level
            ))
        
        # Calculate summary
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        summary = {
            "total_transactions": len(results),
            "anomaly_count": sum(r.is_anomaly for r in results),
            "anomaly_rate": sum(r.is_anomaly for r in results) / len(results),
            "avg_score": np.mean([r.anomaly_score for r in results]),
            "high_risk_count": sum(r.risk_level in ['HIGH', 'CRITICAL'] for r in results),
            "processing_time_ms": processing_time,
            "throughput_tps": len(results) / (processing_time / 1000)
        }
        
        return BatchDetectionResponse(
            results=results,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")


@app.get("/metrics", response_model=ModelMetrics)
async def get_metrics():
    """Get current model performance metrics"""
    # This would typically query a database or monitoring system
    # Simplified version here
    return ModelMetrics(
        total_predictions=0,
        anomaly_rate=0.05,
        avg_score=0.23,
        high_risk_count=0,
        last_retrain=None
    )


@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining (admin endpoint)"""
    # This would trigger the retraining pipeline
    background_tasks.add_task(retrain_models)
    return {"status": "retraining scheduled"}


async def retrain_models():
    """Background task for model retraining"""
    # Implement retraining logic
    print("Starting model retraining...")
    # ... retraining code ...
    print("Model retraining complete!")


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.DEBUG
    )