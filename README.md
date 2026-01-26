# Real-Time Fraud & Anomaly Detection System

## 🚀 Overview
A high-performance anomaly detection pipeline designed to identify fraudulent financial transactions. The system combines an Unsupervised Ensemble (Isolation Forest, KNN) with a Business Rule Engine to provide real-time, explainable risk assessments.

## 📊 Performance & Impact
* **Precision:** ~80% (Supervised XGBoost Baseline)
* **FPR:** < 1% (Minimal customer friction)
* **Latency:** < 400ms per transaction
* **Explainability:** Identifies contributing factors (e.g., "Critical high-value," "Far from home")

## 🛠️ Tech Stack
* **Language:** Python 3.10 (Optimized for ARM64)
* **Modeling:** Scikit-learn, XGBoost, LightGBM
* **API:** FastAPI, Uvicorn, Pydantic V2
* **Infrastructure:** Homebrew, PostgreSQL (Ready for Feast Integration)

## Quick Start

\`\`\`bash
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/train_models.py
python src/api/main.py
\`\`\`

## API Endpoints

- `GET /health` - Health check
- `POST /detect` - Detect single transaction
- `POST /detect/batch` - Batch detection
- `GET /metrics` - Performance metrics

## Documentation

See `docs/` folder for detailed documentation.