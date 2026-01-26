# Fraud-Detection

# Anomaly Detection System

Real-time fraud detection using ensemble machine learning.

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