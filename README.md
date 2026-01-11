# Project 8: Arabic Sentiment Analyzer for Moroccan Films

## Key Objectives (from PDF):
- Adapt for Arabic text: preprocess dialectal Moroccan Arabic reviews
- Fine-tune embeddings for >85% F1-score
- Full MLOps: automated data pipelines, experiment tracking, CI/CD deployment
- Deploy as API for real-time review classification with confidence scores

## Technical Implementation:
- Model: Embedding(128)-LSTM(128)-Dense(1, sigmoid)
- Parameters: max_features=5000, maxlen=500
- Target: val_accuracy >0.87

## MLOps Stack:
- NLP: CAMeL Tools, Hazm
- Tracking: MLflow
- Versioning: DVC
- Orchestration: GitHub Actions
- Deployment: FastAPI + Docker
- Monitoring: Weights & Biases

## Deliverables Checklist:
- [x] Jupyter notebooks (EDA, training)
- [x] FastAPI app
- [x] Docker image
- [ ] MLflow UI screenshots
- [ ] Report: Arabic vs English performance table
- [ ] Deployment latency benchmarks
- [ ] Extensions: Multi-label, Streamlit dashboard
