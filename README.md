# Telco Churn Prediction & LTV Engine

Overview:
Predict customer churn and estimate Customer Lifetime Value (LTV) for Telco dataset. Provides training scripts, SHAP explainability, FastAPI inference endpoints, and instructions for visualization with Apache Superset.

Prerequisites:
- Docker & docker-compose
- (Optional) Python 3.11 and virtualenv to run locally

Quickstart (docker-compose):
1. Copy .env.example to .env and set values.
2. Place telco_customer_churn.csv in ./data/
3. Start services:
   docker-compose up -d --build
4. Load dataset into Postgres:
   docker exec -it <app_container_name> python src/etl/load_data_to_postgres.py /app/data/telco_customer_churn.csv
   (or run locally: python src/etl/load_data_to_postgres.py data/telco_customer_churn.csv)
5. Train models:
   docker exec -it <app_container_name> python src/models/train_models.py
   This will create models in /app/models inside container.
6. Start API:
   accessible at http://localhost:8000
   - POST /predict for single customer JSON
   - POST /predict_batch (multipart CSV)

Superset:
- Access


