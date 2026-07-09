from fastapi import FastAPI, HTTPException, UploadFile, File
from src.models.model_utils import load_model
from src.api.schemas import CustomerInput, PredictionResponse
import pandas as pd
import io
from typing import List
from src.features.feature_engineering import simple_preprocessing, create_features

app = FastAPI(title="Telco Churn & LTV Prediction API")

@app.on_event("startup")
def load_models():
    global clf, reg
    clf = load_model("churn_classifier_v1")
    reg = load_model("ltv_regressor_v1")

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    # Convert input to DataFrame
    data = pd.DataFrame([customer.dict()])
    data = simple_preprocessing(data)
    data = create_features(data)
    numeric_feats = ['tenure','MonthlyCharges','TotalCharges','avg_monthly_revenue','is_senior','paperless']
    categorical_feats = ['gender','Contract','InternetService','PaymentMethod','tenure_bucket']
    X = data[numeric_feats + categorical_feats].fillna(0)
    churn_score = float(clf.predict_proba(X)[:,1][0])
    predicted_ltv = float(reg.predict(X)[0])
    return PredictionResponse(customerID=customer.customerID, churn_score=churn_score, predicted_ltv=predicted_ltv)

@app.post("/predict_batch")
def predict_batch(file: UploadFile = File(...)):
    # Accept CSV with customers
    content = file.file.read()
    df = pd.read_csv(io.BytesIO(content))
    df = simple_preprocessing(df)
    df = create_features(df)
    numeric_feats = ['tenure','MonthlyCharges','TotalCharges','avg_monthly_revenue','is_senior','paperless']
    categorical_feats = ['gender','Contract','InternetService','PaymentMethod','tenure_bucket']
    X = df[numeric_feats + categorical_feats].fillna(0)
    churn_scores = clf.predict_proba(X)[:,1]
    ltvs = reg.predict(X)
    out = df[['customerID']].copy()
    out['churn_score'] = churn_scores
    out['predicted_ltv'] = ltvs
    return out.to_dict(orient='records')
