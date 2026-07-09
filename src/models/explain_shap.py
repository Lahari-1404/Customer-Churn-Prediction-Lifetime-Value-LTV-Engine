import shap
import numpy as np
from src.models.model_utils import load_model, save_model
import pandas as pd
from sqlalchemy import create_engine
from src.config import DATABASE_URL
from src.features.feature_engineering import simple_preprocessing, create_features

def explain():
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql_table('telco_customers_raw', con=engine)
    df = simple_preprocessing(df)
    df = create_features(df)

    # load classifier
    clf = load_model("churn_classifier_v1")

    # We need raw features for the preprocessor
    numeric_feats = ['tenure','MonthlyCharges','TotalCharges','avg_monthly_revenue','is_senior','paperless']
    categorical_feats = ['gender','Contract','InternetService','PaymentMethod','tenure_bucket']
    X = df[numeric_feats + categorical_feats].fillna(0)

    # Get preprocessor and model from pipeline
    pre = clf.named_steps['pre']
    model = clf.named_steps['model']

    X_trans = pre.transform(X)
    # Use TreeExplainer for tree models
    explainer = shap.Explainer(model)
    shap_values = explainer(X_trans)
    # Save shap values summary plot (requires display or save)
    shap.summary_plot(shap_values, features=X_trans, show=False)
    print("SHAP explanation generated (visualization saved manually).")

if __name__ == "__main__":
    explain()
