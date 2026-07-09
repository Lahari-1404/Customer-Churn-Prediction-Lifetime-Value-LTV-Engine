import pandas as pd
from src.models.model_utils import load_model
from sqlalchemy import create_engine
from src.config import DATABASE_URL

def predict_batch(output_csv="predictions.csv"):
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql_table('telco_customers_raw', con=engine)
    clf = load_model("churn_classifier_v1")
    reg = load_model("ltv_regressor_v1")
    # Feature engineering
    from src.features.feature_engineering import simple_preprocessing, create_features
    df2 = simple_preprocessing(df)
    df2 = create_features(df2)

    numeric_feats = ['tenure','MonthlyCharges','TotalCharges','avg_monthly_revenue','is_senior','paperless']
    categorical_feats = ['gender','Contract','InternetService','PaymentMethod','tenure_bucket']
    X = df2[numeric_feats + categorical_feats].fillna(0)

    churn_probs = clf.predict_proba(X)[:,1]
    ltv_preds = reg.predict(X)
    out = df[['customerID']].copy()
    out['churn_score'] = churn_probs
    out['predicted_ltv'] = ltv_preds
    out.to_csv(output_csv, index=False)
    print(f"Wrote predictions to {output_csv}")

if __name__ == "__main__":
    predict_batch()
