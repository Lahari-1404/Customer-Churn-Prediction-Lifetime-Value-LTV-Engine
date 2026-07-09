import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error
from src.config import MODEL_DIR
from src.models.model_utils import save_model
from src.features.feature_engineering import simple_preprocessing, create_features
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data_from_db(engine):
    return pd.read_sql_table('telco_customers_raw', con=engine)

def train():
    from sqlalchemy import create_engine
    from src.config import DATABASE_URL
    engine = create_engine(DATABASE_URL)
    df = load_data_from_db(engine)
    df = simple_preprocessing(df)
    df = create_features(df)

    # Target for churn
    y = df['ChurnFlag']
    # Feature selection
    numeric_feats = ['tenure','MonthlyCharges','TotalCharges','avg_monthly_revenue','is_senior','paperless']
    categorical_feats = ['gender','Contract','InternetService','PaymentMethod','tenure_bucket']

    X = df[numeric_feats + categorical_feats].copy()
    X[numeric_feats] = X[numeric_feats].fillna(0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)
    ])

    # Model: XGBoost classifier (tune hyperparams lightly)
    clf = Pipeline(steps=[
        ('pre', preprocessor),
        ('model', XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    print("Churn Classification Report:")
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))

    save_model(clf, "churn_classifier_v1")

    # LTV Regression: predict expected lifetime revenue (TotalCharges or derived)
    # For active customers (ChurnFlag==0) we can train to predict TotalCharges or use survival modeling.
    # Here we train a regressor to predict TotalCharges as proxy LTV (simpler).
    df_reg = df.copy()
    df_reg = df_reg[df_reg['TotalCharges'].notna()]
    y_reg = df_reg['TotalCharges']
    X_reg = df_reg[numeric_feats + categorical_feats]
    X_reg[numeric_feats] = X_reg[numeric_feats].fillna(0)

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg = Pipeline(steps=[
        ('pre', preprocessor),
        ('model', XGBRegressor(n_estimators=200, random_state=42))
    ])
    reg.fit(Xr_train, yr_train)
    ypred = reg.predict(Xr_test)
    print("LTV Regression RMSE:", np.sqrt(mean_squared_error(yr_test, ypred)))
    save_model(reg, "ltv_regressor_v1")

if __name__ == "__main__":
    train()
