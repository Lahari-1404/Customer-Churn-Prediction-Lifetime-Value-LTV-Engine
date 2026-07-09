import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def simple_preprocessing(df):
    df = df.copy()
    # Fill TotalCharges missing with MonthlyCharges * tenure
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
    # Binary encode target
    if 'Churn' in df.columns:
        df['ChurnFlag'] = df['Churn'].map({'Yes':1, 'No':0})
    # Convert SeniorCitizen to int
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    return df

def create_features(df):
    df = df.copy()
    # Recurring revenue ratio (TotalCharges / tenure) - careful for tenure 0
    df['avg_monthly_revenue'] = df['TotalCharges'] / (df['tenure'].replace(0, np.nan))
    df['avg_monthly_revenue'] = df['avg_monthly_revenue'].fillna(df['MonthlyCharges'])
    # tenure buckets
    df['tenure_bucket'] = pd.cut(df['tenure'], bins=[-1,6,12,24,48,72], labels=['0-6','7-12','13-24','25-48','49-72'])
    # numeric aggregated usage proxies
    df['is_senior'] = df['SeniorCitizen']
    # encode paperless billing binary
    df['paperless'] = df['PaperlessBilling'].map({'Yes':1,'No':0})
    return df
