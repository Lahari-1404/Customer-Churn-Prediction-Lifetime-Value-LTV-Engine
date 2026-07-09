from pydantic import BaseModel
from typing import Optional

class CustomerInput(BaseModel):
    customerID: str
    gender: Optional[str]
    SeniorCitizen: Optional[int]
    Partner: Optional[str]
    Dependents: Optional[str]
    tenure: Optional[int]
    PhoneService: Optional[str]
    MultipleLines: Optional[str]
    InternetService: Optional[str]
    OnlineSecurity: Optional[str]
    OnlineBackup: Optional[str]
    DeviceProtection: Optional[str]
    TechSupport: Optional[str]
    StreamingTV: Optional[str]
    StreamingMovies: Optional[str]
    Contract: Optional[str]
    PaperlessBilling: Optional[str]
    PaymentMethod: Optional[str]
    MonthlyCharges: Optional[float]
    TotalCharges: Optional[float]

class PredictionResponse(BaseModel):
    customerID: str
    churn_score: float
    predicted_ltv: float
