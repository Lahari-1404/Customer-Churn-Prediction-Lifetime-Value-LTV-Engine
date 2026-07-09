import joblib
from pathlib import Path
from src.config import MODEL_DIR

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

def save_model(obj, name):
    path = Path(MODEL_DIR) / f"{name}.joblib"
    joblib.dump(obj, path)
    return str(path)

def load_model(name):
    path = Path(MODEL_DIR) / f"{name}.joblib"
    return joblib.load(path)
