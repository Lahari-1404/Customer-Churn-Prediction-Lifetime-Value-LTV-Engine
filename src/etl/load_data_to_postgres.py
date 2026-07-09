import pandas as pd
from sqlalchemy import create_engine
from src.config import DATABASE_URL
import sys

def load_csv_to_postgres(csv_path, table_name="telco_customers_raw", if_exists="replace"):
    df = pd.read_csv(csv_path)
    # Basic cleaning for TotalCharges (sometimes blank)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    engine = create_engine(DATABASE_URL)
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"Loaded {len(df)} rows to {table_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_data_to_postgres.py <csv_path>")
        sys.exit(1)
    load_csv_to_postgres(sys.argv[1])
