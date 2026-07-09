from dotenv import load_dotenv
import os
load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "telco")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "telco_pass")
DB_NAME = os.getenv("POSTGRES_DB", "telco_db")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
