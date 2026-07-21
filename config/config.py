# import os
# from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent.parent

# MODEL_PATH = os.path.join(BASE_DIR, "model", "best_xgboost_synergy.pkl")
# FEATURES_PATH = os.path.join(BASE_DIR, "model", "model_features.pkl")

# API_TITLE = "Synergy Credit Scoring API"
# API_DESCRIPTION = "Sistem Pendukung Keputusan (SPK) Penentu Akad Syariah Berbasis XGBoost"
# API_VERSION = "1.0"

# BARUUU=============================

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load file .env
load_dotenv()

def get_db_connection():
    try:
        # Ambil masing-masing konfigurasi dari environment variables
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")

        # Cek jika variabel penting belum di-set di .env
        if not all([db_name, db_user, db_password]):
            raise ValueError("Konfigurasi database di .env belum lengkap!")

        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"[ERROR] Gagal koneksi ke database PostgreSQL: {e}")
        raise e