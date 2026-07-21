# """
# Entry point aplikasi FastAPI.

# Menyambungkan semua bagian:
# - membuat objek FastAPI
# - memuat model saat startup
# - mendaftarkan router (endpoint)

# Jalankan dengan:  uvicorn app:app --host 127.0.0.1 --port 8000
# """
# from fastapi import FastAPI

# from config.config import API_TITLE, API_DESCRIPTION, API_VERSION
# from ml.ml_model import model_store
# from routers import predict

# app = FastAPI(
#     title=API_TITLE,
#     description=API_DESCRIPTION,
#     version=API_VERSION,
# )


# @app.on_event("startup")
# def startup_event():
#     """Muat model .pkl sekali saat server start."""
#     model_store.load()


# # Daftarkan endpoint dari router
# app.include_router(predict.router)


# @app.get("/health")
# def health_check():
#     """Cek apakah API hidup."""
#     return {"status": "ok"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.predict import router as predict_router

app = FastAPI(
    title="SYNERGY AI - XGBoost Scoring Engine",
    description="Engine AI untuk Analisis Kelayakan dan Prediksi Akad UMKM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Router
app.include_router(predict_router)

@app.get("/", tags=["Health Check"])
def root():
    return {"status": "online", "message": "SYNERGY AI Engine Active"}