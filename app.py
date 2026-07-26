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