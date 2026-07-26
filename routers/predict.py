# """
# Definisi endpoint prediksi.

# Endpoint di sini tipis: terima request -> panggil service -> kembalikan hasil.
# Logika berat ada di services/prediction.py.
# """
# from fastapi import APIRouter, HTTPException

# from schemas.umkm import UMKMInput, PredictionResponse
# from services.prediction import run_prediction

# router = APIRouter()


# @router.post("/predict", response_model=PredictionResponse)
# def predict_akad(data: UMKMInput):
#     try:
#         input_dict = data.dict()
#         return run_prediction(input_dict)
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Gagal memproses prediksi di server AI: {str(e)}",
#         )

from fastapi import APIRouter, HTTPException
from schemas.umkm import AnalisisRequest, AnalisisResponse, DetailAnalisisResponse
from config.config import get_db_connection
from services.feature_engineering import fetch_and_calculate_features
from services.prediction import process_prediction_and_save

router = APIRouter(prefix="/api", tags=["Analisis Akad"])

# -------------------------------------------------------------
# 1. ENDPOINT POST: TRIGGER KALKULASI, PREDIKSI ML, & SIMPAN
# -------------------------------------------------------------
@router.post("/analyze", response_model=AnalisisResponse)
def analyze_umkm(payload: AnalisisRequest):
    conn = get_db_connection()
    try:
        # Step 1: GET data mentah & Hitung 6 Rasio + Skor
        features_dict, skor_kelayakan, id_pendapatan = fetch_and_calculate_features(
            conn, payload.id_umkm, payload.id_akad_variable
        )

        # Step 2: Predict ML & Save ke PostgreSQL
        id_analisis_baru, akad_label = process_prediction_and_save(
            conn, payload.id_umkm, payload.id_akad_variable, 
            features_dict, skor_kelayakan, id_pendapatan
        )

        return AnalisisResponse(
            status="success",
            id_analisis=id_analisis_baru,
            id_umkm=payload.id_umkm,
            skor_kelayakan=skor_kelayakan,
            akad=akad_label,
            message="Analisis berhasil dihitung, diprediksi ML, dan disimpan ke database."
        )

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()


# -------------------------------------------------------------
# 2. ENDPOINT GET: AMBIL HASIL ANALISIS DENGAN ID ANALISIS
# -------------------------------------------------------------
@router.get("/analisis/{id_analisis}", response_model=DetailAnalisisResponse)
def get_analisis_by_id(id_analisis: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM akad_analisis WHERE id = %s", (id_analisis,))
        result = cursor.fetchone()

        if not result:
            raise HTTPException(status_code=404, detail=f"Data analisis dengan ID {id_analisis} tidak ditemukan!")

        return DetailAnalisisResponse(
            status="success",
            data=dict(result)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        conn.close()


# -------------------------------------------------------------
# 3. ENDPOINT GET: AMBIL SEMUA RIWAYAT ANALISIS MILIK SUATU UMKM
# -------------------------------------------------------------
@router.get("/analisis/umkm/{id_umkm}")
def get_analisis_by_umkm(id_umkm: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM akad_analisis WHERE id_umkm = %s ORDER BY id DESC", (id_umkm,))
        results = cursor.fetchall()

        return {
            "status": "success",
            "count": len(results),
            "data": [dict(r) for r in results]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        conn.close()