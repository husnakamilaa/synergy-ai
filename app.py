from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="Synergy Credit Scoring API", 
    description="Sistem Pendukung Keputusan (SPK) Penentu Akad Syariah Berbasis XGBoost",
    version="1.0"
)

# =============================================================
# 1. LOAD ARTEFAK MODEL & FITUR BLUEPRINT
# =============================================================
MODEL_PATH = "best_xgboost_synergy.pkl"
FEATURES_PATH = "model_features.pkl"

# Pastikan file pkl berada di folder yang sama dengan app.py
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(
        "Artefak '[best_xgboost_synergy.pkl]' atau '[model_features.pkl]' tidak ditemukan di direktori aktif. "
        "Silakan unduh dari Google Drive dan letakkan berdampingan dengan file app.py ini."
    )

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(FEATURES_PATH, 'rb') as f:
    model_features = pickle.load(f)

print("[SUCCESS] Otak XGBoost dan Blueprint Fitur berhasil dimuat ke memori server!")

# =============================================================
# 2. DEFINISI REQUEST BODY (Format JSON yang wajib dikirim oleh BE)
# =============================================================
class UMKMInput(BaseModel):
    Total_Kebutuhan_Modal: int
    Aset_Lancar_Kas: int
    Aset_Tetap_Mesin_Alat: int
    Total_Hutang_Lancar: int
    Total_Pendapatan: int
    Total_Beban: int
    Cash_In: int
    Cash_Out: int
    Biaya_Bahan_Baku: int
    Biaya_Tenaga_Kerja: int
    Biaya_Overhead: int
    Frekuensi_Transaksi_Bulanan: int
    Rata_Rata_Nilai_Transaksi: int
    Saldo_Kas_Akhir: int
    Biaya_Penyusutan_Bulanan: int
    Penyesuaian_Persediaan: int
    Current_Ratio: float
    Net_Profit_Margin: float
    Operating_Expense_Ratio: float
    Pertumbuhan_Pendapatan: float
    Volatilitas_Arus_Kas: float
    Asset_Turnover_Ratio: float
    Labor_Productivity: int
    Cash_Turnover: float
    Lama_Usaha_Bulan: int
    Legalitas_NIB: int
    Riwayat_Pembiayaan_Sebelumnya: str  # 'Belum_Pernah', 'Lancar', 'Pernah_Macet'
    Usia_Pemilik: int
    Lama_Pendidikan_Tahun: int
    Pengalaman_Usaha_Tahun: int
    Jumlah_Karyawan: int
    Sektor_Usaha: str                  # 'Kuliner', 'Kerajinan', 'Pertanian', 'Perdagangan', 'Jasa'
    Skor_Risiko_Sektor: float
    Rasio_Transaksi_Digital: float
    Ada_Penjamin: int
    Kelompok_Tanggung_Renteng: int

# =============================================================
# 3. ENDPOINT API UNTUK PREDIKSI REAL-TIME
# =============================================================
@app.post("/predict")
def predict_akad(data: UMKMInput):
    try:
        # A. Ubah kiriman JSON menjadi Pandas DataFrame (1 Baris)
        input_dict = data.dict()
        df_input = pd.DataFrame([input_dict])
        
        # B. Lakukan One-Hot Encoding instan untuk variabel kategori teks
        df_encoded = pd.get_dummies(df_input, columns=['Sektor_Usaha', 'Riwayat_Pembiayaan_Sebelumnya'])
        
        # C. SELARASKAN FITUR (Anti-Crash): Pastikan kolom sama dengan blueprint training
        for col in model_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0  # Beri nilai 0 jika kolom kategori tersebut tidak aktif di data baru
                
        # Urutkan posisi kolom secara presisi agar XGBoost tidak salah membaca indeks biner
        df_final = df_encoded[model_features]
        
        # D. Eksekusi Prediksi Kelas & Confidence Score
        prediction = model.predict(df_final)[0]
        probability = model.predict_proba(df_final)[0] # Menghasilkan array [Prob_0, Prob_1]
        
        # Mapping hasil angka biner kembali ke teks akad syariah asli
        akad_result = "Musyarakah" if prediction == 1 else "Mudharabah"
        confidence_score = probability[1] if prediction == 1 else probability[0]
        
        # E. Kembalikan Respon JSON ke Backend Web Temanmu
        return {
            "status": "success",
            "rekomendasi_akad": akad_result,
            "confidence_score": round(float(confidence_score) * 100, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses prediksi di server AI: {str(e)}")