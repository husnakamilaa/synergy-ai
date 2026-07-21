# """
# Orkestrator prediksi.

# Menggabungkan langkah prediksi jadi satu alur:
#   1. ubah input jadi DataFrame
#   2. one-hot encoding kolom kategori
#   3. selaraskan kolom dengan blueprint fitur (anti-crash)
#   4. prediksi + hitung confidence
#   5. format hasil

# Tahap 2: langkah kalkulasi fitur akan disisipkan sebelum encoding.
# """
# import pandas as pd

# from ml.ml_model import model_store


# def run_prediction(input_dict: dict) -> dict:
#     """Jalankan prediksi akad dari data input (dict). Kembalikan dict hasil."""
#     model = model_store.model
#     model_features = model_store.model_features

#     # A. JSON -> DataFrame (1 baris)
#     df_input = pd.DataFrame([input_dict])

#     # B. One-Hot Encoding kolom kategori teks
#     df_encoded = pd.get_dummies(
#         df_input,
#         columns=["Sektor_Usaha", "Riwayat_Pembiayaan_Sebelumnya"],
#     )

#     # C. Selaraskan fitur dengan blueprint training
#     for col in model_features:
#         if col not in df_encoded.columns:
#             df_encoded[col] = 0  # kolom kategori tak aktif diisi 0

#     # Urutkan kolom persis sesuai blueprint
#     df_final = df_encoded[model_features]

#     # D. Prediksi kelas & confidence
#     prediction = model.predict(df_final)[0]
#     probability = model.predict_proba(df_final)[0]  # [Prob_0, Prob_1]

#     akad_result = "Musyarakah" if prediction == 1 else "Mudharabah"
#     confidence_score = probability[1] if prediction == 1 else probability[0]

#     # E. Susun hasil
#     return {
#         "status": "success",
#         "rekomendasi_akad": akad_result,
#         "confidence_score": round(float(confidence_score) * 100, 2),
#     }


import pandas as pd
from ml.ml_model import xgboost_model, model_features, label_encoder

def process_prediction_and_save(conn, id_umkm: str, id_akad_variable: int, features_dict: dict, skor_kelayakan: float, id_pendapatan: int):
    # 1. Susun DataFrame sesuai urutan fitur cetakan (model_features.pkl)
    df_input = pd.DataFrame([features_dict])[model_features]

    # 2. Prediksi dengan XGBoost & Decode Label
    pred_num = xgboost_model.predict(df_input)[0]
    akad_label = label_encoder.inverse_transform([pred_num])[0]  # Hasil: 'Mudharabah' atau 'Musyarakah'

    # 3. INSERT HASIL KE TABEL akad_analisis POSTGRESQL
    cursor = conn.cursor()
    query = """
        INSERT INTO akad_analisis (
            id_umkm, id_akad_variable, id_pendapatan,
            current_ratio, net_profit_margin, operating_expense_ratio,
            cashflow_stability_risk, asset_turnover_ratio,
            skor_kelayakan, akad
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    cursor.execute(query, (
        id_umkm, id_akad_variable, id_pendapatan,
        features_dict['current_ratio'],
        features_dict['net_profit_margin'],
        features_dict['operating_expense_ratio'],
        features_dict['cashflow_stability_risk'],
        features_dict['asset_turnover_ratio'],
        skor_kelayakan,
        akad_label
    ))
    
    new_id = cursor.fetchone()['id']
    conn.commit()
    cursor.close()

    return new_id, akad_label