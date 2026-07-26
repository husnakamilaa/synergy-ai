
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
        features_dict['revenue_growth'],
        skor_kelayakan,
        akad_label
    ))
    
    new_id = cursor.fetchone()['id']
    conn.commit()
    cursor.close()

    return new_id, akad_label