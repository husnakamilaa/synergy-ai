
import pickle
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../model')

try:
    with open(os.path.join(MODEL_DIR, 'best_xgboost_synergy.pkl'), 'rb') as f:
        xgboost_model = pickle.load(f)

    with open(os.path.join(MODEL_DIR, 'model_features.pkl'), 'rb') as f:
        model_features = pickle.load(f)

    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
        
    print("[INFO] Berhasil memuat 3 artefak ML (.pkl) ke memori.")

except Exception as e:
    print(f"[ERROR] Gagal memuat artefak model .pkl: {e}")
    raise e