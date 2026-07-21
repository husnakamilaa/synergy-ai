# import os
# import pickle

# from config.config import MODEL_PATH, FEATURES_PATH


# class ModelStore:

#     def __init__(self):
#         self.model = None
#         self.model_features = None

#     def load(self):
#         if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
#             raise FileNotFoundError(
#                 f"Artefak model tidak ditemukan.\n"
#                 f"  MODEL_PATH   : {MODEL_PATH}\n"
#                 f"  FEATURES_PATH: {FEATURES_PATH}\n"
#                 "Pastikan kedua file .pkl ada di folder model/."
#             )

#         with open(MODEL_PATH, "rb") as f:
#             self.model = pickle.load(f)

#         with open(FEATURES_PATH, "rb") as f:
#             self.model_features = pickle.load(f)

#         print("[SUCCESS] Otak XGBoost dan Blueprint Fitur berhasil dimuat ke memori server!")

# model_store = ModelStore()

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