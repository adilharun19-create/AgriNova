"""Local inference engine shared by the Streamlit deployment.

The trained pipeline and metadata are loaded once when the app starts.
This avoids requiring a separate FastAPI process on Streamlit Community Cloud.
"""
from pathlib import Path
import json

import joblib
import pandas as pd

from src.features import buat_alasan

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
META_PATH = ROOT / "artifacts" / "metadata.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
if not META_PATH.exists():
    raise FileNotFoundError(f"Metadata tidak ditemukan: {META_PATH}")

model = joblib.load(MODEL_PATH)
with META_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

THRESHOLD = float(metadata["threshold"])
BATAS_AMAN = metadata["batas_aman_kelembaban_pct"]


def health() -> dict:
    """Return local model readiness information."""
    return {"status": "healthy", "model_loaded": model is not None}


def predict(input_data: dict) -> dict:
    """Generate a prediction using the same response contract as the API."""
    required = metadata.get("urutan_kolom_input", [])
    missing = [key for key in required if key not in input_data]
    if missing:
        raise ValueError(f"Kolom input hilang: {missing}")

    df_input = pd.DataFrame([{key: input_data[key] for key in required}])
    probabilitas = float(model.predict_proba(df_input)[0, 1])
    perlu_siram = probabilitas >= THRESHOLD
    alasan = buat_alasan(input_data, BATAS_AMAN, probabilitas, THRESHOLD)

    return {
        "status": "success",
        "hasil_prediksi": {
            "perlu_siram": bool(perlu_siram),
            "rekomendasi": "Perlu disiram" if perlu_siram else "Belum perlu disiram",
            "probabilitas": round(probabilitas, 4),
            "threshold": THRESHOLD,
        },
        "alasan": alasan,
        "data_sensor": input_data,
    }
