
from pathlib import Path
import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from src.features import buat_alasan


# ==========================================
# 1. LOKASI PROJECT DAN MODEL
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "artifacts" / "model.joblib"
META_PATH = ROOT / "artifacts" / "metadata.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

if not META_PATH.exists():
    raise FileNotFoundError(f"Metadata tidak ditemukan: {META_PATH}")


# ==========================================
# 2. LOAD MODEL
# ==========================================

model = joblib.load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

THRESHOLD = float(metadata["threshold"])
BATAS_AMAN = metadata["batas_aman_kelembaban_pct"]


# ==========================================
# 3. INISIALISASI FASTAPI
# ==========================================

app = FastAPI(
    title="Smart Irrigation IoT",
    description="Machine Learning API untuk prediksi kebutuhan penyiraman tanaman.",
    version="1.0.0"
)


# ==========================================
# 4. FORMAT INPUT SENSOR
# ==========================================

class DataSensor(BaseModel):

    tanaman: Literal["cabai", "selada", "tomat"]

    jenis_tanah: Literal["pasir", "lempung", "liat"]

    umur_tanaman_hari: int = Field(ge=0, le=365)

    kelembaban_tanah_pct: float = Field(ge=0.1, le=100)

    suhu_udara_C: float = Field(ge=0, le=50)

    kelembaban_udara_pct: float = Field(ge=0, le=100)

    cahaya_lux: float = Field(ge=0, le=150000)

    curah_hujan_mm: float = Field(ge=0, le=100)

    prakiraan_hujan_6jam_mm: float = Field(ge=0, le=200)

    jam: int = Field(ge=0, le=23)


# ==========================================
# 5. HALAMAN UTAMA
# ==========================================

@app.get("/")
def home():

    return {
        "status": "success",
        "message": "Smart Irrigation IoT API is running",
        "model": metadata["algoritma"],
        "version": metadata["versi_model"]
    }


# ==========================================
# 6. HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ==========================================
# 7. PREDIKSI IRIGASI
# ==========================================

@app.post("/predict")
def predict(data: DataSensor):

    try:

        input_data = data.model_dump()

        df_input = pd.DataFrame([input_data])

        # Prediksi probabilitas
        probabilitas = float(
            model.predict_proba(df_input)[0, 1]
        )

        # Keputusan berdasarkan threshold model
        perlu_siram = probabilitas >= THRESHOLD

        # Penjelasan hasil
        alasan = buat_alasan(
            input_data,
            BATAS_AMAN,
            probabilitas,
            THRESHOLD
        )

        return {

            "status": "success",

            "hasil_prediksi": {

                "perlu_siram": bool(perlu_siram),

                "rekomendasi": (
                    "Perlu disiram"
                    if perlu_siram
                    else "Belum perlu disiram"
                ),

                "probabilitas": round(probabilitas, 4),

                "threshold": THRESHOLD

            },

            "alasan": alasan,

            "data_sensor": input_data

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
