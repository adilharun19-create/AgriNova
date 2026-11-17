"""Fitur untuk model irigasi pintar (Kasus 07).

Dipakai di dua tempat supaya fiturnya selalu sama:
- saat training (notebook), dan
- saat prediksi (API / app).

Input model = satu pembacaan per petak, dengan kolom KOLOM_INPUT.
`kelembaban_tanah_pct` di sini berarti PEMBACAAN TERAKHIR SEBELUM keputusan
(saat training: nilai jam sebelumnya, untuk menghindari kebocoran data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TANAMAN = ["cabai", "selada", "tomat"]
JENIS_TANAH = ["lempung", "liat", "pasir"]

KOLOM_INPUT = [
    "tanaman", "jenis_tanah", "umur_tanaman_hari", "kelembaban_tanah_pct",
    "suhu_udara_C", "kelembaban_udara_pct", "cahaya_lux",
    "curah_hujan_mm", "prakiraan_hujan_6jam_mm", "jam",
]

JAM_SIANG = (5, 17)  # agronom hanya menyiram jam 05–17

ALIAS_TANAMAN = {"cabe": "cabai", "lettuce": "selada"}


def normalisasi_tanaman(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().replace(ALIAS_TANAMAN)


# ---------------------------------------------------------------------------
# Pembersihan data mentah (dipakai saat training, butuh data deret waktu)
# ---------------------------------------------------------------------------
def bersihkan_data_mentah(df_raw: pd.DataFrame, min_jam_macet: int = 4) -> tuple[pd.DataFrame, dict]:
    """Bersihkan irigasi.csv dan buat kolom input model.

    Mengembalikan (df_bersih, laporan) dengan laporan = jumlah baris per masalah.
    Kolom `kelembaban_tanah_pct` di hasil sudah digeser 1 jam (pembacaan terakhir).
    """
    lap = {}
    df = df_raw.copy()
    df["waktu"] = pd.to_datetime(df["waktu"])

    n = len(df)
    df = df.drop_duplicates(subset=["waktu", "petak"]).sort_values(["petak", "waktu"]).reset_index(drop=True)
    lap["duplikat"] = n - len(df)

    tanaman_baru = normalisasi_tanaman(df["tanaman"])
    lap["nama_tanaman_tidak_konsisten"] = int((tanaman_baru != df["tanaman"]).sum())
    df["tanaman"] = tanaman_baru

    k = "kelembaban_tanah_pct"
    mustahil = ~df[k].between(0.1, 100)
    sama = df[k].eq(df.groupby("petak")[k].shift(1))
    panjang_run = df.groupby(["petak", (~sama).cumsum()])[k].transform("size")
    macet = panjang_run >= min_jam_macet
    cahaya_neg = df["cahaya_lux"] < 0
    lap["kelembaban_tanah_mustahil"] = int(mustahil.sum())
    lap["sensor_tanah_macet"] = int(macet.sum())
    lap["cahaya_negatif"] = int(cahaya_neg.sum())
    lap["suhu_kosong"] = int(df["suhu_udara_C"].isna().sum())
    lap["prakiraan_hujan_kosong"] = int(df["prakiraan_hujan_6jam_mm"].isna().sum())

    df.loc[mustahil | macet, k] = np.nan
    df.loc[cahaya_neg, "cahaya_lux"] = np.nan
    for kol in [k, "suhu_udara_C", "cahaya_lux"]:
        df[kol] = df.groupby("petak")[kol].transform(lambda s: s.interpolate(limit_direction="both"))
    df["prakiraan_hujan_6jam_mm"] = df["prakiraan_hujan_6jam_mm"].fillna(0)

    df["jam"] = df["waktu"].dt.hour
    # Perbaikan leakage: pakai pembacaan jam sebelumnya
    df["kelembaban_tanah_saat_itu"] = df[k]
    df[k] = df.groupby("petak")[k].shift(1)
    df = df.dropna(subset=[k]).reset_index(drop=True)

    siang = df["jam"].between(*JAM_SIANG)
    lap["label_positif_malam"] = int(((~siang) & (df["perlu_siram"] == 1)).sum())
    return df, lap


def label_latih(df: pd.DataFrame) -> pd.Series:
    """Label untuk training: positif di malam hari dianggap salah label → 0."""
    y = df["perlu_siram"].copy()
    y[~df["jam"].between(*JAM_SIANG) & (y == 1)] = 0
    return y


# ---------------------------------------------------------------------------
# Transformer fitur (masuk ke dalam pipeline joblib)
# ---------------------------------------------------------------------------
class FiturIrigasi(BaseEstimator, TransformerMixin):
    """Ubah input mentah (KOLOM_INPUT) jadi matriks fitur.

    Saat fit, batas aman kelembaban tiap tanaman dihitung dari data:
    titik tengah antara kuantil 90% kelembaban yang disiram dan kuantil 5%
    kelembaban yang tidak disiram (siang hari saja).
    """

    def fit(self, X, y=None):
        X = self._siapkan(X)
        self.batas_aman_ = {"cabai": 33.0, "tomat": 37.0, "selada": 43.0}  # cadangan
        if y is not None:
            y = pd.Series(np.asarray(y), index=X.index)
            siang = X["jam"].between(*JAM_SIANG)
            for t in TANAMAN:
                m = siang & (X["tanaman"] == t)
                pos = X.loc[m & (y == 1), "kelembaban_tanah_pct"]
                neg = X.loc[m & (y == 0), "kelembaban_tanah_pct"]
                if len(pos) >= 10 and len(neg) >= 10:
                    self.batas_aman_[t] = float(round((pos.quantile(0.9) + neg.quantile(0.05)) / 2, 1))
        self.feature_names_out_ = self.transform(X.head(1)).columns.tolist()
        return self

    @staticmethod
    def _siapkan(X) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        hilang = [c for c in KOLOM_INPUT if c not in X.columns]
        if hilang:
            raise ValueError(f"Kolom input hilang: {hilang}")
        X = X[KOLOM_INPUT]
        X["tanaman"] = normalisasi_tanaman(X["tanaman"])
        X["jenis_tanah"] = X["jenis_tanah"].astype(str).str.strip().str.lower()
        num = [c for c in KOLOM_INPUT if c not in ("tanaman", "jenis_tanah")]
        X[num] = X[num].apply(pd.to_numeric, errors="coerce").astype(float)
        X["prakiraan_hujan_6jam_mm"] = X["prakiraan_hujan_6jam_mm"].fillna(0)
        return X

    def transform(self, X):
        X = self._siapkan(X)
        out = X.drop(columns=["tanaman", "jenis_tanah"])
        out["siang"] = X["jam"].between(*JAM_SIANG).astype(int)
        batas = X["tanaman"].map(getattr(self, "batas_aman_", {}))
        out["selisih_batas"] = X["kelembaban_tanah_pct"] - batas
        for t in TANAMAN:
            out[f"tanaman_{t}"] = (X["tanaman"] == t).astype(int)
        for j in JENIS_TANAH:
            out[f"jenis_tanah_{j}"] = (X["jenis_tanah"] == j).astype(int)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


# ---------------------------------------------------------------------------
# Alasan untuk ditampilkan di app
# ---------------------------------------------------------------------------
def buat_alasan(baris: dict, batas_aman: dict, prob: float, threshold: float) -> list[str]:
    """Kalimat penjelasan sederhana berbasis aturan domain untuk satu input."""
    alasan = []
    t = normalisasi_tanaman(pd.Series([baris["tanaman"]])).iloc[0]
    k = float(baris["kelembaban_tanah_pct"])
    b = batas_aman.get(t)
    if b is not None:
        if k < b:
            alasan.append(f"Kelembaban tanah {k:.1f}% di bawah batas aman {t} ({b:.0f}%).")
        else:
            alasan.append(f"Kelembaban tanah {k:.1f}% masih di atas batas aman {t} ({b:.0f}%).")
    jam = int(baris["jam"])
    if not (JAM_SIANG[0] <= jam <= JAM_SIANG[1]):
        alasan.append(f"Jam {jam:02d}:00 di luar jam siram ({JAM_SIANG[0]:02d}–{JAM_SIANG[1]:02d}).")
    hujan = float(baris.get("prakiraan_hujan_6jam_mm") or 0)
    if hujan > 2:
        alasan.append(f"Prakiraan hujan 6 jam ke depan {hujan:.1f} mm, penyiraman bisa ditunda.")
    if str(baris.get("jenis_tanah", "")).lower() == "pasir" and b is not None and k < b + 3:
        alasan.append("Tanah pasir cepat kering.")
    alasan.append(f"Probabilitas perlu siram {prob:.0%} (threshold {threshold:.0%}).")
    return alasan
