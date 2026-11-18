# AgriNova · Tim Kasus 07

**Smart Irrigation IoT · Sistem Prediksi Kebutuhan Penyiraman Berbasis Machine Learning**

- **Kasus:** 7 — Smart Irrigation IoT
- **Aplikasi publik:** https://agrinova-fslyvdjrtdvbzjmavdy7hk.streamlit.app
- **Repositori:** https://github.com/adilharun19-create/AgriNova
- **Video demonstrasi:** Belum dicantumkan; tambahkan tautan setelah video diunggah.

## Anggota tim

| Anggota | NIM | Bagian yang dikerjakan |
|---|---:|---|
| Adil Harun Alfarizi | 2406176 | Pengembangan proyek bersama |
| Arya Adimanggala | 2403676 | Pengembangan proyek bersama |

## Masalah dan pengguna

AgriNova ditujukan sebagai alat bantu bagi petani, agronom, dan pengelola lahan untuk memperkirakan apakah suatu petak tanaman perlu disiram berdasarkan kondisi tanaman dan lingkungan. Pengguna memasukkan jenis tanaman dan tanah, umur tanaman, kelembapan tanah, suhu dan kelembapan udara, intensitas cahaya, curah hujan, prakiraan hujan enam jam ke depan, serta jam pengamatan. Sistem mengembalikan probabilitas dan rekomendasi penyiraman. Prediksi yang terlewat dapat membuat tanaman mengalami stres air, sedangkan rekomendasi yang keliru dapat menyebabkan pemborosan air atau penyiraman berlebihan.

## Data dan pembersihan

Dataset mentah `data/irigasi.csv` berisi **17.726 baris dan 12 kolom**, dengan catatan per jam dari **1 Agustus sampai 31 Oktober 2025**. Data mencakup 8 petak (P01–P08), tiga jenis tanaman (cabai, tomat, selada), dan tiga jenis tanah (pasir, lempung, liat). Target yang diprediksi adalah `perlu_siram` (0 = belum perlu disiram, 1 = perlu disiram).

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---:|---|
| Duplikat berdasarkan `waktu` dan `petak` | 150 | Dihapus dengan `drop_duplicates` berdasarkan kombinasi waktu dan petak. |
| Penulisan nama tanaman tidak konsisten | 527 | Spasi dibersihkan, huruf diubah menjadi lowercase, dan alias `cabe`/`lettuce` dinormalisasi menjadi `cabai`/`selada`. |
| Kelembapan tanah tidak masuk akal (di luar 0,1–100%) | 35 | Diubah menjadi nilai kosong (`NaN`), lalu diinterpolasi per petak. |
| Sensor kelembapan tanah macet (nilai sama selama ≥4 jam) | 84 | Ditandai sebagai `NaN`, lalu diinterpolasi per petak. |
| Intensitas cahaya negatif | 35 | Diubah menjadi `NaN`, lalu diinterpolasi per petak. |
| Suhu udara kosong | 261 | Diinterpolasi berdasarkan urutan waktu pada petak yang sama. |
| Prakiraan hujan enam jam kosong | 176 | Diisi 0 mm sesuai aturan preprocessing proyek. |
| Label positif di luar jam penyiraman (05.00–17.00) | 40 | Diubah menjadi label 0 hanya pada data training; label validation dan test tidak diubah. |

Jumlah pada tabel adalah jumlah kejadian/baris yang terdeteksi untuk tiap masalah; kategori dapat saling tumpang tindih sehingga angkanya tidak dijumlahkan untuk menghitung total baris. Setelah duplikat dihapus dan pembacaan pertama tiap petak dikeluarkan karena tidak memiliki pembacaan kelembapan satu jam sebelumnya, data yang digunakan berjumlah **17.568 baris**. Pergeseran pembacaan kelembapan satu jam ke belakang dilakukan untuk mencegah kebocoran data (*data leakage*): model menggunakan pembacaan terakhir sebelum keputusan penyiraman.

## Metode dan hasil

Metrik utama adalah **F1-score untuk kelas positif `perlu_siram = 1`**, karena kelas positif hanya sekitar 4,25% dari data mentah. Pembagian data dilakukan berdasarkan waktu: training sampai 30 September 2025, validation 1–14 Oktober 2025, dan test 15–31 Oktober 2025.

Pada notebook pencarian model (`notebooks/01_pencarian_model.ipynb`), lima algoritma dibandingkan pada validation dengan threshold 0,5. Hasil F1-score-nya:

| Metode/model | F1-score validation |
|---|---:|
| Baseline wajib: kelembapan tanah mentah < 35% | 0,191 |
| Aturan domain (batas aman tanaman, jam 05–17, prakiraan hujan ≤ 2 mm) | 0,805 |
| Logistic Regression | 0,435 |
| Decision Tree (depth 6) | 0,656 |
| Random Forest | 0,849 |
| Extra Trees | 0,792 |
| **HistGradientBoosting** | **0,851** |

HistGradientBoosting dipilih sebagai kandidat utama karena memperoleh F1-score validation tertinggi pada perbandingan tersebut dan memiliki rata-rata F1 tertinggi pada pemeriksaan stabilitas empat periode: **0,896**, dibandingkan Random Forest **0,894**.

Pada tahap final (`notebooks/02_model_final.ipynb`), HistGradientBoosting dituning dan threshold dipilih menggunakan validation. Hasil validasi empat periode memiliki rata-rata F1 **0,892**; pada prediksi validation gabungan, F1 **0,901**, precision **0,921**, dan recall **0,882**. Angka ini merupakan evaluasi validation tahap final, bukan hasil test.

Evaluasi akhir pada test (15–31 Oktober 2025):

| Metode | F1-score test |
|---|---:|
| Baseline wajib: kelembapan tanah mentah < 35% | 0,182 |
| Aturan domain | 0,810 |
| **HistGradientBoosting (model final)** | **0,828** |

Model final pada test memperoleh **precision 0,897** dan **recall 0,769**.

**Threshold yang dipakai: 0,5.** Threshold dipilih sebagai threshold terendah yang F1-nya masih berada dalam 0,005 dari nilai maksimum pada validasi empat periode. Pemilihan ini condong ke recall karena notifikasi kebutuhan penyiraman yang terlewat dapat menyebabkan tanaman mengalami stres. Hasil test dipakai sebagai evaluasi akhir, bukan untuk memilih model atau threshold.

## Aplikasi dan API

Aplikasi publik dibangun menggunakan **Streamlit**. Pada deployment Streamlit saat ini, `app.py` memanggil `src/predictor.py`; model dan metadata dimuat dalam proses aplikasi yang sama. Dengan demikian, aplikasi publik **tidak memanggil server FastAPI terpisah**.

Proyek juga menyediakan implementasi API terpisah di `src/api.py` menggunakan FastAPI:
- `GET /` — informasi status dan versi model.
- `GET /health` — pemeriksaan kesiapan model.
- `POST /predict` — menerima data sensor tervalidasi dan mengembalikan probabilitas serta rekomendasi penyiraman.

Dokumentasi interaktif API dapat dibuka secara lokal melalui `http://127.0.0.1:8000/docs` setelah server FastAPI dijalankan.

## Cara menjalankan

Gunakan versi Python dan library yang kompatibel dengan artifact model. Metadata model mencatat Python **3.15.2**, scikit-learn **1.9.1**, pandas **3.1.1**, NumPy **2.5.5**, dan joblib **1.6.0**.

### Menjalankan aplikasi Streamlit

```bash
python -m venv .venv
```

Aktifkan environment pada Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instal dependensi dan jalankan aplikasi:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

### Menjalankan FastAPI secara terpisah (opsional)

Dependensi FastAPI tidak tercantum pada `requirements.txt` deployment Streamlit. Instal dependensinya terlebih dahulu:

```bash
python -m pip install fastapi uvicorn pydantic
python -m uvicorn src.api:app --reload
```

Kemudian buka dokumentasi Swagger di `http://127.0.0.1:8000/docs`.

## Struktur repositori

```text
AgriNova/
├── app.py                         # Antarmuka Streamlit
├── requirements.txt               # Dependensi deployment Streamlit
├── data/
│   └── irigasi.csv                # Dataset mentah
├── notebooks/
│   ├── 01_pencarian_model.ipynb   # EDA, preprocessing, dan perbandingan model
│   └── 02_model_final.ipynb       # Tuning, threshold, evaluasi test, simpan artifact
├── artifacts/
│   ├── model.joblib               # Pipeline model HistGradientBoosting final
│   └── metadata.json              # Threshold, fitur, batas aman, metrik, versi library
└── src/
    ├── __init__.py
    ├── api.py                     # API FastAPI (opsional, terpisah)
    ├── features.py                # Pembersihan, feature engineering, dan alasan prediksi
    └── predictor.py               # Inference lokal yang dipakai Streamlit
```

## Keterbatasan

- Dataset yang digunakan mencakup periode Agustus–Oktober 2025, delapan petak, tiga jenis tanaman, dan tiga jenis tanah. Kinerja pada tanaman, tanah, wilayah, musim, atau kondisi iklim di luar cakupan data belum dapat dipastikan.
- Aplikasi meminta pengguna memasukkan kondisi sensor dan prakiraan hujan. Deployment saat ini belum menunjukkan integrasi langsung dengan perangkat sensor IoT atau layanan prakiraan cuaca eksternal.
- Prakiraan hujan yang kosong pada data training diisi 0 mm; asumsi ini dapat memengaruhi hasil jika nilai kosong sebenarnya berarti data cuaca tidak tersedia, bukan tidak ada hujan.
- Kelas `perlu_siram = 1` jauh lebih sedikit daripada kelas 0. Walaupun evaluasi menggunakan F1-score dan pembobotan kelas, model masih dapat melewatkan sebagian kasus yang membutuhkan penyiraman.
- Hasil prediksi bergantung pada ketepatan dan kalibrasi input. Data sensor yang rusak, tidak terkalibrasi, atau berada di luar rentang penggunaan dapat menghasilkan rekomendasi yang tidak dapat diandalkan.
- Rekomendasi merupakan alat bantu keputusan, bukan pengganti pemeriksaan kondisi tanaman dan keputusan agronom di lapangan.
