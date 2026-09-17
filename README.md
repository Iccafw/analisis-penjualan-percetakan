# Analisis & Peramalan Penjualan Perusahaan Percetakan

Analisis data penjualan harian sebuah perusahaan percetakan (Agustus 2022 – November 2023),
mencakup pembersihan data, analisis eksploratif, pemodelan peramalan pendapatan, dan
klasifikasi portofolio produk untuk perencanaan persediaan.

---

## Ringkasan Proyek

| | |
|---|---|
| **Domain** | Manufaktur / percetakan B2B |
| **Data** | 1.076 transaksi, 15 bulan, 5 kolom |
| **Masalah** | Perusahaan tidak punya dasar kuantitatif untuk merencanakan produksi dan stok bahan baku |
| **Pendekatan** | Time series forecasting + klasifikasi ABC-XYZ |
| **Hasil** | Model peramalan mingguan dengan MAPE ~34% (backtesting), plus identifikasi 53 produk kandidat rasionalisasi |

---

## Temuan Utama

**1. Sistem pencatatan produk bermasalah serius.**
Dataset memuat 94 nama produk unik, tetapi setelah normalisasi hanya tersisa
**9 kategori** yang benar-benar berbeda. Variasi seperti `Dupleks310` / `Duplex310`,
atau `GreaseProof40` / `Greseproof40` / `Gressproff`, menyebabkan penjualan satu produk
terpecah ke beberapa nama — artinya **laporan penjualan yang selama ini dihasilkan
perusahaan kemungkinan besar keliru.** Ditemukan pula satu entri berisi nama perusahaan
(`CemerlangIndahSelaras`) yang tampaknya salah masuk kolom produk.

**2. Data harian terlalu jarang untuk diramalkan.**
Sebanyak **39% hari kalender tidak memiliki transaksi sama sekali**. Peramalan
dilakukan pada level mingguan, yang menurunkan koefisien variasi dari 1,45 menjadi 0,64
tanpa menghilangkan informasi tren.

**3. Pemilihan model berbasis AIC bisa menyesatkan.**
Grid search SARIMA tanpa batasan memilih `SARIMA(2,2,2)x(0,1,1,4)` — model yang
**kalah dari baseline Naive** (MAPE 39,8% vs 35,3%) dan menghasilkan prediksi 80% di
atas rata-rata historis akibat *over-differencing*. Setelah ruang pencarian dibatasi
sesuai hasil uji stasioneritas ADF, MAPE turun ke 26,9%.

**4. Evaluasi satu kali terlalu optimis.**
Dengan satu pembagian train/test, model terbaik mencapai MAPE 19,2%. Dengan
**rolling-origin cross-validation** (6 fold), angka realistisnya adalah
**34,5% ± 18,4**. Angka kedua inilah yang dipakai sebagai ekspektasi.

**5. Portofolio produk sangat timpang.**
Dari 87 produk, hanya **2 yang masuk kelas AX** (bernilai tinggi dan permintaannya stabil),
sementara **53 masuk kelas CZ** (bernilai kecil dan tidak dapat diprediksi). Tiga kategori
teratas menyumbang 77,4% pendapatan.

---

## Struktur Repositori

```
.
├── README.md
├── requirements.txt                        # Untuk deploy dashboard (Streamlit Cloud)
├── requirements-notebook.txt               # Untuk menjalankan notebook (lokal/Colab)
├── Analisis_Penjualan_Percetakan.ipynb    # Notebook analisis lengkap (15 bagian)
├── app.py                                  # Dashboard Streamlit
└── data/
    └── data_penjualan.csv                  # Dataset (pemisah titik koma)
```

---

## Cara Menjalankan

### Notebook

**Google Colab (disarankan):**
1. Buka [colab.research.google.com](https://colab.research.google.com)
2. `File` → `Upload notebook` → pilih `Analisis_Penjualan_Percetakan.ipynb`
3. Jalankan sel dari atas; sel kedua akan meminta unggahan `data_penjualan.csv`

**Lokal:**
```bash
pip install -r requirements-notebook.txt
jupyter notebook Analisis_Penjualan_Percetakan.ipynb
```

> Catatan: sel instalasi Prophet memakan waktu 2–3 menit, dan grid search SARIMA
> sekitar 3–5 menit. Backtesting rolling-origin juga perlu beberapa menit.

### Dashboard

**Lokal:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501`, lalu unggah file CSV melalui panel sebelah kiri.

**Deploy ke Streamlit Community Cloud (gratis, dapat link publik):**
1. Upload repo ini ke GitHub (public)
2. Buka [share.streamlit.io](https://share.streamlit.io), login dengan akun GitHub
3. **New app** → pilih repo ini → Main file path: `app.py` → **Deploy**
4. Tunggu 3–5 menit, dapat link seperti `namakamu-penjualan.streamlit.app`

---

## Isi Notebook

| Bagian | Isi |
|---|---|
| 1 | Setup & Load Data |
| 2 | Audit Kualitas Data — nilai kosong, konsistensi aritmatika, duplikasi, penamaan produk |
| 3 | Data Cleaning & Feature Engineering — normalisasi nama, ekstraksi kategori & gramatur |
| 4 | Exploratory Data Analysis — tren, Pareto, pola hari, analisis harga |
| 5 | Persiapan Time Series — justifikasi granularitas mingguan |
| 6 | Dekomposisi & Uji Stasioneritas — ADF, ACF/PACF |
| 7 | Modeling — 9 model dibandingkan |
| 8 | Perbandingan Model & Prediksi Akhir |
| 9 | Backtesting Rolling-Origin |
| 10 | Analisis ABC-XYZ |
| 11 | Efek Kalender — Ramadan, Lebaran, hari libur nasional |
| 12 | Deteksi Anomali |
| 13 | Rekonsiliasi Hierarkis |
| 14 | Export Hasil ke Excel |
| 15 | Kesimpulan Bisnis & Keterbatasan |

---

## Model yang Dibandingkan

Baseline sederhana disertakan secara sengaja. Model kompleks yang tidak mengalahkan
tebakan "pakai nilai minggu lalu" tidak layak dipakai di produksi.

| Model | Jenis | MAPE (single split) |
|---|---|---|
| Simple Exponential Smoothing | Statistik | 19,2% |
| Ensemble (3 model) | Gabungan | 20,1% |
| Prophet | Statistik | 24,6% |
| XGBoost | Machine learning | 25,1% |
| SARIMA(0,1,2)x(1,0,1,4) | Statistik | 26,9% |
| Random Forest | Machine learning | 27,4% |
| Naive | *Baseline* | 35,3% |
| Moving Average (4) | *Baseline* | 36,2% |
| Seasonal Naive | *Baseline* | 71,1% |

Hasil **backtesting rolling-origin** (6 fold, horizon 4 minggu) — angka yang lebih
dapat dipercaya:

| Model | MAPE rata-rata | Simpangan baku |
|---|---|---|
| SARIMA | 34,5% | 18,4 |
| Holt-Winters | 36,8% | 13,2 |
| Moving Average (4) | 37,7% | 20,9 |
| Naive | 51,4% | 38,0 |

Perhatikan kolom simpangan baku: Holt-Winters rata-ratanya sedikit lebih buruk dari
SARIMA, tetapi jauh lebih **konsisten**. Untuk perencanaan bisnis, stabilitas sering
lebih berharga daripada rata-rata yang sedikit lebih baik.

---

## Rekomendasi untuk Perusahaan

1. **Perbaiki sistem input produk (prioritas tertinggi).** Gunakan daftar produk baku
   berbentuk dropdown, bukan kolom teks bebas. Tanpa ini, setiap laporan penjualan
   yang dihasilkan berpotensi keliru.

2. **Terapkan strategi berbeda per kelas produk.** Produk AX layak distok otomatis;
   53 produk kelas CZ sebaiknya diproduksi berdasarkan pesanan saja, atau
   dipertimbangkan untuk dihentikan.

3. **Gunakan forecast per kategori untuk perencanaan bahan baku,** dengan batas atas
   interval kepercayaan sebagai dasar *safety stock*.

4. **Rencanakan dengan rentang, bukan angka tunggal.** Volatilitas pendapatan mingguan
   cukup tinggi (CV 0,64).

---

## Keterbatasan

| Keterbatasan | Dampak |
|---|---|
| Data hanya ~15 bulan | Belum cukup untuk memastikan pola musiman tahunan |
| Hanya satu siklus Ramadan-Lebaran | Efek kalender masih berupa hipotesis, belum terverifikasi |
| Tidak ada data pelanggan | Analisis retensi dan segmentasi tidak memungkinkan |
| Tidak ada data biaya/HPP | Analisis terbatas pada pendapatan, bukan profitabilitas |
| Status 40 baris duplikat belum terverifikasi | Perlu konfirmasi apakah order kembar adalah transaksi terpisah |
| MAPE backtesting 30–40% | Wajar untuk data B2B volatil, tetapi bukan presisi tinggi |

Model ini layak dipakai sebagai **alat bantu** perencanaan jangka pendek (4–8 minggu),
bukan sebagai angka mutlak.

---

## Teknologi

`pandas` · `numpy` · `statsmodels` · `scikit-learn` · `xgboost` · `prophet` ·
`matplotlib` · `seaborn` · `plotly` · `streamlit`

---

## Sumber Data

(https://www.kaggle.com/datasets/jabirmuktabir/data-penjualan-produk-cetakan)
