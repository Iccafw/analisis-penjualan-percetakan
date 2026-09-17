"""
Dashboard Analisis & Peramalan Penjualan Percetakan
====================================================
Menjalankan:
    pip install -r requirements.txt
    streamlit run app.py

Dashboard ini memuat ulang logika pembersihan dan peramalan dari notebook
analisis, dikemas agar dapat digunakan oleh pengguna non-teknis.
"""

import warnings
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Konfigurasi halaman
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Penjualan Percetakan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALET = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
         "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

FAMILY = ["DUPLEKS", "IVORY", "CRAFT", "FOODPAK", "GREASEPROOF",
          "HVS", "KINSTRUK", "BOWL", "CUPSTOCK"]

KOREKSI_TYPO = {
    "DUPLEX": "DUPLEKS",
    "GRESEPROOF": "GREASEPROOF",
    "GRESSPROFF": "GREASEPROOF",
    "GRESPROOF": "GREASEPROOF",
    "CRAF270": "CRAFT270",
}


# ──────────────────────────────────────────────────────────────────────
# Fungsi bantu
# ──────────────────────────────────────────────────────────────────────
def format_rupiah(nilai: float) -> str:
    """Format angka ke Rupiah singkat (Rb / Jt / M)."""
    if pd.isna(nilai):
        return "-"
    if abs(nilai) >= 1e9:
        return f"Rp {nilai / 1e9:,.2f} M"
    if abs(nilai) >= 1e6:
        return f"Rp {nilai / 1e6:,.1f} Jt"
    if abs(nilai) >= 1e3:
        return f"Rp {nilai / 1e3:,.0f} Rb"
    return f"Rp {nilai:,.0f}"


def normalisasi_nama(nama: str) -> str:
    """Seragamkan penulisan nama produk."""
    s = re.sub(r"[^A-Z0-9]", "", str(nama).upper())
    for salah, benar in KOREKSI_TYPO.items():
        s = s.replace(salah, benar)
    return s


def ambil_kategori(nama: str) -> str:
    for f in FAMILY:
        if nama.startswith(f):
            return f.title()
    return "Lainnya"


def ambil_gramatur(nama: str) -> float:
    m = re.search(r"(\d{3})", nama)
    if m:
        g = int(m.group(1))
        return g if 100 <= g <= 900 else np.nan
    return np.nan


@st.cache_data(show_spinner=False)
def muat_dan_bersihkan(sumber) -> pd.DataFrame:
    """Baca CSV, bersihkan, dan tambahkan kolom turunan."""
    df = pd.read_csv(sumber, sep=";")

    wajib = {"Tanggal", "Jenis Produk", "Jumlah Order", "Harga", "Total"}
    kurang = wajib - set(df.columns)
    if kurang:
        raise ValueError(f"Kolom berikut tidak ditemukan: {', '.join(sorted(kurang))}")

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Tanggal"])

    for kolom in ["Jumlah Order", "Harga", "Total"]:
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce")
    df = df.dropna(subset=["Jumlah Order", "Harga", "Total"])

    df["produk_bersih"] = df["Jenis Produk"].apply(normalisasi_nama)
    df["Kategori"] = df["produk_bersih"].apply(ambil_kategori)
    df["Gramatur"] = df["produk_bersih"].apply(ambil_gramatur)
    df["Bulan"] = df["Tanggal"].dt.to_period("M").dt.to_timestamp()
    df["Hari"] = df["Tanggal"].dt.day_name()

    return df.sort_values("Tanggal").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def buat_time_series(df: pd.DataFrame, kategori: str = "Semua") -> pd.Series:
    """Agregasi ke deret mingguan, hari tanpa transaksi diisi nol."""
    sub = df if kategori == "Semua" else df[df["Kategori"] == kategori]
    if sub.empty:
        return pd.Series(dtype=float)

    harian = sub.groupby("Tanggal")["Total"].sum()
    rentang = pd.date_range(df["Tanggal"].min(), df["Tanggal"].max(), freq="D")
    harian = harian.reindex(rentang, fill_value=0)

    mingguan = harian.resample("W").sum()
    return mingguan.iloc[1:-1] if len(mingguan) > 2 else mingguan


@st.cache_data(show_spinner=False)
def jalankan_forecast(ts: pd.Series, horizon: int, metode: str):
    """Latih model pada seluruh deret dan ramalkan ke depan."""
    if len(ts) < 12:
        return None, None

    if metode == "SARIMA":
        model = SARIMAX(
            ts, order=(0, 1, 2), seasonal_order=(1, 0, 1, 4),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        hasil = model.get_forecast(steps=horizon)
        titik = hasil.predicted_mean.clip(lower=0)
        interval = hasil.conf_int(alpha=0.20)
        interval.columns = ["bawah", "atas"]
        interval = interval.clip(lower=0)

    elif metode == "Holt-Winters":
        model = ExponentialSmoothing(
            ts, trend="add", seasonal="add", seasonal_periods=4
        ).fit()
        titik = model.forecast(horizon).clip(lower=0)
        galat = np.std(model.resid)
        interval = pd.DataFrame(
            {"bawah": (titik - 1.28 * galat).clip(lower=0),
             "atas": titik + 1.28 * galat},
            index=titik.index,
        )

    else:  # Moving Average
        nilai = ts.iloc[-4:].mean()
        idx = pd.date_range(ts.index[-1] + pd.Timedelta(weeks=1),
                            periods=horizon, freq="W")
        titik = pd.Series(nilai, index=idx)
        galat = ts.iloc[-8:].std()
        interval = pd.DataFrame(
            {"bawah": (titik - 1.28 * galat).clip(lower=0),
             "atas": titik + 1.28 * galat},
            index=idx,
        )

    return titik, interval


@st.cache_data(show_spinner=False)
def klasifikasi_abc_xyz(df: pd.DataFrame) -> pd.DataFrame:
    """Klasifikasi produk berdasarkan nilai (ABC) dan prediktabilitas (XYZ)."""
    matriks = df.pivot_table(
        index="produk_bersih", columns=df["Tanggal"].dt.to_period("M"),
        values="Total", aggfunc="sum",
    ).fillna(0)

    pendapatan = matriks.sum(axis=1).sort_values(ascending=False)
    kumulatif = pendapatan.cumsum() / pendapatan.sum()
    cv = (matriks.std(axis=1) / matriks.mean(axis=1).replace(0, np.nan)).fillna(99)

    hasil = pd.DataFrame({
        "Produk": pendapatan.index,
        "Pendapatan": pendapatan.values,
        "CV": cv.reindex(pendapatan.index).values,
        "ABC": pd.cut(kumulatif, [0, 0.80, 0.95, 1.01], labels=["A", "B", "C"]).values,
        "XYZ": pd.cut(cv.reindex(pendapatan.index), [-0.01, 0.5, 1.0, 999],
                      labels=["X", "Y", "Z"]).values,
    })
    hasil["Kelas"] = hasil["ABC"].astype(str) + hasil["XYZ"].astype(str)
    return hasil


# ──────────────────────────────────────────────────────────────────────
# Sidebar: input data
# ──────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Pengaturan")

berkas = st.sidebar.file_uploader(
    "Unggah data penjualan (.csv)",
    type=["csv"],
    help="File CSV dengan pemisah titik koma (;) dan kolom: "
         "Tanggal, Jenis Produk, Jumlah Order, Harga, Total",
)

if berkas is None:
    st.title("📊 Dashboard Analisis & Peramalan Penjualan")
    st.markdown(
        """
        Selamat datang. Dashboard ini menganalisis data penjualan harian perusahaan
        percetakan dan menghasilkan peramalan pendapatan untuk perencanaan jangka pendek.

        **Untuk memulai, unggah file CSV melalui panel di sebelah kiri.**

        ---

        #### Format data yang dibutuhkan

        File CSV dengan pemisah **titik koma (`;`)** dan kolom berikut:

        | Kolom | Keterangan |
        |---|---|
        | `Tanggal` | Tanggal transaksi, format `DD/MM/YYYY` |
        | `Jenis Produk` | Nama produk |
        | `Jumlah Order` | Jumlah unit terjual |
        | `Harga` | Harga satuan |
        | `Total` | Nilai transaksi |

        #### Yang tersedia di dashboard ini

        - **Ringkasan** — indikator utama dan tren pendapatan
        - **Analisis Produk** — kontribusi kategori, Pareto, klasifikasi ABC-XYZ
        - **Peramalan** — proyeksi pendapatan mingguan dengan interval kepercayaan
        - **Kualitas Data** — audit konsistensi dan konsolidasi nama produk
        """
    )
    st.stop()

# Muat data
try:
    df = muat_dan_bersihkan(berkas)
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()

if df.empty:
    st.error("Tidak ada baris data yang valid setelah pembersihan.")
    st.stop()

# Filter
st.sidebar.subheader("Filter Data")

tgl_min, tgl_maks = df["Tanggal"].min().date(), df["Tanggal"].max().date()
rentang = st.sidebar.date_input(
    "Rentang tanggal", value=(tgl_min, tgl_maks),
    min_value=tgl_min, max_value=tgl_maks,
)

daftar_kategori = ["Semua"] + sorted(df["Kategori"].unique().tolist())
pilih_kategori = st.sidebar.multiselect(
    "Kategori produk", options=daftar_kategori, default=["Semua"],
)

df_filter = df.copy()
if isinstance(rentang, tuple) and len(rentang) == 2:
    df_filter = df_filter[
        (df_filter["Tanggal"].dt.date >= rentang[0])
        & (df_filter["Tanggal"].dt.date <= rentang[1])
    ]
if pilih_kategori and "Semua" not in pilih_kategori:
    df_filter = df_filter[df_filter["Kategori"].isin(pilih_kategori)]

if df_filter.empty:
    st.warning("Tidak ada data yang cocok dengan filter. Silakan longgarkan filter.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Pengaturan Peramalan")
horizon = st.sidebar.slider("Horizon (minggu ke depan)", 4, 16, 8)
metode = st.sidebar.selectbox("Metode", ["SARIMA", "Holt-Winters", "Moving Average"])
kategori_forecast = st.sidebar.selectbox(
    "Ramalkan untuk", ["Semua"] + sorted(df["Kategori"].unique().tolist()),
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Peramalan bersifat indikatif untuk perencanaan jangka pendek, "
    "bukan angka pasti. Selalu perhatikan interval kepercayaannya."
)

# ──────────────────────────────────────────────────────────────────────
# Halaman utama
# ──────────────────────────────────────────────────────────────────────
st.title("📊 Dashboard Analisis & Peramalan Penjualan")
st.caption(
    f"Periode data: {df_filter['Tanggal'].min():%d %b %Y} – "
    f"{df_filter['Tanggal'].max():%d %b %Y}  •  "
    f"{len(df_filter):,} transaksi"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Ringkasan", "📦 Analisis Produk", "🔮 Peramalan", "🧹 Kualitas Data"]
)

# ── Tab 1: Ringkasan ──────────────────────────────────────────────────
with tab1:
    total_pendapatan = df_filter["Total"].sum()
    total_unit = df_filter["Jumlah Order"].sum()
    n_hari = df_filter["Tanggal"].nunique()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Pendapatan", format_rupiah(total_pendapatan))
    k2.metric("Total Unit Terjual", f"{total_unit:,.0f}")
    k3.metric("Jumlah Transaksi", f"{len(df_filter):,}")
    k4.metric("Rata-rata per Transaksi", format_rupiah(df_filter["Total"].mean()))

    st.markdown("### Tren Pendapatan Bulanan")
    bulanan = df_filter.groupby("Bulan").agg(
        Pendapatan=("Total", "sum"), Transaksi=("Total", "size")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bulanan["Bulan"], y=bulanan["Pendapatan"],
        mode="lines+markers", name="Pendapatan",
        line=dict(color=PALET[0], width=3), fill="tozeroy",
        fillcolor="rgba(76,114,176,0.15)",
        hovertemplate="%{x|%b %Y}<br>Rp %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=380, hovermode="x unified",
        yaxis_title="Pendapatan (Rp)", xaxis_title=None,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    kiri, kanan = st.columns(2)

    with kiri:
        st.markdown("### Pendapatan per Kategori")
        per_kat = (df_filter.groupby("Kategori")["Total"].sum()
                   .sort_values(ascending=True).reset_index())
        fig = px.bar(per_kat, x="Total", y="Kategori", orientation="h",
                     color_discrete_sequence=[PALET[0]])
        fig.update_layout(height=340, xaxis_title="Pendapatan (Rp)",
                          yaxis_title=None, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with kanan:
        st.markdown("### Pola Hari dalam Seminggu")
        urutan = ["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"]
        label = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
        per_hari = (df_filter.groupby("Hari")["Total"].sum()
                    .reindex(urutan).fillna(0).reset_index())
        per_hari["Hari"] = label

        fig = px.bar(per_hari, x="Hari", y="Total",
                     color_discrete_sequence=[PALET[1]])
        fig.update_layout(height=340, yaxis_title="Pendapatan (Rp)",
                          xaxis_title=None, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Analisis Produk ────────────────────────────────────────────
with tab2:
    st.markdown("### Analisis Pareto")
    st.caption("Mengidentifikasi produk yang menyumbang mayoritas pendapatan.")

    pareto = (df_filter.groupby("produk_bersih")["Total"].sum()
              .sort_values(ascending=False).reset_index())
    pareto["Kumulatif %"] = pareto["Total"].cumsum() / pareto["Total"].sum() * 100
    n_80 = int((pareto["Kumulatif %"] <= 80).sum()) + 1

    st.info(
        f"**{n_80} dari {len(pareto)} produk** "
        f"({n_80 / len(pareto) * 100:.0f}%) menyumbang 80% total pendapatan."
    )

    top = pareto.head(20)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top["produk_bersih"], y=top["Total"],
                         name="Pendapatan", marker_color=PALET[0]))
    fig.add_trace(go.Scatter(x=top["produk_bersih"], y=top["Kumulatif %"],
                             name="Kumulatif %", yaxis="y2",
                             line=dict(color=PALET[3], width=3), mode="lines+markers"))
    fig.add_hline(y=80, line_dash="dash", line_color="gray", yref="y2")
    fig.update_layout(
        height=430,
        yaxis=dict(title="Pendapatan (Rp)"),
        yaxis2=dict(title="Kumulatif (%)", overlaying="y", side="right", range=[0, 105]),
        xaxis_tickangle=-45, margin=dict(t=20, b=100),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Klasifikasi ABC-XYZ")
    st.caption(
        "**ABC** mengukur nilai kontribusi pendapatan; "
        "**XYZ** mengukur seberapa stabil dan mudah diramal permintaannya."
    )

    kls = klasifikasi_abc_xyz(df_filter)

    kiri, kanan = st.columns([1, 1.4])

    with kiri:
        silang = pd.crosstab(kls["ABC"], kls["XYZ"])
        st.markdown("**Jumlah produk per kelas**")
        st.dataframe(silang, use_container_width=True)

        n_ax = int((kls["Kelas"] == "AX").sum())
        n_cz = int((kls["Kelas"] == "CZ").sum())
        st.metric("Kelas AX (bernilai tinggi & stabil)", n_ax)
        st.metric("Kelas CZ (bernilai kecil & acak)", n_cz)

    with kanan:
        plot = kls.copy()
        plot["CV_plot"] = plot["CV"].clip(upper=5)
        fig = px.scatter(
            plot, x="CV_plot", y="Pendapatan", color="ABC",
            hover_name="Produk", log_y=True,
            color_discrete_map={"A": PALET[3], "B": PALET[1], "C": "#8C8C8C"},
            labels={"CV_plot": "Koefisien Variasi (makin kanan makin sulit diramal)",
                    "Pendapatan": "Pendapatan (Rp, skala log)"},
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
        fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
        fig.update_layout(height=420, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Lihat strategi pengelolaan per kelas"):
        st.markdown(
            """
            | Kelas | Karakteristik | Strategi |
            |---|---|---|
            | **AX** | Nilai tinggi, permintaan stabil | Stok otomatis, target layanan tertinggi |
            | **AY** | Nilai tinggi, cukup fluktuatif | Pantau ketat, *safety stock* sedang |
            | **AZ** | Nilai tinggi, tidak menentu | Kelola per order, hindari stok besar |
            | **BX/BY** | Nilai menengah | Review berkala, *reorder point* standar |
            | **CZ** | Nilai kecil, tidak menentu | *Make-to-order*, kandidat rasionalisasi |
            """
        )

    st.markdown("**Produk kelas A (penyumbang 80% pendapatan)**")
    tabel_a = kls[kls["ABC"] == "A"][["Produk", "Pendapatan", "CV", "Kelas"]]
    st.dataframe(
        tabel_a.style.format({"Pendapatan": "Rp {:,.0f}", "CV": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )

# ── Tab 3: Peramalan ──────────────────────────────────────────────────
with tab3:
    st.markdown(f"### Peramalan Pendapatan Mingguan — {kategori_forecast}")

    ts = buat_time_series(df, kategori_forecast)

    if len(ts) < 12:
        st.warning(
            "Data historis belum cukup untuk peramalan "
            f"(tersedia {len(ts)} minggu, minimal 12 minggu)."
        )
    else:
        with st.spinner("Melatih model..."):
            titik, interval = jalankan_forecast(ts, horizon, metode)

        if titik is None:
            st.error("Peramalan gagal dijalankan.")
        else:
            k1, k2, k3 = st.columns(3)
            k1.metric(f"Proyeksi Total {horizon} Minggu", format_rupiah(titik.sum()))
            k2.metric("Rata-rata per Minggu", format_rupiah(titik.mean()))
            selisih = (titik.mean() - ts.mean()) / ts.mean() * 100
            k3.metric("vs Rata-rata Historis", f"{selisih:+.1f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts.index, y=ts.values, name="Historis",
                line=dict(color=PALET[0], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=interval.index, y=interval["atas"], name="Batas Atas",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=interval.index, y=interval["bawah"], name="Interval 80%",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(196,78,82,0.18)",
            ))
            fig.add_trace(go.Scatter(
                x=titik.index, y=titik.values, name="Prediksi",
                line=dict(color=PALET[3], width=3, dash="dash"),
                mode="lines+markers",
            ))
            fig.add_vline(x=ts.index[-1], line_dash="dot", line_color="gray")
            fig.update_layout(
                height=450, hovermode="x unified",
                yaxis_title="Pendapatan Mingguan (Rp)",
                margin=dict(t=20, b=20),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Rincian peramalan**")
            tabel = pd.DataFrame({
                "Minggu Berakhir": titik.index.strftime("%d %b %Y"),
                "Prediksi": titik.values,
                "Batas Bawah": interval["bawah"].values,
                "Batas Atas": interval["atas"].values,
            })
            st.dataframe(
                tabel.style.format({
                    "Prediksi": "Rp {:,.0f}",
                    "Batas Bawah": "Rp {:,.0f}",
                    "Batas Atas": "Rp {:,.0f}",
                }),
                use_container_width=True, hide_index=True,
            )

            st.download_button(
                "⬇️ Unduh hasil peramalan (CSV)",
                data=tabel.to_csv(index=False).encode("utf-8"),
                file_name=f"forecast_{kategori_forecast.lower()}_{horizon}minggu.csv",
                mime="text/csv",
            )

            st.warning(
                "**Cara membaca angka ini.** Berdasarkan backtesting, kesalahan "
                "peramalan rata-rata berkisar 30–40%. Gunakan rentang antara batas "
                "bawah dan batas atas sebagai dasar perencanaan, bukan angka "
                "prediksi tunggal."
            )

# ── Tab 4: Kualitas Data ──────────────────────────────────────────────
with tab4:
    st.markdown("### Audit Kualitas Data")

    k1, k2, k3 = st.columns(3)

    hitung_ulang = df_filter["Jumlah Order"] * df_filter["Harga"]
    n_tidak_cocok = int((hitung_ulang != df_filter["Total"]).sum())
    k1.metric("Baris tidak konsisten", n_tidak_cocok,
              help="Baris di mana Total ≠ Jumlah Order × Harga")

    n_duplikat = int(df_filter.duplicated().sum())
    k2.metric("Baris duplikat persis", n_duplikat,
              help="Pada data transaksi, order kembar bisa jadi sah")

    k3.metric("Nilai kosong", int(df_filter.isna().sum().sum()))

    st.markdown("---")
    st.markdown("### Konsolidasi Nama Produk")

    mentah = df_filter["Jenis Produk"].nunique()
    bersih = df_filter["produk_bersih"].nunique()
    kategori = df_filter["Kategori"].nunique()

    k1, k2, k3 = st.columns(3)
    k1.metric("Nama produk mentah", mentah)
    k2.metric("Setelah normalisasi", bersih, delta=f"{bersih - mentah}")
    k3.metric("Kategori akhir", kategori, delta=f"{kategori - mentah}")

    st.info(
        "Variasi penulisan seperti `Dupleks310` vs `Duplex310`, atau "
        "`GreaseProof40` vs `Greseproof40` vs `Gressproff`, membuat laporan "
        "penjualan menjadi keliru karena penjualan satu produk terpecah ke "
        "beberapa nama. **Solusi jangka panjang: gunakan daftar produk baku "
        "(dropdown) pada sistem input, bukan kolom teks bebas.**"
    )

    with st.expander("Lihat pemetaan nama mentah → kategori"):
        pemetaan = (df_filter.groupby(["Kategori", "Jenis Produk"])
                    .agg(Transaksi=("Total", "size"), Pendapatan=("Total", "sum"))
                    .reset_index().sort_values(["Kategori", "Pendapatan"],
                                               ascending=[True, False]))
        st.dataframe(
            pemetaan.style.format({"Pendapatan": "Rp {:,.0f}"}),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.markdown("### Cakupan Hari Transaksi")

    rentang_penuh = pd.date_range(df_filter["Tanggal"].min(),
                                  df_filter["Tanggal"].max(), freq="D")
    hari_ada = df_filter["Tanggal"].nunique()
    hari_kosong = len(rentang_penuh) - hari_ada

    k1, k2, k3 = st.columns(3)
    k1.metric("Hari dalam kalender", len(rentang_penuh))
    k2.metric("Hari ada transaksi", hari_ada)
    k3.metric("Hari tanpa transaksi", hari_kosong,
              delta=f"{hari_kosong / len(rentang_penuh) * 100:.0f}% dari periode",
              delta_color="off")

    st.caption(
        "Proporsi hari tanpa transaksi yang tinggi adalah alasan peramalan "
        "dilakukan pada level **mingguan**, bukan harian — agregasi mingguan "
        "jauh menurunkan noise tanpa menghilangkan informasi tren."
    )

    st.markdown("---")
    st.markdown("### Data Bersih")
    st.dataframe(
        df_filter[["Tanggal", "Jenis Produk", "Kategori", "Gramatur",
                   "Jumlah Order", "Harga", "Total"]].head(200),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇️ Unduh data bersih (CSV)",
        data=df_filter.to_csv(index=False).encode("utf-8"),
        file_name="data_penjualan_bersih.csv",
        mime="text/csv",
    )
