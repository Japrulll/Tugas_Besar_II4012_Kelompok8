import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
 
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Player Scouting – EA FC",
    page_icon="⚽",
    layout="wide",
)
 
# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0;
        letter-spacing: 1px;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #64748b;
        margin-top: 2px;
    }
    .stAlert { border-radius: 10px; }
    .metric-card {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 6px 0;
        border-left: 4px solid #3b82f6;
    }
    .player-card {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 6px 0;
        transition: box-shadow 0.2s;
    }
    .player-card:hover {
        box-shadow: 0 4px 12px rgba(59,130,246,0.15);
    }
    .badge {
        display: inline-block;
        background: #eff6ff;
        color: #2563eb;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.76rem;
        margin-right: 4px;
        border: 1px solid #bfdbfe;
    }
    .player-info-detail {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 14px;
        margin: 10px 0;
        border-radius: 8px;
    }
    .slot-empty {
        background: #f1f5f9;
        border: 2px dashed #cbd5e1;
        border-radius: 8px;
        padding: 6px 10px;
        color: #94a3b8;
        font-size: 0.82rem;
        margin: 3px 0;
    }
    .slot-filled {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 0.82rem;
        margin: 3px 0;
    }
    hr { border-color: #e2e8f0; }
 
    /* Field styles */
    .field-wrapper {
        background: linear-gradient(180deg, #166534 0%, #15803d 40%, #16a34a 60%, #15803d 100%);
        border-radius: 16px;
        padding: 20px;
        position: relative;
        border: 3px solid #14532d;
        box-shadow: inset 0 0 60px rgba(0,0,0,0.3), 0 8px 32px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)
 
# ─── Formations ──────────────────────────────────────────────────────────────────
FORMATIONS = {
    "4-3-3": [
        {"slot": "GK",   "label": "Kiper",              "row": 5},
        {"slot": "RB",   "label": "Bek Kanan",           "row": 4},
        {"slot": "CB1",  "label": "Bek Tengah",          "row": 4},
        {"slot": "CB2",  "label": "Bek Tengah",          "row": 4},
        {"slot": "LB",   "label": "Bek Kiri",            "row": 4},
        {"slot": "CM1",  "label": "Gelandang",           "row": 3},
        {"slot": "CM2",  "label": "Gelandang",           "row": 3},
        {"slot": "CM3",  "label": "Gelandang",           "row": 3},
        {"slot": "RW",   "label": "Sayap Kanan",         "row": 2},
        {"slot": "ST",   "label": "Striker",             "row": 2},
        {"slot": "LW",   "label": "Sayap Kiri",          "row": 2},
    ],
    "4-4-2": [
        {"slot": "GK",   "label": "Kiper",              "row": 5},
        {"slot": "RB",   "label": "Bek Kanan",           "row": 4},
        {"slot": "CB1",  "label": "Bek Tengah",          "row": 4},
        {"slot": "CB2",  "label": "Bek Tengah",          "row": 4},
        {"slot": "LB",   "label": "Bek Kiri",            "row": 4},
        {"slot": "RM",   "label": "Gelandang Kanan",     "row": 3},
        {"slot": "CM1",  "label": "Gelandang",           "row": 3},
        {"slot": "CM2",  "label": "Gelandang",           "row": 3},
        {"slot": "LM",   "label": "Gelandang Kiri",      "row": 3},
        {"slot": "ST1",  "label": "Striker",             "row": 2},
        {"slot": "ST2",  "label": "Striker",             "row": 2},
    ],
    "4-2-3-1": [
        {"slot": "GK",   "label": "Kiper",               "row": 5},
        {"slot": "RB",   "label": "Bek Kanan",            "row": 4},
        {"slot": "CB1",  "label": "Bek Tengah",           "row": 4},
        {"slot": "CB2",  "label": "Bek Tengah",           "row": 4},
        {"slot": "LB",   "label": "Bek Kiri",             "row": 4},
        {"slot": "CDM1", "label": "Gelandang Bertahan",   "row": 3},
        {"slot": "CDM2", "label": "Gelandang Bertahan",   "row": 3},
        {"slot": "RAM",  "label": "Gelandang Serang Kanan","row": 2},
        {"slot": "CAM",  "label": "Gelandang Serang",     "row": 2},
        {"slot": "LAM",  "label": "Gelandang Serang Kiri","row": 2},
        {"slot": "ST",   "label": "Striker",              "row": 1},
    ],
    "3-5-2": [
        {"slot": "GK",   "label": "Kiper",               "row": 5},
        {"slot": "CB1",  "label": "Bek Tengah",           "row": 4},
        {"slot": "CB2",  "label": "Bek Tengah",           "row": 4},
        {"slot": "CB3",  "label": "Bek Tengah",           "row": 4},
        {"slot": "RWB",  "label": "Wing-Back Kanan",      "row": 3},
        {"slot": "CM1",  "label": "Gelandang",            "row": 3},
        {"slot": "CM2",  "label": "Gelandang",            "row": 3},
        {"slot": "CM3",  "label": "Gelandang",            "row": 3},
        {"slot": "LWB",  "label": "Wing-Back Kiri",       "row": 3},
        {"slot": "ST1",  "label": "Striker",              "row": 2},
        {"slot": "ST2",  "label": "Striker",              "row": 2},
    ],
    "5-3-2": [
        {"slot": "GK",   "label": "Kiper",               "row": 5},
        {"slot": "RB",   "label": "Bek Kanan",            "row": 4},
        {"slot": "CB1",  "label": "Bek Tengah",           "row": 4},
        {"slot": "CB2",  "label": "Bek Tengah",           "row": 4},
        {"slot": "CB3",  "label": "Bek Tengah",           "row": 4},
        {"slot": "LB",   "label": "Bek Kiri",             "row": 4},
        {"slot": "CM1",  "label": "Gelandang",            "row": 3},
        {"slot": "CM2",  "label": "Gelandang",            "row": 3},
        {"slot": "CM3",  "label": "Gelandang",            "row": 3},
        {"slot": "ST1",  "label": "Striker",              "row": 2},
        {"slot": "ST2",  "label": "Striker",              "row": 2},
    ],
}
 
SLOT_POSITION_FILTER = {
    "GK":   ["GK", "Goalkeeper"],
    "LB":   ["LB", "Left Back", "Back"],
    "RB":   ["RB", "Right Back", "Back"],
    "CB1":  ["CB", "Centre-Back", "Center Back"],
    "CB2":  ["CB", "Centre-Back", "Center Back"],
    "CB3":  ["CB", "Centre-Back", "Center Back"],
    "LM":   ["LM", "Left Mid"],
    "RM":   ["RM", "Right Mid"],
    "CM1":  ["CM", "Midfielder", "Mid"],
    "CM2":  ["CM", "Midfielder", "Mid"],
    "CM3":  ["CM", "Midfielder", "Mid"],
    "CDM1": ["CDM", "Defensive Mid", "DM"],
    "CDM2": ["CDM", "Defensive Mid", "DM"],
    "LAM":  ["CAM", "AM", "Attacking Mid"],
    "CAM":  ["CAM", "AM", "Attacking Mid"],
    "RAM":  ["CAM", "AM", "Attacking Mid"],
    "LWB":  ["LWB", "Wing-Back", "Back"],
    "RWB":  ["RWB", "Wing-Back", "Back"],
    "LW":   ["LW", "Left Wing", "Winger"],
    "RW":   ["RW", "Right Wing", "Winger"],
    "ST":   ["ST", "CF", "Forward", "Striker", "Centre Forward"],
    "ST1":  ["ST", "CF", "Forward", "Striker"],
    "ST2":  ["ST", "CF", "Forward", "Striker"],
}
 
# ─── Session State ────────────────────────────────────────────────────────────────
for key, default in {
    "df_clean": None,
    "df_num": None,
    "df_cat": None,
    "df_master": None,
    "cosine_final": None,
    "processing_done": False,
    "selected_squad": {},
    "clicked_player": None,
    "w_cat": 0.2,
    "w_num": 0.8,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
 
# ─── Helper Functions ─────────────────────────────────────────────────────────────
 
def preprocess(df_raw):
    df = df_raw.copy()
 
    url_cols_to_drop = [
        c for c in df.columns
        if "url" in c.lower() and c not in ("shieldUrl", "avatarUrl")
    ]
    df = df.drop(columns=url_cols_to_drop, errors="ignore")
    df = df.drop(columns=df.columns[df.columns.str.contains("diff", case=False)], errors="ignore")
    id_cols = [c for c in df.columns if c.lower() == "id" or c.lower().endswith("_id") or "/id" in c.lower()]
    df = df.drop(columns=id_cols, errors="ignore")
    df = df.drop(columns=df.columns[df.columns.str.contains("shortLabel", case=False)], errors="ignore")
    for col in list(df.columns):
        if df[col].nunique() <= 1:
            df.drop(columns=col, inplace=True)
 
    kolom_buang = [
        "stats/pac/value","stats/sho/value","stats/pas/value",
        "stats/dri/value","stats/def/value","stats/phy/value",
        "stats/sprintSpeed/value","stats/interceptions/value",
        "stats/longShots/value","stats/curve/value",
        "rank","birthdate","overallRating","team/isPopular",
    ]
    df = df.drop(columns=[c for c in kolom_buang if c in df.columns], errors="ignore")
 
    kamus = {
        "position/label": "position_name",
        "team/label": "team_name",
        "nationality/label": "nationality",
    }
    df = df.rename(columns={k: v for k, v in kamus.items() if k in df.columns})
    df.columns = (df.columns
                  .str.replace("stats/", "", regex=False)
                  .str.replace("/value", "", regex=False)
                  .str.replace("/", "_", regex=False))
 
    for col in ["firstName", "lastName", "commonName"]:
        if col not in df.columns:
            df[col] = ""
    fullname = (df["firstName"].fillna("") + " " + df["lastName"].fillna("") +
                " '" + df["commonName"].fillna("") + "'")
    fullname = fullname.str.replace(" ''", "", regex=False).str.strip()
    df.insert(0, "fullName", fullname)
    df = df.drop(columns=["firstName","lastName","commonName"], errors="ignore")
 
    col_hi_nan = df.columns[df.isna().mean() > 0.7]
    df = df.drop(columns=col_hi_nan)
    df = df.fillna("-")
    df.insert(0, "id", [f"player{i}" for i in range(len(df))])
    return df
 
 
def build_features(df_clean):
    df_num = df_clean.select_dtypes(include="number").copy()
    df_cat = df_clean.select_dtypes(include="object").copy()
    df_num.insert(0, "id", df_cat["id"])
    df_num.insert(1, "fullName", df_cat["fullName"])
    stand_scaler = StandardScaler()
    rob_scaler   = RobustScaler()
    numeric_cols = df_num.columns[2:]
    for col in numeric_cols:
        if abs(df_num[col].mean() - df_num[col].median()) > 1:
            df_num[col] = rob_scaler.fit_transform(df_num[[col]])
        else:
            df_num[col] = stand_scaler.fit_transform(df_num[[col]])
    return df_num, df_cat
 
 
def build_similarity(df_cat, df_num, w_cat, w_num):
    df_cat = df_cat.copy()
    df_cat["combined"] = df_cat.astype(str).agg(" ".join, axis=1)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df_cat["combined"])
    cosine_cat = cosine_similarity(count_matrix)
    cosine_num = cosine_similarity(df_num.iloc[:, 2:].values)
    return (cosine_cat * w_cat) + (cosine_num * w_num)
 
 
def run_processing(df_master, w_cat, w_num):
    df_clean     = preprocess(df_master)
    df_num, df_cat = build_features(df_clean)
    cosine_final = build_similarity(df_cat, df_num, w_cat, w_num)
    return df_clean, df_num, df_cat, cosine_final
 
 
def get_recommendations(nama, df_clean, cosine_final, top_n=10):
    hasil = df_clean[df_clean["fullName"].str.contains(nama, case=False, na=False)]
    if len(hasil) == 0:
        return None, None
    idx = hasil.index[0]
    player_info = hasil.iloc[0]
    sim_scores = sorted(enumerate(cosine_final[idx]), key=lambda x: x[1], reverse=True)
    top_idx = [i[0] for i in sim_scores[1:top_n + 1]]
    cols = ["fullName","position_name","team_name","nationality"]
    cols = [c for c in cols if c in df_clean.columns]
    return player_info, df_clean[cols].iloc[top_idx].reset_index(drop=True)
 
 
def render_shield(shield, width=70):
    if shield and str(shield) not in ["-","nan","","None","nan"]:
        return f"<img src='{shield}' width='{width}' style='border-radius:6px; display:block; margin:0 auto'/>"
    return f"<div style='font-size:{width//3}px;text-align:center'>🙈</div>"
 
 
# ─── Header ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">⚽ AI Player Scouting – EA FC</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Rekomendasi Pemain Pengganti menggunakan Cosine Similarity</p>', unsafe_allow_html=True)
st.markdown("---")
 
# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Dataset")
 
    # ── Auto-detect local CSV files ──
    DATA_DIR = "data"  # Letakkan CSV di folder ./data/
    local_csvs = []
    if os.path.isdir(DATA_DIR):
        local_csvs = sorted([
            os.path.join(DATA_DIR, f)
            for f in os.listdir(DATA_DIR)
            if f.endswith(".csv")
        ])
 
    dataset_source = st.radio(
        "Sumber Dataset",
        ["📁 File Lokal (./data/)", "⬆️ Upload Manual"],
        index=0,
    )
 
    uploaded_files = []
    chosen_locals  = []
 
    if dataset_source == "📁 File Lokal (./data/)":
        if local_csvs:
            chosen_locals = st.multiselect(
                "Pilih file CSV",
                local_csvs,
                default=local_csvs,
                format_func=os.path.basename,
            )
        else:
            st.warning("Tidak ada CSV di folder `./data/`. Letakkan file CSV di sana.")
    else:
        uploaded_files = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            accept_multiple_files=True,
        )
 
    st.markdown("---")
    st.header("⚙️ Parameter Model")
    w_cat = st.slider("Bobot Kategorik (w_cat)", 0.0, 1.0, 0.2, 0.05)
    w_num = round(1.0 - w_cat, 2)
    st.write(f"Bobot Numerik (w_num): **{w_num}**")
    top_n = st.slider("Jumlah Rekomendasi (Top-N)", 3, 30, 10)
 
    run_btn = st.button("🚀 Proses Data", use_container_width=True, type="primary")
 
    # ── Auto-load on first visit if local CSVs exist ──
    if (not st.session_state.processing_done
            and local_csvs
            and dataset_source == "📁 File Lokal (./data/)"
            and not run_btn):
        with st.spinner("Auto-loading dataset lokal…"):
            try:
                dfs = [pd.read_csv(f) for f in local_csvs]
                df_master = pd.concat(dfs, ignore_index=True)
                df_clean, df_num, df_cat, cosine_final = run_processing(df_master, w_cat, w_num)
                st.session_state.update({
                    "df_master": df_master,
                    "df_clean": df_clean,
                    "df_num": df_num,
                    "df_cat": df_cat,
                    "cosine_final": cosine_final,
                    "processing_done": True,
                    "w_cat": w_cat,
                    "w_num": w_num,
                })
                st.sidebar.success("✅ Dataset lokal berhasil dimuat!")
            except Exception as e:
                st.sidebar.error(f"❌ Gagal auto-load: {e}")
 
# ─── Manual process button ────────────────────────────────────────────────────────
if run_btn:
    files_to_use = chosen_locals if dataset_source == "📁 File Lokal (./data/)" else uploaded_files
    if not files_to_use:
        st.sidebar.error("⚠️ Pilih / upload minimal 1 file CSV.")
    else:
        with st.spinner("Memproses data…"):
            try:
                dfs = [pd.read_csv(f) for f in files_to_use]
                df_master = pd.concat(dfs, ignore_index=True)
                df_clean, df_num, df_cat, cosine_final = run_processing(df_master, w_cat, w_num)
                st.session_state.update({
                    "df_master": df_master,
                    "df_clean": df_clean,
                    "df_num": df_num,
                    "df_cat": df_cat,
                    "cosine_final": cosine_final,
                    "processing_done": True,
                    "w_cat": w_cat,
                    "w_num": w_num,
                })
                st.sidebar.success("✅ Data berhasil diproses!")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")
                st.exception(e)
 
# ─── Main Content ─────────────────────────────────────────────────────────────────
if not st.session_state.processing_done:
    st.info("👈 Letakkan CSV di folder `./data/` agar langsung ter-load otomatis, atau upload manual via sidebar.")
    st.markdown("""
    ### Cara Penggunaan
    1. Letakkan file `ea-sports-fc-*.csv` di folder `./data/` → langsung auto-load
    2. Atau upload manual via sidebar
    3. Atur bobot fitur & klik **Proses Data**
    4. Bangun squad di **Squad Builder** atau cari rekomendasi di tab **Rekomendasi**
    """)
else:
    df_clean     = st.session_state.df_clean
    df_master    = st.session_state.df_master
    df_num       = st.session_state.df_num
    df_cat       = st.session_state.df_cat
    cosine_final = st.session_state.cosine_final
 
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏟️ Squad Builder", "🔍 Rekomendasi", "📊 EDA", "🗃️ Dataset", "ℹ️ Info Model"
    ])
 
    # ═══════════════════════════════════════════════════════════
    # TAB 1 — SQUAD BUILDER
    # ═══════════════════════════════════════════════════════════
    with tab1:
        st.subheader("🏟️ Squad Builder")
 
        col_formation, col_reset = st.columns([3, 1])
        with col_formation:
            selected_formation = st.selectbox(
                "Pilih Formasi", list(FORMATIONS.keys()), key="formation_select"
            )
        with col_reset:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset Squad", use_container_width=True):
                st.session_state.selected_squad   = {}
                st.session_state.clicked_player   = None
                st.rerun()
 
        formation_slots = FORMATIONS[selected_formation]
        filled_count    = len(st.session_state.selected_squad)
        st.markdown("---")
 
        # ── Two-column layout: Cari pemain | Daftar squad ──────────────
        col_left, col_right = st.columns([3, 2])
 
        with col_left:
            st.markdown("**🔎 Cari & Tambah Pemain**")
 
            empty_slots = [s for s in formation_slots if s["slot"] not in st.session_state.selected_squad]
 
            if not empty_slots:
                st.success("✅ Squad sudah penuh! Lihat tampilan di bawah.")
            else:
                slot_options = [f"{s['slot']}  –  {s['label']}" for s in empty_slots]
                target_slot_str = st.selectbox("Isi posisi", slot_options, key="slot_select")
                slot_key = target_slot_str.split("  –  ")[0].strip()
 
                search_term = st.text_input(
                    "Nama pemain", placeholder="Cth: Salah, Haaland, De Bruyne…",
                    key="squad_search"
                )
 
                use_pos_filter = st.checkbox("Filter berdasarkan posisi slot", value=True)
 
                if search_term.strip():
                    if use_pos_filter:
                        keywords = SLOT_POSITION_FILTER.get(slot_key, [])
                        mask = df_clean["position_name"].str.contains(
                            "|".join(keywords), case=False, na=False
                        ) if keywords else pd.Series([True] * len(df_clean))
                        found = df_clean[
                            df_clean["fullName"].str.contains(search_term, case=False, na=False) & mask
                        ].head(10)
                        if len(found) == 0:
                            st.caption("Tidak ada hasil dengan filter posisi, menampilkan semua…")
                            found = df_clean[
                                df_clean["fullName"].str.contains(search_term, case=False, na=False)
                            ].head(10)
                    else:
                        found = df_clean[
                            df_clean["fullName"].str.contains(search_term, case=False, na=False)
                        ].head(10)
 
                    if len(found) > 0:
                        chosen_name = st.selectbox(
                            "Pilih pemain",
                            found["fullName"].values,
                            key="player_select"
                        )
 
                        # Preview mini card
                        prev_row = found[found["fullName"] == chosen_name].iloc[0]
                        prev_shield = prev_row.get("shieldUrl", None)
                        prev_img = render_shield(prev_shield, 55)
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:12px;"
                            f"background:#f8fafc;border-radius:10px;padding:10px 14px;"
                            f"border:1px solid #e2e8f0;margin:8px 0'>"
                            f"  {prev_img}"
                            f"  <div>"
                            f"    <div style='font-weight:600;font-size:0.9rem'>{prev_row['fullName']}</div>"
                            f"    <div style='color:#64748b;font-size:0.78rem'>"
                            f"      {prev_row.get('position_name','-')} · {prev_row.get('team_name','-')}"
                            f"    </div>"
                            f"  </div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
 
                        if st.button("➕ Tambah ke Squad", use_container_width=True, type="primary"):
                            row = found[found["fullName"] == chosen_name].iloc[0]
                            st.session_state.selected_squad[slot_key] = {
                                "name":        row["fullName"],
                                "position":    row.get("position_name", "-"),
                                "team":        row.get("team_name", "-"),
                                "nationality": row.get("nationality", "-"),
                                "index":       found.index[found["fullName"] == chosen_name][0],
                                "shield_url":  row.get("shieldUrl", None),
                            }
                            st.rerun()
                    else:
                        st.warning("Pemain tidak ditemukan.")
 
        with col_right:
            st.markdown(f"**📋 Squad Saat Ini ({filled_count}/11)**")
 
            for slot_info in formation_slots:
                slot  = slot_info["slot"]
                label = slot_info["label"]
                if slot in st.session_state.selected_squad:
                    p = st.session_state.selected_squad[slot]
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.markdown(
                            f"<div class='slot-filled'>"
                            f"  <span style='color:#2563eb;font-weight:600;font-size:0.72rem'>{label}</span><br>"
                            f"  <span style='font-weight:600'>{p['name']}</span> "
                            f"  <span style='color:#64748b;font-size:0.75rem'>{p['team']}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with c2:
                        if st.button("❌", key=f"remove_{slot}", use_container_width=True):
                            del st.session_state.selected_squad[slot]
                            if st.session_state.clicked_player == slot:
                                st.session_state.clicked_player = None
                            st.rerun()
                else:
                    st.markdown(
                        f"<div class='slot-empty'>⬜ {label}</div>",
                        unsafe_allow_html=True,
                    )
 
        st.markdown("---")
 
        # ── Display: squad visual ───────────────────────────────────────
        if filled_count == 11:
            st.markdown("### 👥 Tim Anda")
 
            # Group by row
            rows_dict: dict[int, list] = {}
            for s in formation_slots:
                rows_dict.setdefault(s["row"], []).append(s)
 
            st.markdown(
                "<div style='background:linear-gradient(180deg,#166534,#15803d 40%,#16a34a 65%,#15803d 100%);"
                "border-radius:16px;padding:24px 12px;border:3px solid #14532d;"
                "box-shadow:inset 0 0 60px rgba(0,0,0,.35),0 8px 32px rgba(0,0,0,.2)'>",
                unsafe_allow_html=True,
            )
 
            for row_num in sorted(rows_dict.keys()):
                slots_in_row = rows_dict[row_num]
                n = len(slots_in_row)
                cols = st.columns(n)
                for col, slot_info in zip(cols, slots_in_row):
                    slot   = slot_info["slot"]
                    player = st.session_state.selected_squad.get(slot)
                    if not player:
                        continue
                    with col:
                        shield = player.get("shield_url")
                        img_html = render_shield(shield, 72)
                        st.markdown(
                            f"<div style='display:flex;flex-direction:column;align-items:center;"
                            f"text-align:center;padding:4px'>"
                            f"  {img_html}"
                            f"  <div style='margin-top:5px;background:rgba(0,0,0,.55);"
                            f"border-radius:6px;padding:3px 7px'>"
                            f"    <div style='color:#facc15;font-size:0.62rem;font-weight:700;"
                            f"letter-spacing:.5px;text-transform:uppercase'>{slot_info['label']}</div>"
                            f"    <div style='color:#fff;font-size:0.75rem;font-weight:600;"
                            f"line-height:1.3;max-width:90px;word-break:break-word'>{player['name']}</div>"
                            f"    <div style='color:#94a3b8;font-size:0.65rem'>{player['team']}</div>"
                            f"  </div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        if st.button("🔄 Ganti", key=f"squad_btn_{slot}", use_container_width=True):
                            st.session_state.clicked_player = slot
 
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")
 
            # ── Rekomendasi pengganti ────────────────────────────────────
            if st.session_state.clicked_player is not None:
                clicked_key = st.session_state.clicked_player
                player_info = st.session_state.selected_squad.get(clicked_key)
 
                if player_info:
                    st.markdown(f"### 🎯 Rekomendasi Pengganti untuk **{player_info['name']}**")
 
                    col_photo, col_info = st.columns([1, 3])
                    with col_photo:
                        shield = player_info.get("shield_url")
                        if shield and str(shield) not in ["-","nan","","None"]:
                            st.image(shield, width=130)
                        else:
                            st.markdown("🙈")
                    with col_info:
                        st.markdown(
                            f"<div class='player-info-detail'>"
                            f"<strong>Pemain Asli:</strong> {player_info['name']}<br>"
                            f"<strong>Posisi:</strong> {player_info['position']}<br>"
                            f"<strong>Tim:</strong> {player_info['team']}</div>",
                            unsafe_allow_html=True,
                        )
 
                    player_idx = player_info["index"]
                    sim_scores = sorted(enumerate(cosine_final[player_idx]), key=lambda x: x[1], reverse=True)
                    top_idx    = [i[0] for i in sim_scores[1:11]]
                    rec_cols   = ["fullName","position_name","team_name","nationality"] + \
                                 (["shieldUrl"] if "shieldUrl" in df_clean.columns else [])
                    recs = df_clean.iloc[top_idx][rec_cols].reset_index()
 
                    st.markdown("**Top 10 Rekomendasi Pengganti:**")
                    for i, row in recs.iterrows():
                        rec_shield = row.get("shieldUrl") if "shieldUrl" in row.index else None
                        c_img, c1, c2, c3, c4, c5 = st.columns([0.6, 2.2, 1.2, 1.4, 1, 0.9])
                        with c_img:
                            if rec_shield and str(rec_shield) not in ["-","nan","","None"]:
                                st.image(rec_shield, width=50)
                        with c1:
                            st.write(f"**{i+1}. {row['fullName']}**")
                        with c2:
                            st.caption(f"⚽ {row['position_name']}")
                        with c3:
                            st.caption(f"🏟️ {row['team_name']}")
                        with c4:
                            st.caption(f"🌍 {row['nationality']}")
                        with c5:
                            if st.button("Ganti", key=f"replace_squad_{i}"):
                                st.session_state.selected_squad[clicked_key] = {
                                    "name":        row["fullName"],
                                    "position":    row["position_name"],
                                    "team":        row["team_name"],
                                    "nationality": row["nationality"],
                                    "index":       recs.iloc[i]["index"],
                                    "shield_url":  rec_shield,
                                }
                                st.session_state.clicked_player = None
                                st.success(f"✅ {row['fullName']} menggantikan {player_info['name']}!")
                                st.rerun()
 
                    if st.button("✖ Tutup Rekomendasi"):
                        st.session_state.clicked_player = None
                        st.rerun()
 
        elif filled_count > 0:
            st.info(f"⏳ Pilih {11 - filled_count} pemain lagi untuk melengkapi squad.")
 
    # ═══════════════════════════════════════════════════════════
    # TAB 2 — REKOMENDASI
    # ═══════════════════════════════════════════════════════════
    with tab2:
        st.subheader("🔍 Cari Pemain Pengganti")
 
        col_search, col_n = st.columns([3, 1])
        with col_search:
            nama_input = st.text_input("Nama Pemain", placeholder="Contoh: Casemiro, Mbappe, Salah…")
        with col_n:
            top_n_local = st.number_input("Top-N", min_value=1, max_value=50, value=top_n)
 
        if nama_input.strip():
            player_info, recommendations = get_recommendations(
                nama_input.strip(), df_clean, cosine_final, int(top_n_local)
            )
 
            if recommendations is None:
                st.warning(f"Pemain **'{nama_input}'** tidak ditemukan.")
            else:
                st.markdown("---")
                st.markdown("#### 🎯 Pemain yang Dicari")
 
                ref_match  = df_clean[df_clean["fullName"].str.contains(nama_input.strip(), case=False, na=False)]
                ref_shield = (ref_match.iloc[0].get("shieldUrl")
                              if len(ref_match) > 0 and "shieldUrl" in ref_match.columns
                              else None)
 
                col_ref_img, col_ref_info = st.columns([1, 4])
                with col_ref_img:
                    if ref_shield and str(ref_shield) not in ["-","nan","","None"]:
                        st.image(ref_shield, width=120)
                with col_ref_info:
                    info_cols = st.columns(4)
                    for ci, (field, label) in enumerate([
                        ("fullName","Nama"), ("position_name","Posisi"),
                        ("team_name","Tim"), ("nationality","Kebangsaan")
                    ]):
                        with info_cols[ci]:
                            val = player_info.get(field, "-") if field in player_info.index else "-"
                            st.metric(label, val)
 
                st.markdown("---")
                st.markdown(f"#### 🏆 Top-{int(top_n_local)} Rekomendasi")
 
                rec_cols  = ["fullName","position_name","team_name","nationality"] + \
                            (["shieldUrl"] if "shieldUrl" in df_clean.columns else [])
                hasil_rec = df_clean[rec_cols][df_clean.index.isin(recommendations.index)]
 
                for i, (_, row) in enumerate(hasil_rec.iterrows(), 1):
                    shield = row.get("shieldUrl") if "shieldUrl" in row.index else None
                    col_img, col_text = st.columns([0.7, 5])
                    with col_img:
                        if shield and str(shield) not in ["-","nan","","None"]:
                            st.image(shield, width=60)
                    with col_text:
                        st.markdown(
                            f"<div class='player-card'>"
                            f"<strong>{i}. {row['fullName']}</strong><br>"
                            f"<span class='badge'>⚽ {row['position_name']}</span>"
                            f"<span class='badge'>🏟️ {row['team_name']}</span>"
                            f"<span class='badge'>🌍 {row['nationality']}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
 
                st.markdown("---")
                st.download_button(
                    "⬇️ Download Rekomendasi (CSV)",
                    data=recommendations.to_csv(index=False).encode("utf-8"),
                    file_name=f"rekomendasi_{nama_input.replace(' ','_')}.csv",
                    mime="text/csv",
                )
 
    # ═══════════════════════════════════════════════════════════
    # TAB 3 — EDA
    # ═══════════════════════════════════════════════════════════
    with tab3:
        st.subheader("📊 Exploratory Data Analysis")
 
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("**Distribusi Rating Pemain**")
            if "overallRating" in df_master.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df_master["overallRating"].dropna(), bins=20, kde=True, color="#3b82f6", ax=ax)
                ax.set_xlabel("Overall Rating")
                ax.set_ylabel("Jumlah Pemain")
                ax.set_title("Distribusi Rating")
                st.pyplot(fig); plt.close()
            else:
                st.info("Kolom `overallRating` tidak ditemukan.")
 
        with col_e2:
            st.markdown("**Jumlah Pemain Berdasarkan Posisi**")
            pos_col_use = next(
                (c for c in ["position_name"] + list(df_master.columns)
                 if "position" in (c or "").lower() and c in df_clean.columns),
                None
            )
            if pos_col_use:
                order_pos = df_clean[pos_col_use].value_counts().index
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.countplot(data=df_clean, x=pos_col_use, order=order_pos[:15], palette="viridis", ax=ax2)
                ax2.set_xlabel("Posisi"); ax2.set_ylabel("Jumlah")
                ax2.set_title("Pemain per Posisi (Top 15)")
                plt.xticks(rotation=45, ha="right"); plt.tight_layout()
                st.pyplot(fig2); plt.close()
            else:
                st.info("Kolom posisi tidak ditemukan.")
 
        st.markdown("---")
        st.markdown("**Heatmap Korelasi Fitur Numerik**")
        numeric_df = df_clean.select_dtypes(include="number")
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig3, ax3 = plt.subplots(figsize=(14, 10))
            sns.heatmap(corr, vmin=-1, vmax=1, linewidths=0.3, ax=ax3, cmap="coolwarm")
            ax3.set_title("Korelasi In-Game Stats")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(fontsize=8); plt.tight_layout()
            st.pyplot(fig3); plt.close()
 
    # ═══════════════════════════════════════════════════════════
    # TAB 4 — DATASET
    # ═══════════════════════════════════════════════════════════
    with tab4:
        st.subheader("🗃️ Preview Dataset (Setelah Cleaning)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Pemain",   f"{len(df_clean):,}")
        c2.metric("Total Fitur",    f"{len(df_clean.columns)}")
        c3.metric("Fitur Numerik",  f"{len(df_clean.select_dtypes('number').columns)}")
        st.dataframe(df_clean.head(100), use_container_width=True)
        st.markdown("**Daftar Kolom:**")
        st.write(list(df_clean.columns))
        st.download_button(
            "⬇️ Download Dataset Bersih (CSV)",
            data=df_clean.to_csv(index=False).encode("utf-8"),
            file_name="players_clean.csv",
            mime="text/csv",
        )
 

    # ─── Tab 5: Info Model ────────────────────────────────────────────────
    with tab5:
        st.subheader("ℹ️ Informasi Model")

        st.markdown("""
        ### Metodologi
        Sistem ini menggunakan **Cosine Similarity** berbasis dua jenis fitur:
        
        | Jenis Fitur | Teknik | Keterangan |
        |---|---|---|
        | **Kategorik** | CountVectorizer | Posisi, tim, kebangsaan, dll. |
        | **Numerik** | StandardScaler / RobustScaler + Cosine Similarity | Statistik in-game (pace, shoot, pass, dll.) |

        ### Formula Skor Akhir
        ```
        cosine_final = (w_cat × cosine_cat) + (w_num × cosine_num)
        ```

        ### Pipeline Pre-processing
        1. Hapus kolom URL, ID, diff, shortLabel
        2. Hapus kolom dengan ≤1 nilai unik
        3. Hapus kolom redundan (pac, sho, pas, dll.)
        4. Rename kolom agar lebih rapi
        5. Gabungkan firstName + lastName → fullName
        6. Drop kolom dengan >70% nilai kosong
        7. Isi nilai kosong dengan "-"
        8. Scaling fitur numerik (Standard / Robust tergantung distribusi)
        """)

        st.markdown("---")
        st.markdown("**Bobot yang Digunakan Saat Ini:**")
        col_w1, col_w2 = st.columns(2)
        col_w1.metric("w_cat (Kategorik)", f"{w_cat:.2f}")
        col_w2.metric("w_num (Numerik)", f"{w_num:.2f}")