import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import io

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
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        margin-top: 0;
    }
    .stAlert { border-radius: 10px; }
    .metric-card {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 6px 0;
        border-left: 4px solid #4361ee;
    }
    .player-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
    }
    .badge {
        display: inline-block;
        background: #4361ee;
        color: #fff;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin-right: 4px;
    }
    hr { border-color: #eee; }
    .squad-container {
        background: linear-gradient(135deg, #1e5c20 0%, #2d7a2f 100%);
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        position: relative;
        aspect-ratio: 16 / 10;
    }
    .squad-position {
        text-align: center;
        margin: 10px auto;
    }
    .player-button {
        background: #ff6b6b;
        color: white;
        border: none;
        border-radius: 50%;
        width: 70px;
        height: 70px;
        font-weight: bold;
        cursor: pointer;
        font-size: 10px;
        padding: 5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .player-button:hover {
        background: #ff8787;
        transform: scale(1.1);
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    }
    .player-info-detail {
        background: #f8f9fa;
        border-left: 4px solid #ff6b6b;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">⚽ AI Player Scouting – EA FC</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Rekomendasi Pemain Pengganti menggunakan Cosine Similarity</p>', unsafe_allow_html=True)
st.markdown("---")

# ─── Session State ───────────────────────────────────────────────────────────────
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "df_num" not in st.session_state:
    st.session_state.df_num = None
if "df_cat" not in st.session_state:
    st.session_state.df_cat = None
if "cosine_final" not in st.session_state:
    st.session_state.cosine_final = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False
if "selected_squad" not in st.session_state:
    st.session_state.selected_squad = {}
if "clicked_player" not in st.session_state:
    st.session_state.clicked_player = None

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Dataset")
    st.caption("Upload satu atau lebih file CSV EA FC (misal: `ea-sports-fc-24.csv`)")
    uploaded_files = st.file_uploader(
        "Pilih file CSV",
        type=["csv"],
        accept_multiple_files=True,
    )

    st.markdown("---")
    st.header("⚙️ Parameter Model")
    w_cat = st.slider("Bobot Fitur Kategorik (w_cat)", 0.0, 1.0, 0.2, 0.05)
    w_num = round(1.0 - w_cat, 2)
    st.write(f"Bobot Fitur Numerik (w_num): **{w_num}**")

    top_n = st.slider("Jumlah Rekomendasi (Top-N)", 3, 30, 10)

    run_btn = st.button("🚀 Proses Data", use_container_width=True, type="primary")

# ─── Helper Functions ────────────────────────────────────────────────────────────

def load_and_clean(files):
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    return df


def preprocess(df_master):
    df = df_master.copy()

    # Drop URL, ID, diff columns
    for kw in ["url", "diff"]:
        df = df.drop(columns=df.columns[df.columns.str.contains(kw, case=False)], errors="ignore")
    # Drop id columns (careful: only columns whose name IS 'id' or ends with id variants)
    id_cols = [c for c in df.columns if c.lower() == "id" or c.lower().endswith("_id") or "/id" in c.lower()]
    df = df.drop(columns=id_cols, errors="ignore")
    # Drop shortLabel
    df = df.drop(columns=df.columns[df.columns.str.contains("shortLabel", case=False)], errors="ignore")
    # Drop columns with ≤1 unique values
    for col in list(df.columns):
        if df[col].nunique() <= 1:
            df.drop(columns=col, inplace=True)

    # Drop manually specified redundant columns
    kolom_buang = [
        "stats/pac/value", "stats/sho/value", "stats/pas/value",
        "stats/dri/value", "stats/def/value", "stats/phy/value",
        "stats/sprintSpeed/value", "stats/interceptions/value",
        "stats/longShots/value", "stats/curve/value",
        "rank", "birthdate", "overallRating", "team/isPopular",
    ]
    df = df.drop(columns=[c for c in kolom_buang if c in df.columns], errors="ignore")

    # Rename columns
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

    # Build fullName
    for col in ["firstName", "lastName", "commonName"]:
        if col not in df.columns:
            df[col] = ""
    fullname = (df["firstName"].fillna("") + " " + df["lastName"].fillna("") +
                " '" + df["commonName"].fillna("") + "'")
    fullname = fullname.str.replace(" ''", "", regex=False).str.strip()
    df.insert(0, "fullName", fullname)
    df = df.drop(columns=["firstName", "lastName", "commonName"], errors="ignore")

    # Drop columns with >70% NaN
    col_hi_nan = df.columns[df.isna().mean() > 0.7]
    df = df.drop(columns=col_hi_nan)

    df = df.fillna("-")

    # Add synthetic id
    df.insert(0, "id", [f"player{i}" for i in range(len(df))])

    return df


def build_features(df_clean):
    df_num = df_clean.select_dtypes(include="number").copy()
    df_cat = df_clean.select_dtypes(include="object").copy()

    df_num.insert(0, "id", df_cat["id"])
    df_num.insert(1, "fullName", df_cat["fullName"])

    # Scale numeric features
    stand_scaler = StandardScaler()
    rob_scaler = RobustScaler()
    numeric_cols = df_num.columns[2:]
    for col in numeric_cols:
        if abs(df_num[col].mean() - df_num[col].median()) > 1:
            df_num[col] = rob_scaler.fit_transform(df_num[[col]])
        else:
            df_num[col] = stand_scaler.fit_transform(df_num[[col]])

    return df_num, df_cat


def build_similarity(df_cat, df_num, w_cat, w_num):
    # Categorical similarity
    df_cat = df_cat.copy()
    df_cat["combined"] = df_cat.astype(str).agg(" ".join, axis=1)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df_cat["combined"])
    cosine_cat = cosine_similarity(count_matrix)

    # Numeric similarity
    cosine_num = cosine_similarity(df_num.iloc[:, 2:].values)

    # Final weighted similarity
    cosine_final = (cosine_cat * w_cat) + (cosine_num * w_num)
    return cosine_final


def get_recommendations(nama, df_clean, cosine_final, top_n=10):
    hasil = df_clean[df_clean["fullName"].str.contains(nama, case=False, na=False)]
    if len(hasil) == 0:
        return None, None
    idx = hasil.index[0]
    player_info = hasil.iloc[0]
    sim_scores = sorted(enumerate(cosine_final[idx]), key=lambda x: x[1], reverse=True)
    top_idx = [i[0] for i in sim_scores[1:top_n + 1]]
    cols = ["fullName", "position_name", "team_name", "nationality"]
    cols = [c for c in cols if c in df_clean.columns]
    return player_info, df_clean[cols].iloc[top_idx].reset_index(drop=True)


# ─── Squad Functions ─────────────────────────────────────────────────────────────
FORMATIONS = {
    "4-4-2": {
        "GK": (1, 1),
        "DEF": [(1, 0.25), (1, 0.35), (1, 0.65), (1, 0.75)],
        "MID": [(0.5, 0.2), (0.5, 0.4), (0.5, 0.6), (0.5, 0.8)],
        "FWD": [(0.2, 0.35), (0.2, 0.65)],
    },
    "4-3-3": {
        "GK": (1, 1),
        "DEF": [(1, 0.2), (1, 0.35), (1, 0.65), (1, 0.8)],
        "MID": [(0.5, 0.2), (0.5, 0.5), (0.5, 0.8)],
        "FWD": [(0.2, 0.25), (0.2, 0.5), (0.2, 0.75)],
    },
    "4-2-3-1": {
        "GK": (1, 1),
        "DEF": [(1, 0.2), (1, 0.35), (1, 0.65), (1, 0.8)],
        "MID": [(0.55, 0.35), (0.55, 0.65), (0.3, 0.25), (0.3, 0.5), (0.3, 0.75)],
        "FWD": [(0.1, 0.5)],
    }
}

def display_squad_formation(squad, formation="4-3-3"):
    """Display squad in a football field formation"""
    st.markdown(f"""
    <div class="squad-container">
        <!-- Squad will be rendered using Streamlit columns below -->
    </div>
    """, unsafe_allow_html=True)


def get_top_players_by_position(df_clean, position, n=5):
    """Get top rated players by position"""
    pos_players = df_clean[df_clean['position_name'].str.contains(position, case=False, na=False)]
    return pos_players.head(n)


# ─── Processing ──────────────────────────────────────────────────────────────────
if run_btn:
    if not uploaded_files:
        st.sidebar.error("⚠️ Upload minimal 1 file CSV terlebih dahulu.")
    else:
        with st.spinner("Memproses data… Ini mungkin memakan beberapa detik."):
            try:
                df_master = load_and_clean(uploaded_files)
                df_clean = preprocess(df_master)
                df_num, df_cat = build_features(df_clean)
                cosine_final = build_similarity(df_cat, df_num, w_cat, w_num)

                st.session_state.df_master = df_master
                st.session_state.df_clean = df_clean
                st.session_state.df_num = df_num
                st.session_state.df_cat = df_cat
                st.session_state.cosine_final = cosine_final
                st.session_state.processing_done = True
                st.sidebar.success("✅ Data berhasil diproses!")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")
                st.exception(e)

# ─── Main Content ────────────────────────────────────────────────────────────────
if not st.session_state.processing_done:
    st.info("👈 Upload file CSV di sidebar dan klik **Proses Data** untuk memulai.")
    st.markdown("""
    ### Cara Penggunaan
    1. Upload satu atau lebih file CSV dengan nama pola `ea-sports-fc-*.csv`
    2. Atur bobot fitur kategorik vs numerik
    3. Klik **Proses Data**
    4. Cari pemain dan lihat rekomendasi pengganti
    """)
else:
    df_clean = st.session_state.df_clean
    df_master = st.session_state.df_master
    df_num = st.session_state.df_num
    df_cat = st.session_state.df_cat
    cosine_final = st.session_state.cosine_final

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏟️ Squad Builder", "🔍 Rekomendasi", "📊 EDA", "🗃️ Dataset", "ℹ️ Info Model"])

    # ─── Tab 1: Squad Builder ──────────────────────────────────────────────────
    with tab1:
        st.subheader("🏟️ Squad Builder - Pilih 11 Pemain")
        
        # Formation selection
        col_formation, col_reset = st.columns([3, 1])
        with col_formation:
            selected_formation = st.selectbox("Pilih Formasi", list(FORMATIONS.keys()), key="formation_select")
        with col_reset:
            if st.button("🔄 Reset Squad", use_container_width=True):
                st.session_state.selected_squad = {}
                st.session_state.clicked_player = None
                st.rerun()
        
        st.markdown("---")
        
        # Squad selection area
        col_select1, col_select2 = st.columns(2)
        
        with col_select1:
            st.markdown("**Cari Pemain Untuk Squad**")
            search_term = st.text_input("Ketik nama pemain", placeholder="Cth: Mbappe, Haaland, Salah...")
            
            if search_term.strip():
                found_players = df_clean[df_clean["fullName"].str.contains(search_term, case=False, na=False)].head(10)
                if len(found_players) > 0:
                    selected_player_name = st.selectbox(
                        "Pilih pemain dari hasil pencarian",
                        found_players["fullName"].values,
                        key="player_select"
                    )
                    
                    if st.button("➕ Tambah ke Squad", use_container_width=True):
                        if len(st.session_state.selected_squad) < 11:
                            player_data = found_players[found_players["fullName"] == selected_player_name].iloc[0]
                            squad_key = len(st.session_state.selected_squad) + 1
                            st.session_state.selected_squad[squad_key] = {
                                "name": player_data["fullName"],
                                "position": player_data.get("position_name", "-"),
                                "team": player_data.get("team_name", "-"),
                                "nationality": player_data.get("nationality", "-"),
                                "index": found_players.index[found_players["fullName"] == selected_player_name][0]
                            }
                            st.success(f"✅ {selected_player_name} ditambahkan ke squad!")
                            st.rerun()
                        else:
                            st.error("⚠️ Squad sudah penuh (11 pemain)!")
                else:
                    st.warning("Pemain tidak ditemukan.")
        
        with col_select2:
            st.markdown(f"**Squad Saat Ini ({len(st.session_state.selected_squad)}/11)**")
            if st.session_state.selected_squad:
                for idx, player in st.session_state.selected_squad.items():
                    col_name, col_pos, col_remove = st.columns([2, 1, 0.5])
                    with col_name:
                        st.write(f"{idx}. {player['name']}")
                    with col_pos:
                        st.caption(player['position'])
                    with col_remove:
                        if st.button("❌", key=f"remove_{idx}", use_container_width=True):
                            del st.session_state.selected_squad[idx]
                            # Reorder squad
                            reordered = {}
                            for i, (_, player_data) in enumerate(st.session_state.selected_squad.items(), 1):
                                reordered[i] = player_data
                            st.session_state.selected_squad = reordered
                            st.rerun()
            else:
                st.info("Belum ada pemain di squad.")
        
        st.markdown("---")
        
        # Display Squad if 11 players are selected
        if len(st.session_state.selected_squad) == 11:
            st.markdown("### 👥 Tim Anda (11 Pemain)")
            st.markdown("---")
            
            # Display in grid
            cols = st.columns(4)
            for idx, (key, player) in enumerate(st.session_state.selected_squad.items()):
                with cols[idx % 4]:
                    if st.button(
                        f"{player['name']}\n({player['position']})\n{player['team']}",
                        key=f"squad_player_{key}",
                        use_container_width=True
                    ):
                        st.session_state.clicked_player = key
            
            st.markdown("---")
            
            # Show substitution recommendations if a player is clicked
            if st.session_state.clicked_player is not None:
                clicked_key = st.session_state.clicked_player
                player_info = st.session_state.selected_squad[clicked_key]
                
                st.markdown(f"### 🎯 Rekomendasi Pengganti untuk {player_info['name']}")
                
                # Get the player's index in the dataframe
                player_idx = player_info['index']
                
                # Get top recommendations
                sim_scores = sorted(enumerate(cosine_final[player_idx]), key=lambda x: x[1], reverse=True)
                top_idx = [i[0] for i in sim_scores[1:11]]  # Get top 10 recommendations
                
                recommendations = df_clean.iloc[top_idx][["fullName", "position_name", "team_name", "nationality"]]
                
                st.markdown(f"""
                <div class="player-info-detail">
                    <strong>Pemain Asli:</strong> {player_info['name']}<br>
                    <strong>Posisi:</strong> {player_info['position']}<br>
                    <strong>Tim:</strong> {player_info['team']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Top 10 Rekomendasi Pengganti:**")
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    col1, col2, col3, col4 = st.columns([2, 1.2, 1.2, 1])
                    with col1:
                        st.write(f"**{i}. {row['fullName']}**")
                    with col2:
                        st.caption(f"⚽ {row['position_name']}")
                    with col3:
                        st.caption(f"🏟️ {row['team_name']}")
                    with col4:
                        st.caption(f"🌍 {row['nationality']}")
                    
                    if st.button("Ganti Pemain", key=f"replace_{i}"):
                        st.session_state.selected_squad[clicked_key] = {
                            "name": row['fullName'],
                            "position": row['position_name'],
                            "team": row['team_name'],
                            "nationality": row['nationality'],
                            "index": recommendations.index[i-1]
                        }
                        st.session_state.clicked_player = None
                        st.success(f"✅ {row['fullName']} menggantikan {player_info['name']}!")
                        st.rerun()
        elif len(st.session_state.selected_squad) > 0:
            st.info(f"⏳ Silakan pilih {11 - len(st.session_state.selected_squad)} pemain lagi untuk melengkapi squad.")

    # ─── Tab 2: Rekomendasi ────────────────────────────────────────────────────
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
                st.warning(f"Pemain **'{nama_input}'** tidak ditemukan dalam dataset.")
            else:
                st.markdown("---")
                st.markdown("#### 🎯 Pemain yang Dicari")
                info_cols = st.columns(4)
                disp_fields = [
                    ("fullName", "Nama"),
                    ("position_name", "Posisi"),
                    ("team_name", "Tim"),
                    ("nationality", "Kebangsaan"),
                ]
                for ci, (field, label) in enumerate(disp_fields):
                    with info_cols[ci]:
                        val = player_info.get(field, "-") if field in player_info.index else "-"
                        st.metric(label, val)

                st.markdown("---")
                st.markdown(f"#### 🏆 Top-{int(top_n_local)} Pemain Pengganti yang Direkomendasikan")

                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    name = row.get("fullName", "-")
                    pos = row.get("position_name", "-")
                    team = row.get("team_name", "-")
                    nat = row.get("nationality", "-")
                    st.markdown(
                        f"""<div class="player-card">
                        <strong>{i}. {name}</strong><br>
                        <span class="badge">⚽ {pos}</span>
                        <span class="badge">🏟️ {team}</span>
                        <span class="badge">🌍 {nat}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                csv_out = recommendations.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Rekomendasi (CSV)",
                    data=csv_out,
                    file_name=f"rekomendasi_{nama_input.replace(' ', '_')}.csv",
                    mime="text/csv",
                )

    # ─── Tab 3: EDA ───────────────────────────────────────────────────────
    with tab3:
        st.subheader("📊 Exploratory Data Analysis")

        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.markdown("**Distribusi Rating Pemain**")
            if "overallRating" in df_master.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df_master["overallRating"].dropna(), bins=20, kde=True, color="#4361ee", ax=ax)
                ax.set_xlabel("Overall Rating (OVR)")
                ax.set_ylabel("Jumlah Pemain")
                ax.set_title("Distribusi Rating Pemain EA FC")
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Kolom `overallRating` tidak ditemukan di dataset.")

        with col_e2:
            pos_col = None
            for cand in ["alternatePositions/0/shortLabel", "position_name", "alternatePositions_0_label"]:
                if cand in df_master.columns or cand in df_clean.columns:
                    pos_col = cand if cand in df_master.columns else cand
                    break

            st.markdown("**Jumlah Pemain Berdasarkan Posisi**")
            src = df_master if pos_col and pos_col in df_master.columns else df_clean
            pos_col_use = pos_col if pos_col and pos_col in src.columns else None
            if not pos_col_use:
                # fallback
                for cand in src.columns:
                    if "position" in cand.lower():
                        pos_col_use = cand
                        break

            if pos_col_use:
                order_posisi = src[pos_col_use].value_counts().index
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.countplot(data=src, x=pos_col_use, order=order_posisi[:15], palette="viridis", ax=ax2)
                ax2.set_xlabel("Posisi")
                ax2.set_ylabel("Jumlah Pemain")
                ax2.set_title("Jumlah Pemain per Posisi (Top 15)")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
            else:
                st.info("Kolom posisi tidak ditemukan.")

        st.markdown("---")
        st.markdown("**Heatmap Korelasi Fitur Numerik (setelah cleaning)**")
        numeric_df = df_clean.select_dtypes(include="number")
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig3, ax3 = plt.subplots(figsize=(14, 10))
            sns.heatmap(corr, vmin=-1, vmax=1, linewidths=0.3, ax=ax3, cmap="coolwarm")
            ax3.set_title("Korelasi In-Game Stats Pemain Lapangan EA FC")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()
        else:
            st.info("Tidak ada kolom numerik untuk ditampilkan.")

    # ─── Tab 4: Dataset ───────────────────────────────────────────────────
    with tab4:
        st.subheader("🗃️ Preview Dataset (Setelah Cleaning)")

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Pemain", f"{len(df_clean):,}")
        col_m2.metric("Total Fitur", f"{len(df_clean.columns)}")
        col_m3.metric("Fitur Numerik", f"{len(df_clean.select_dtypes('number').columns)}")

        st.dataframe(df_clean.head(100), use_container_width=True)

        st.markdown("**Daftar Kolom:**")
        st.write(list(df_clean.columns))

        full_csv = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Dataset Bersih (CSV)",
            data=full_csv,
            file_name="players_dataset_fix.csv",
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