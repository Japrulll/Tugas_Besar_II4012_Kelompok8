import streamlit as st
from PIL import Image
import base64
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pickle
import google.generativeai as genai


# Config
im = Image.open("EA_FC_26_LOGO.png")
st.set_page_config(
    page_title="AI Payer Scouting",
    page_icon=im,
    layout="wide",
)
api_key_gemini = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key_gemini)
llm_model = genai.GenerativeModel('gemini-flash-latest')
# print("--- DAFTAR MODEL YANG BISA DIPAKAI ---")
# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)
# print("--------------------------------------")

# Data
with open("aiscout_data.pkl","rb") as file:
    df_clean, cosin_sim_final = pickle.load(file)   

# Style 
@st.cache_data
def get_bg_image(image):
    with open(image,"rb") as file:
        data = file.read()
    return base64.b64encode(data).decode()

img = get_bg_image("bg-image.png")

page_by_img = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"]{{
    background-image: url("data:image/png;base64,{img}") !important;
    background-size: cover;
    color: #e8f0e8;
    font-family: 'DM Sans', sans-serif;
}}
[data-testid="stHeader"] {{
    background-color: transparent;
}}

hr {{
    border-color: #715413 !important;
}}

.stTabs [data-baseweb="tab-list"] {{ background: transparent; gap: 8px; }}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: #e8f0e8;
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 1px;
    font-size: 1rem;
}}
.stTabs [aria-selected="true"] * {{
    color: #e8f0e8 !important;
    font-weight: bold !important;
}}

[data-testid="stWidgetLabel"] p {{
    color: #e8f0e8 !important;
    font-weight: bold !important;
}}

.stButton button {{
    background: transparent !important;
    border: 1px solid #715413 !important;
    color: #e8f0e8 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}}
.stButton button:hover {{
    background: #807a38 !important;
    border: 2px solid #d1cb5c !important;
}}

</style>
"""

# Formasi 
FORMATIONS = {
    "4-3-3": {
        "rows": [
            {"label": "FWD", "slots": 3},
            {"label": "MID", "slots": 3},
            {"label": "DEF", "slots": 4},
            {"label": "GK",  "slots": 1},
        ]
    },
    "4-4-2": {
        "rows": [
            {"label": "FWD", "slots": 2},
            {"label": "MID", "slots": 4},
            {"label": "DEF", "slots": 4},
            {"label": "GK",  "slots": 1},
        ]
    },
    "4-2-3-1": {
        "rows": [
            {"label": "FWD", "slots": 1},
            {"label": "CAM", "slots": 3},
            {"label": "CDM", "slots": 2},
            {"label": "DEF", "slots": 4},
            {"label": "GK",  "slots": 1},
        ]
    },
    "3-5-2": {
        "rows": [
            {"label": "FWD", "slots": 2},
            {"label": "MID", "slots": 5},
            {"label": "DEF", "slots": 3},
            {"label": "GK",  "slots": 1},
        ]
    },
    "5-3-2": {
        "rows": [
            {"label": "FWD", "slots": 2},
            {"label": "MID", "slots": 3},
            {"label": "DEF", "slots": 5},
            {"label": "GK",  "slots": 1},
        ]
    },
}

POSITION_RULE = {
    "Goalkeeper": ["Goalkeeper"],

    "Center Back":                  ["Center Back", "Right Back", "Left Back"],
    "Right Back":                   ["Right Back", "Center Back", "Right Midfielder"],
    "Left Back":                    ["Left Back", "Center Back", "Left Midfielder"],

    "Center Midfielder":            ["Center Midfielder", "Center Defensive Midfielder", "Center Attacking Midfielder"],
    "Center Defensive Midfielder":  ["Center Defensive Midfielder", "Center Midfielder"],
    "Center Attacking Midfielder":  ["Center Attacking Midfielder", "Center Midfielder", "Right Winger", "Left Winger"],
    "Right Midfielder":             ["Right Midfielder", "Right Winger", "Center Midfielder"],
    "Left Midfielder":              ["Left Midfielder", "Left Winger", "Center Midfielder"],

    "Striker":                      ["Striker", "Center Forward", "Right Winger", "Left Winger"],
    "Center Forward":               ["Center Forward", "Striker", "Center Attacking Midfielder"],
    "Right Winger":                 ["Right Winger", "Striker", "Right Midfielder", "Center Attacking Midfielder"],
    "Left Winger":                  ["Left Winger", "Striker", "Left Midfielder", "Center Attacking Midfielder"],
}

defaults = {
    "squad" : {},
    "sel_slot": None,
    "formation": "4-4-2",
    "compare_candidate": None,
    "llm_insight": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Helper Functions
def get_slot(formation_name):
    ids = []
    for i in FORMATIONS[formation_name]["rows"]:
        for j in range (i["slots"]):
            ids.append(f"{i["label"]}_{j}")
    return ids

def get_recommendations(nama_pemain, top_n=5):
    search = df_clean[df_clean["fullName"].str.contains(nama_pemain,
                                                        case=False,
                                                        na=False)]
    if search.empty:
        return None
    
    idx = search.index[0]
    ref_pos = search.iloc[0].get("position_name","")
    
    all_scores = list(enumerate(cosin_sim_final[idx]))
    all_scores = [(i,s) for i,s in all_scores if i != idx]
    all_scores = sorted(all_scores, key=lambda x:x[1], reverse=True)

    same_pos = [ref_pos]
    rlv_pos = POSITION_RULE.get(ref_pos,[ref_pos])

    result_indices = []

    main_pos = [i for i,s in all_scores
                if df_clean.loc[i,"position_name"] in same_pos][:top_n]
    result_indices.extend(main_pos)

    if len(result_indices) < top_n:
        remain = top_n - len(result_indices)
        other_pos = [i for i,s in all_scores
                     if df_clean.loc[i, "position_name"] in rlv_pos][:remain]
        result_indices.extend(other_pos)

    if len(result_indices) < top_n:
        remain = top_n - len(result_indices)
        other_other_pos = [i for i,s in all_scores
                     if i not in result_indices][:remain]
        result_indices.extend(other_other_pos)
    
    cols = ["fullName", "position_name", "team_name", "player_card"]
    cols = [c for c in cols if c in df_clean.columns]
    return df_clean[cols].iloc[result_indices].reset_index(drop=True)

def player_image(url, fallback="⚽"):
    return url if (pd.notna(url) and str(url).strip() not in ["", "nan", "None"]) else None

def add_player_to_slot(slot_id, player_row):
    st.session_state.squad[slot_id] = {
        "name": player_row["fullName"],
        "position": player_row.get("position_name", "-"),
        "team": player_row.get("team_name", "-"),
        "image": player_row.get("player_card", None),
    }

def get_ai_scout_report(p_lama, p_baru):
    prompt = f"""
    Sebagai pemandu bakat sepak bola profesional, berikan analisis taktis maksimal 4 kalimat 
    tentang prospek mengganti {p_lama['name']} ({p_lama['position']} di tim ea fc 26 saya) 
    dengan {p_baru['fullName']} ({p_baru['position_name']} di {p_baru.get('team_name', '-')}).
    Fokus pada gaya bermain dan kecocokannya.
    """
    
    try:
        # 3. Proses pengiriman prompt ke server Google dan menunggu balasan (response)
        response = llm_model.generate_content(prompt)
        
        # 4. Mengekstrak teks dari keseluruhan data balasan
        hasil_teks = response.text
        
        return hasil_teks
        
    except Exception as e:
        # Jaga-jaga kalau internet putus atau API error
        return f"Mohon maaf, AI Scout sedang tidak bisa dihubungi saat ini. Detail error: {str(e)}"

st.markdown(page_by_img, unsafe_allow_html=True)
st.title("AI Player Scouting")
st.markdown("---")

tab_squad, tab_rec = st.tabs(["🏟️  SQUAD BUILDER", "🔍  REKOMENDASI"])

with tab_squad:
    ctrl1, ctrl2, ctrl3 = st.columns([5, 5, 1])
    with ctrl1:
        new_formation = st.selectbox("Formasi", list(FORMATIONS.keys()), index=list(FORMATIONS.keys()).index(st.session_state.formation), key="form_select")
        if new_formation != st.session_state.formation:
            st.session_state.formation = new_formation
            st.session_state.squad = {}
            st.session_state.sel_slot = None
            st.rerun()
    with ctrl2:
        search_q = st.selectbox("🔎 Cari Pemain", df_clean["fullName"], index=None, placeholder="Nama pemain…", key="squad_search")
    with ctrl3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.squad = {}
            st.session_state.sel_slot = None
            st.rerun()
    
    # ── Search results ────────────────────────────────────────────────────────
    # Cek apakah search_q ada isinya (tidak None)
    if search_q:
        found = df_clean[df_clean["fullName"].str.contains(search_q, case=False, na=False)]
        
        if not found.empty:
            idx = found.iloc[0]
            img = found.iloc[0].get("player_card","")
            name = found.iloc[0].get("fullName","")
            post = found.iloc[0].get("position_name","")
            team = found.iloc[0].get("team_name","")
            liga = found.iloc[0].get("leagueName","")
            nati = found.iloc[0].get("nationality","")

            slot_ids = get_slot(st.session_state.formation)
            filled_slots = list(st.session_state.squad.keys())
            empty_slots = [s for s in slot_ids if s not in filled_slots]

            st.markdown(f"**Pemain ditemukan** — pilih slot kosong lalu klik Tambah:")
            sc1, sc2 = st.columns([1,4],gap="large")
            with sc1:
                st.image(img, use_container_width=True)
            with sc2:
                st.markdown(f"# {name}")
                st.write("—"*114)
                st.markdown(f"**Posisi Pemain:** {post}")
                st.markdown(f"**Klub Asal:** {team}")
                st.markdown(f"**Liga:** {liga   }")
                st.markdown(f"**Kebangsaan:** {nati}")
                st.markdown("<br>", unsafe_allow_html=True)
                if empty_slots:
                    ins_btn, _ = st.columns([2,8])
                    with ins_btn:
                        slot_pick = st.selectbox("Pilih Slot Lapangan", empty_slots, key="slotpick_squad_search")
                        if st.button("➕ Tambah", key="add_squad_search", use_container_width=True):
                            add_player_to_slot(slot_pick, idx)
                            st.success(f"✅ {idx['fullName']} berhasil ditambahkan!")
                            st.rerun()
                   
            st.markdown("---")
        else:
            st.warning("Pemain tidak ditemukan.")
            st.markdown("---")

    # ── Pitch ─────────────────────────────────────────────────────────────────
    formation_data = FORMATIONS[st.session_state.formation]
    slot_ids       = get_slot(st.session_state.formation)
    squad          = st.session_state.squad
    total_slots    = len(slot_ids)
    total_filled   = len(squad)


    pitch1, pitch2 = st.columns([7,3],gap=None)
    with pitch1:
        st.markdown("## Lineup Pemain")
        st.markdown(f"FORMASI {st.session_state.formation} · {total_filled}/{total_slots} PEMAIN")
        slot_cursor = 0
        for row_data in formation_data["rows"]:
            n = row_data["slots"]
            label = row_data["label"]
            row_slot_ids = slot_ids[slot_cursor: slot_cursor + n]
            slot_cursor += n

            # Menambahkan kolom padding agar baris pemain berada di tengah
            pad = (6 - n) // 2
            all_cols = st.columns([0.5]*pad + [0.5]*n + [0.5]*pad) if pad > 0 else st.columns(n)
            player_cols = all_cols[pad:pad+n] if pad > 0 else all_cols

            for ci, (col, sid) in enumerate(zip(player_cols, row_slot_ids)):
                with col:
                    if sid in squad:
                        p = squad[sid]
                        img = player_image(p["image"])
                        is_selected = st.session_state.sel_slot == sid

                        st.markdown(
                            f'<div style="text-align:center;padding:8px;align-items:center;justify-content:center;">'
                            f'  <img src="{img}" style="width:40%; border-radius:10px;"/>'
                            f'  <div style="text-align:center; font-weight:bold; color:#e8f0e8;">{p["name"].split()[-1]}</div>'
                            f'  <div style="text-align:center; color:#d1cb5c;">{label}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        btn_label = "✅ Dipilih" if is_selected else "🔄 Ganti"
                        if st.button(btn_label, key=f"sel_{sid}", use_container_width=True):
                            st.session_state.sel_slot = None if is_selected else sid
                            st.rerun()  
                    else:
                        # Slot kosong di lapangan
                        st.markdown(
                            f'<div style="text-align:center;padding:8px">'
                            f'  <div style="width:80px;height:80px;border:2px dashed #715413;border-radius:8px;'
                            f'      display:flex;align-items:center;justify-content:center;margin:0 auto;color:#715413;font-size:1.4rem">+</div>'
                            f'  <div class="player-pos" style="margin-top:4px; color:#e8f0e8;">{label}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
    with pitch2:
        st.markdown("#### Statistik Tim")
        if total_filled < total_slots:
            st.markdown("⏳ Lengkapi 11 pemain untuk melihat statistik tim.")
        else:
            squad_names = [p["name"] for p in squad.values()]
            df_squad = df_clean[df_clean["fullName"].isin(squad_names)].drop_duplicates("fullName")

            STAT_GROUPS = {
                "Fisik":     ["acceleration", "agility", "jumping", "stamina", "strength", "balance"],
                "Menyerang": ["finishing", "shotPower", "volleys", "penalties", "positioning", "headingAccuracy"],
                "Teknik":    ["ballControl", "dribbling", "crossing", "freeKickAccuracy", "longPassing", "shortPassing"],
                "Bertahan":  ["defensiveAwareness", "standingTackle", "aggression"],
                "Mental":    ["composure", "reactions", "vision"],
            }

            radar_labels = list(STAT_GROUPS.keys())
            radar_values = []
            for group_cols in STAT_GROUPS.values():
                available = [c for c in group_cols if c in df_squad.columns]
                if available:
                    radar_values.append(df_squad[available].mean().mean())
                else:
                    radar_values.append(0)

            fig = px.line_polar(df_squad,r=radar_values,theta=radar_labels, line_close=True)
            fig.update_traces(fill='toself', fillcolor='rgba(209, 203, 92, 0.2)',line_color='#d1cb5c')
            fig.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=False, range=[0, 100]),
                    angularaxis=dict(color='#e8f0e8')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            fig, ax = plt.subplots()
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            ax.tick_params(colors='#e8f0e8', labelsize=10, labelfontfamily='DM Sans')

            bars = ax.barh(radar_labels,radar_values, color="#d1cb5c")
            ax.set_title("Statistik Lengkap", color='#e8f0e8', fontfamily='DM Sans', fontweight='bold')
            ax.bar_label(bars, padding=5, color='#e8f0e8', fontfamily='DM Sans', fontweight='bold')

            st.pyplot(fig, use_container_width=True)

            st.markdown("---")
            
            sorted_stats = sorted(zip(radar_labels, radar_values), key=lambda x: x[1])
            kategori_terendah = sorted_stats[0][0] 
            nilai_terendah = sorted_stats[0][1]
            kategori_terendah_1 = sorted_stats[1][0] 
            nilai_terendah_1 = sorted_stats[1][1]

            st.markdown("<p style='color:#e8f0e8; font-weight:bold; font-size:1.1rem; margin-bottom:5px;'>Insight</p>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="
                    background-color: rgba(209, 203, 92, 0.2); 
                    color: #e8f0e8; 
                    padding: 15px; 
                    border: 4px solid #715413;
                    border-radius: 8px; 
                    font-family: 'DM Sans', sans-serif; 
                    font-size: 0.95rem;
                    line-height: 1.5;
                    box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
                ">
                    Tim Anda memiliki kekurangan di sektor <b>{kategori_terendah}</b> ({nilai_terendah:.2f}) dan 
                    <b>{kategori_terendah_1}</b> ({nilai_terendah_1:.2f}). Anda disarankan untuk mengoptimalisasi sektor-sektor tersebut
                    untuk mendapatkan tim yang lebih baik secara statistik.
                </div>
                """, 
                unsafe_allow_html=True
            )


    st.markdown("---")

    # ── Rekomendasi Pengganti ─────────────────────────────────────────────────
    if st.session_state.sel_slot:
        sid = st.session_state.sel_slot
        p   = squad.get(sid)
        if p:
            st.markdown(f"### 🎯 Rekomendasi Pengganti untuk **{p['name']}**")
            st.markdown(
                f"<div style='padding: 10px; background: rgba(0,0,0,0.5); border-radius: 8px; margin-bottom: 20px;'>"
                f"<b>{p['name']}</b> · {p['position']} · {p['team']}"
                f"</div>",
                unsafe_allow_html=True
            )

            recs = get_recommendations(p["name"], top_n=20)
            if recs is not None:
                
                # ── JIKA ADA KANDIDAT YANG SEDANG DIBANDINGKAN ──
                if st.session_state.compare_candidate is not None:
                    candidate = st.session_state.compare_candidate
                    
                    st.markdown("### ⚖️ Perbandingan Pemain")
                    st.markdown("---")
                    
                    # Layout 3 Kolom: [Pemain Lama] [Info/Statistik] [Kandidat Baru]
                    c1, c2, c3 = st.columns([1, 1.5, 1], gap="medium")
                    
                    with c1:
                        st.markdown("<div style='text-align:center; color:#ccc;'>Pemain Saat Ini</div>", unsafe_allow_html=True)
                        if p.get("image"):
                            st.image(player_image(p["image"]), use_container_width=True)
                        st.markdown(f"<h4 style='text-align:center;'>{p['name']}</h4>", unsafe_allow_html=True)
                    
                    with c3:
                        st.markdown("<div style='text-align:center; color:#d1cb5c;'>Kandidat Pengganti</div>", unsafe_allow_html=True)
                        if candidate.get("player_card"):
                            st.image(player_image(candidate["player_card"]), use_container_width=True)
                        st.markdown(f"<h4 style='text-align:center;'>{candidate['fullName']}</h4>", unsafe_allow_html=True)

                    with c2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align:center;'><b>Posisi:</b><br>{p['position']} ➔ {candidate.get('position_name','-')}</div>", unsafe_allow_html=True)
                        
                        # ── BAGIAN AI SCOUT REPORT ──
                        st.markdown("<div style='text-align:center; color:#d1cb5c; font-size:1rem; margin-top:20px;'><b>Pendapat LLM: Gemini</b></div>", unsafe_allow_html=True)
                        
                        # Cek apakah insight sudah ada di memori, jika belum, panggil LLM
                        if st.session_state.llm_insight is None:
                            with st.spinner("Scout sedang menganalisis kecocokan pemain..."):
                                st.session_state.llm_insight = get_ai_scout_report(p, candidate)
                        
                        # Tampilkan hasil analisis LLM dalam kotak keren
                        st.markdown(
                            f"<div style='background:rgba(0, 0, 0, 0.4); padding:12px; border-radius:8px; font-size:1rem; text-align:center; margin-bottom:15px; color:#e8f0e8; border: 1px solid rgba(209, 203, 92, 0.3);'>"
                            f"<i>{st.session_state.llm_insight}</i>"
                            f"</div>", 
                            unsafe_allow_html=True
                        )
                        # ─────────────────────────────

                        # Tombol Aksi
                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            if st.button("✅ Konfirmasi", use_container_width=True):
                                add_player_to_slot(sid, candidate)
                                st.session_state.compare_candidate = None
                                st.session_state.llm_insight = None # Bersihkan lagi
                                st.session_state.selected_slot = None
                                st.rerun()
                        with btn_col2:
                            if st.button("❌ Batal", use_container_width=True):
                                st.session_state.compare_candidate = None
                                st.session_state.llm_insight = None # Bersihkan lagi
                                st.rerun()
                    
                # ── JIKA BELUM ADA YANG DIBANDINGKAN (TAMPILKAN LIST PEMAIN) ──
                else:
                    for row_start in range(0, 20, 10):
                        rec_cols = st.columns(10)
                        for ci, col in enumerate(rec_cols):
                            ri = row_start + ci
                            if ri >= len(recs):
                                break   
                            rec = recs.iloc[ri]
                            img = player_image(rec.get("player_card"))
                            with col:
                                if img:
                                    st.image(img, use_container_width=True)
                                else:
                                    st.markdown("<div style='height:80px;display:flex;align-items:center;justify-content:center;font-size:2rem'>⚽</div>", unsafe_allow_html=True)
                                
                                st.markdown(f"<div class='rec-name' style='font-weight:bold; font-size:0.8rem;'>{rec['fullName']}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='rec-meta' style='font-size:0.7rem; color:#ccc;'>{rec.get('position_name','-')}<br>{rec.get('team_name','-')}</div>", unsafe_allow_html=True)
                                
                                # Tombol ini sekarang fungsinya memicu layar perbandingan, bukan langsung menukar
                                if st.button("Bandingkan", key=f"pick_rec_{ri}_{sid}", use_container_width=True):
                                    st.session_state.compare_candidate = rec
                                    st.session_state.llm_insight = None
                                    st.rerun()
            else:
                st.warning("Pemain tidak ditemukan di dataset untuk rekomendasi.")
    