import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import date
from model import predict_aqi, suggest_precautions, get_actual_vs_predicted, LOCATION_FILES

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirSense · AQI Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS / ANIMATIONS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg-base:    #070b14;
    --bg-card:    rgba(15, 22, 40, 0.85);
    --bg-glass:   rgba(255, 255, 255, 0.04);
    --border:     rgba(99, 160, 255, 0.15);
    --accent:     #3b82f6;
    --accent-2:   #60efff;
    --text-main:  #e2eaf8;
    --text-muted: #6b82a8;
}

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--text-main); }

/* ── Page transition ── */
@keyframes pageEnter {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0); }
}
.page-wrapper { animation:pageEnter 0.45s cubic-bezier(0.22,1,0.36,1) both; }

.stApp {
    background:var(--bg-base);
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(59,130,246,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(96,239,255,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 60% 50%, rgba(167,139,250,0.04) 0%, transparent 50%);
    background-attachment:fixed;
}

.particles { position:fixed; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:0; overflow:hidden; }
.particle  { position:absolute; border-radius:50%; background:radial-gradient(circle,rgba(59,130,246,0.6),transparent); animation:floatUp linear infinite; opacity:0; }
@keyframes floatUp {
    0%   { transform:translateY(100vh) scale(0); opacity:0; }
    10%  { opacity:0.4; }
    90%  { opacity:0.2; }
    100% { transform:translateY(-10vh) scale(1); opacity:0; }
}

section[data-testid="stSidebar"] { background:linear-gradient(180deg,#080e1c 0%,#0a1428 100%) !important; border-right:1px solid var(--border) !important; }
section[data-testid="stSidebar"] * { color:var(--text-main) !important; }
h1,h2,h3 { font-family:'Syne',sans-serif !important; letter-spacing:-0.02em; }

[data-testid="metric-container"] {
    background:var(--bg-card); border:1px solid var(--border); border-radius:16px;
    padding:20px 24px !important; backdrop-filter:blur(20px);
    transition:transform 0.2s ease,box-shadow 0.2s ease; animation:fadeSlideUp 0.5s ease both;
}
[data-testid="metric-container"]:hover { transform:translateY(-3px); box-shadow:0 12px 40px rgba(59,130,246,0.15); }
[data-testid="stMetricLabel"] { color:var(--text-muted) !important; font-size:0.78rem !important; text-transform:uppercase; letter-spacing:0.08em; }
[data-testid="stMetricValue"] { color:var(--text-main)  !important; font-family:'Syne',sans-serif !important; font-size:1.7rem !important; }

.stButton > button {
    background:linear-gradient(135deg,#1d4ed8,#3b82f6) !important; color:white !important;
    border:none !important; border-radius:12px !important; font-family:'Syne',sans-serif !important;
    font-weight:700 !important; font-size:0.9rem !important; letter-spacing:0.04em !important;
    padding:0.6rem 1.2rem !important; transition:all 0.25s ease !important;
    box-shadow:0 4px 20px rgba(59,130,246,0.35) !important; position:relative; overflow:hidden;
}
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 30px rgba(59,130,246,0.5) !important; }
.stButton > button:active { transform:translateY(0) !important; }

.stSelectbox > div > div,
.stDateInput > div > div > input {
    background:var(--bg-card) !important; border:1px solid var(--border) !important;
    border-radius:12px !important; color:var(--text-main) !important; font-family:'DM Sans',sans-serif !important;
}
.stSlider > div > div > div { background:linear-gradient(90deg,#1d4ed8,#60efff) !important; }
hr  { border-color:var(--border) !important; }
.stSpinner > div { border-top-color:var(--accent) !important; }
.js-plotly-plot  { border-radius:16px; }

/* ── Core keyframes ── */
@keyframes fadeSlideUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn      { from{opacity:0} to{opacity:1} }
@keyframes pulse-glow  { 0%,100%{box-shadow:var(--glow-soft)} 50%{box-shadow:var(--glow-strong)} }
@keyframes scanline    { 0%{top:-5%} 100%{top:105%} }
@keyframes countUp     { from{opacity:0;transform:scale(0.8)} to{opacity:1;transform:scale(1)} }

/* ── Skeleton shimmer ── */
@keyframes skeletonShimmer {
    0%   { background-position:-400px 0; }
    100% { background-position:400px 0; }
}
.skeleton {
    background:linear-gradient(90deg,rgba(255,255,255,0.04) 25%,rgba(255,255,255,0.09) 50%,rgba(255,255,255,0.04) 75%);
    background-size:400px 100%;
    animation:skeletonShimmer 1.4s infinite linear;
    border-radius:12px;
}
.skeleton-card  { height:110px; border-radius:16px; margin-bottom:12px; }
.skeleton-chart { height:300px; border-radius:16px; margin-bottom:12px; }
.skeleton-row   { height:16px; border-radius:8px; margin-bottom:8px; }
.skeleton-row.short { width:55%; }

/* ── Glass Card ── */
.glass-card {
    background:var(--bg-card); border:1px solid var(--border); border-radius:20px;
    padding:24px 28px; backdrop-filter:blur(24px); animation:fadeSlideUp 0.6s ease both;
    position:relative; overflow:hidden;
}
.glass-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,rgba(99,160,255,0.4),transparent);
}

/* ── AQI Hero ── */
.aqi-hero {
    text-align:center; padding:32px 16px; border-radius:24px;
    background:var(--bg-card); border:1px solid var(--border);
    backdrop-filter:blur(30px); animation:fadeSlideUp 0.7s ease both;
    position:relative; overflow:hidden; box-sizing:border-box; width:100%;
}
.aqi-hero::after {
    content:''; position:absolute; width:200%; height:2px;
    background:linear-gradient(90deg,transparent,rgba(96,239,255,0.5),transparent);
    animation:scanline 4s linear infinite; left:-50%;
}
.aqi-number {
    font-family:'DM Sans',sans-serif; font-size:clamp(1.8rem,4.5vw,2.8rem);
    font-weight:700; line-height:1.1; white-space:nowrap; overflow:hidden;
    text-overflow:ellipsis; max-width:100%; display:block;
    background:linear-gradient(135deg,#fff 30%,var(--accent-2));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    letter-spacing:-0.02em; margin:0 auto;
}
.aqi-category { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; margin-top:8px; }
.aqi-subtitle  { color:var(--text-muted); font-size:0.85rem; margin-top:6px; letter-spacing:0.05em; }
.mood-emoji    { font-size:3rem; display:block; margin:0 auto 4px auto; animation:fadeSlideUp 0.5s ease both; }

/* ── Accordion ── */
.prec-accordion {
    background:var(--bg-glass); border:1px solid var(--border); border-radius:16px;
    margin-bottom:10px; overflow:hidden; transition:border-color 0.2s; animation:fadeSlideUp 0.5s ease both;
}
.prec-accordion:hover { border-color:rgba(99,160,255,0.35); }
.prec-acc-header {
    display:flex; align-items:center; justify-content:space-between;
    padding:14px 20px; cursor:pointer; user-select:none;
    font-family:'Syne',sans-serif; font-weight:700; font-size:0.9rem; letter-spacing:0.04em;
}
.prec-acc-arrow { font-size:0.75rem; transition:transform 0.25s ease; color:var(--text-muted); }
.prec-acc-body  { padding:0 20px 14px 20px; }
.prec-item { display:flex; align-items:flex-start; gap:10px; margin-bottom:8px; font-size:0.87rem; color:var(--text-muted); line-height:1.5; transition:color 0.2s; }
.prec-item:hover { color:var(--text-main); }
.prec-dot  { width:6px; height:6px; border-radius:50%; background:var(--accent); flex-shrink:0; margin-top:6px; }

/* ── Sparkline ── */
.sparkline-wrap {
    background:var(--bg-card); border:1px solid var(--border); border-radius:16px;
    padding:16px 20px 8px 20px; animation:fadeSlideUp 0.6s ease both;
}
.sparkline-label { font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--text-muted); margin-bottom:8px; }

/* ── Compare card ── */
.compare-card {
    background:var(--bg-card); border:1px solid var(--border); border-radius:20px;
    padding:28px 20px; backdrop-filter:blur(24px); animation:fadeSlideUp 0.6s ease both; text-align:center;
}

/* ── Count-up metric card ── */
.cu-metric {
    background:var(--bg-card); border:1px solid var(--border); border-radius:16px;
    padding:20px 24px; animation:fadeSlideUp 0.5s ease both;
    transition:transform 0.2s,box-shadow 0.2s;
}
.cu-metric:hover { transform:translateY(-3px); box-shadow:0 12px 40px rgba(59,130,246,0.15); }

/* ── Export card ── */
.export-card {
    background:linear-gradient(135deg,rgba(15,22,40,0.95),rgba(10,15,30,0.98));
    border:1px solid var(--border); border-radius:20px; padding:28px 32px;
    animation:fadeSlideUp 0.5s ease both;
}

/* ── Misc ── */
.datetime-banner {
    display:flex; align-items:center; gap:16px; background:rgba(59,130,246,0.07);
    border:1px solid rgba(59,130,246,0.2); border-radius:14px; padding:14px 20px;
    font-size:0.9rem; color:var(--text-muted); margin:10px 0 20px 0; animation:fadeIn 0.4s ease;
}
.datetime-banner strong { color:var(--text-main); font-weight:500; }
.datetime-sep  { width:4px; height:4px; border-radius:50%; background:var(--border); flex-shrink:0; }
.hour-display  { text-align:center; font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:700; color:var(--accent-2); margin-top:4px; letter-spacing:0.06em; }
.section-header {
    font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:var(--text-main);
    letter-spacing:0.04em; padding:0 0 12px 0; border-bottom:1px solid var(--border);
    margin-bottom:20px; display:flex; align-items:center; gap:10px;
}
.page-title {
    font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800; letter-spacing:-0.03em;
    background:linear-gradient(135deg,#e2eaf8 40%,var(--accent-2));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:6px;
}
.page-subtitle { color:var(--text-muted); font-size:0.95rem; margin-bottom:28px; }
.scale-pill { display:flex; align-items:center; gap:8px; padding:6px 12px; border-radius:100px; margin-bottom:8px; background:rgba(255,255,255,0.03); font-size:0.82rem; transition:background 0.2s; }
.scale-pill:hover { background:rgba(255,255,255,0.07); }
.scale-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.stDataFrame { border-radius:12px; overflow:hidden; }
.stAlert { border-radius:12px !important; }

/* ── Hide only menu/footer, keep sidebar fully visible ── */
#MainMenu { visibility:hidden; }
footer    { visibility:hidden; }

/* Hide the collapse toggle button so sidebar stays static */
[data-testid="collapsedControl"]        { display:none !important; }
[data-testid="stSidebarCollapseButton"] { display:none !important; }

/* Ensure sidebar is always shown, never collapsed */
section[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    transform: none !important;
    min-width: 240px !important;
}
</style>

<div class="particles" id="particles"></div>
<script>
(function(){
    const c=document.getElementById('particles');
    if(!c)return;
    for(let i=0;i<18;i++){
        const p=document.createElement('div');
        p.className='particle';
        const s=Math.random()*4+2;
        p.style.cssText=`width:${s}px;height:${s}px;left:${Math.random()*100}%;animation-duration:${Math.random()*20+15}s;animation-delay:${Math.random()*15}s;`;
        c.appendChild(p);
    }
})();
</script>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS & HELPERS
# ──────────────────────────────────────────────────────────────────────────────
AQI_BANDS = [
    (50,  "#34d399", "Good",         "🟢"),   # green
    (100, "#a3e635", "Satisfactory", "🟡"),   # yellow-green
    (200, "#fbbf24", "Moderate",     "🟠"),   # amber
    (300, "#fb7138", "Poor",         "🟠"),   # orange
    (400, "#dc3c3c", "Very Poor",    "🔴"),   # red
    (500, "#a032b4", "Severe",       "🟣"),   # purple
]
MOOD_EMOJI = [
    (50,  "😊", "Air is clean — enjoy outdoors freely!"),
    (100, "🙂", "Mostly fine — sensitive groups take mild care."),
    (200, "😐", "Moderate pollution — limit prolonged outdoor exposure."),
    (300, "😷", "Unhealthy — wear a mask, avoid outdoor exertion."),
    (400, "🤢", "Very poor air — stay indoors, use air purifier."),
    (500, "☠️", "Hazardous — avoid all outdoor activity immediately!"),
]
GLOW_MAP = [
    (50,  "rgba(52,211,153",  "0.30", "0.60"),   # green   — Good
    (100, "rgba(163,230,53",  "0.28", "0.55"),   # yel-grn — Satisfactory
    (200, "rgba(251,191,36",  "0.28", "0.55"),   # amber   — Moderate
    (300, "rgba(251,113,56",  "0.30", "0.58"),   # orange  — Poor
    (400, "rgba(220,60,60",   "0.32", "0.62"),   # red     — Very Poor
    (500, "rgba(160,50,180",  "0.32", "0.62"),   # purple  — Severe
]
GROUP_ICONS = {
    "Children (0-12 yrs)":   "🧒",
    "Children (0\u201312 yrs)":  "🧒",
    "Teenagers / Athletes":  "🏃",
    "Healthy Adults":        "🧑",
    "Elderly (65+)":         "👴",
    "Pregnant Women":        "🤰",
    "Asthma / COPD Patients":"💊",
    "Heart Disease Patients":"❤️",
    "Outdoor Workers":       "👷",
}
LOCATION_LIST = list(LOCATION_FILES.keys())


def get_aqi_color(aqi):
    for t, c, *_ in AQI_BANDS:
        if aqi <= t: return c
    return "#f472b6"

def get_aqi_label(aqi):
    for t, _, l, *_ in AQI_BANDS:
        if aqi <= t: return l
    return "Severe"

def get_mood(aqi):
    for t, em, msg in MOOD_EMOJI:
        if aqi <= t: return em, msg
    return "☠️", "Hazardous — avoid all outdoor activity immediately!"

def get_glow(aqi):
    rgb, s, b = next((r,s,b) for t,r,s,b in GLOW_MAP if aqi<=t)
    return (
        f"0 0 22px {rgb},{s}), 0 0 6px {rgb},0.15)",
        f"0 0 45px {rgb},{b}), 0 0 90px {rgb},0.25)",
        f"{rgb},0.35)",
    )

def normalise_precautions(precautions):
    if isinstance(precautions, dict): return precautions
    prec_dict, current_group = {}, "General"
    KW = ("(", "Patients", "Workers", "Adults", "Women", "Elderly", "Teenagers")
    for item in precautions:
        item = item.strip()
        if not item or item == "\n": continue
        if any(k in item for k in KW) and len(item) < 60:
            current_group = item; prec_dict.setdefault(current_group, [])
        else:
            clean = item.replace("\n\n", "").strip()
            if clean: prec_dict.setdefault(current_group, []).append(clean)
    return prec_dict

def plotly_dark_layout():
    return dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#8099c0"),
        margin=dict(l=20, r=20, t=40, b=20),
    )

def render_skeleton(n_cards=4, show_chart=True):
    cols = st.columns(n_cards)
    for col in cols:
        col.markdown("<div class='skeleton skeleton-card'></div>", unsafe_allow_html=True)
    if show_chart:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='skeleton skeleton-chart'></div>", unsafe_allow_html=True)
        st.markdown("<div class='skeleton skeleton-row'></div>", unsafe_allow_html=True)
        st.markdown("<div class='skeleton skeleton-row short'></div>", unsafe_allow_html=True)

def metric_card_html(label, value, accent, tooltip, delay):
    num_str = value.replace("%", "")
    suffix  = "%" if "%" in value else ""
    try:
        target   = float(num_str)
        decimals = len(num_str.split(".")[-1]) if "." in num_str else 0
    except Exception:
        target, decimals, suffix = 0, 2, ""
    return (
        f"<div class='cu-metric' style='border-top:2px solid {accent};animation-delay:{delay}s'>"
        f"  <div style='font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:#6b82a8;margin-bottom:8px'>{label}</div>"
        f"  <div style='font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;color:{accent};animation:countUp 0.6s {delay+0.15}s ease both'>"
        f"    <span class='cu-val' data-target='{target}' data-suffix='{suffix}' data-dec='{decimals}'>{value}</span>"
        f"  </div>"
        f"  <div style='font-size:0.75rem;color:#3b4a6b;margin-top:6px'>{tooltip}</div>"
        f"</div>"
    )


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 10px 0">
        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                    letter-spacing:-0.02em;background:linear-gradient(135deg,#e2eaf8,#60efff);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
            AirSense
        </div>
        <div style="color:#6b82a8;font-size:0.8rem;margin-top:2px;letter-spacing:0.06em">
            AQI INTELLIGENCE PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigate",
        ["🏠  Prediction", "🆚  City Comparison", "📊  Model Performance", "📖  Methodology"],
        label_visibility="collapsed")
    st.divider()
    st.markdown("<div style='font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:#6b82a8;margin-bottom:12px'>AQI Reference Scale</div>", unsafe_allow_html=True)
    for _, color, label, emoji in AQI_BANDS:
        st.markdown(
            f"<div class='scale-pill'><div class='scale-dot' style='background:{color}'></div>"
            f"<span style='color:#b0c4de;font-size:0.82rem'>{emoji} {label}</span></div>",
            unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='color:#3b4a6b;font-size:0.75rem;text-align:center'>Final Year Project · 2025</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
if "Prediction" in page:
    st.markdown("<div class='page-wrapper'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex;align-items:center;gap:14px;margin-bottom:6px'>
        <svg width="42" height="42" viewBox="0 0 42 42" fill="none" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="windGrad" x1="0" y1="0" x2="42" y2="42" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stop-color="#60efff"/>
              <stop offset="100%" stop-color="#3b82f6"/>
            </linearGradient>
          </defs>
          <!-- Cloud body -->
          <path d="M10 27a7 7 0 0 1 1.5-13.8A9 9 0 0 1 28 16a6 6 0 0 1 5 9H10z"
                fill="url(#windGrad)" opacity="0.9"/>
          <!-- Wind lines -->
          <line x1="6"  y1="31" x2="22" y2="31" stroke="#60efff" stroke-width="2.2" stroke-linecap="round"/>
          <line x1="6"  y1="35" x2="18" y2="35" stroke="#3b82f6" stroke-width="2.2" stroke-linecap="round"/>
          <line x1="6"  y1="39" x2="26" y2="39" stroke="#60efff" stroke-width="1.6" stroke-linecap="round" opacity="0.6"/>
        </svg>
        <span class='page-title' style='margin-bottom:0'>AQI Prediction</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Select a location, date & hour — get AI-driven air quality forecasts with group-tailored health advisories.</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1.5, 2, 1.5, 1])
    with c1:
        st.markdown("<div style='font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;color:#6b82a8;margin-bottom:6px'>📍 Location</div>", unsafe_allow_html=True)
        selected_location = st.selectbox("Location", LOCATION_LIST, index=0, label_visibility="collapsed")
    with c2:
        st.markdown("<div style='font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;color:#6b82a8;margin-bottom:6px'>📅 Date</div>", unsafe_allow_html=True)
        selected_date = st.date_input("Date", value=date(2025, 1, 20), label_visibility="collapsed")
    with c3:
        st.markdown("<div style='font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;color:#6b82a8;margin-bottom:6px'>🕐 Hour</div>", unsafe_allow_html=True)
        selected_hour = st.slider("Hour", 0, 23, 15, format="%d:00", label_visibility="collapsed")
        hour_label = "12:00 noon" if selected_hour == 12 else f"{selected_hour}:00 {'AM' if selected_hour < 12 else 'PM'}"
        st.markdown(f"<div class='hour-display'>⏱ {hour_label}</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("🔍 Predict AQI", use_container_width=True)

    assembled_dt = f"{selected_date} {selected_hour:02d}:00:00"
    st.markdown(
        f"<div class='datetime-banner'><span>📍</span><strong>{selected_location}</strong>"
        f"<div class='datetime-sep'></div><span>{selected_date.strftime('%A, %d %B %Y')}</span>"
        f"<div class='datetime-sep'></div><span>{hour_label}</span></div>",
        unsafe_allow_html=True)

    if predict_clicked:
        skel_ph = st.empty()
        with skel_ph.container():
            render_skeleton(4, True)

        try:
            predicted_aqi         = predict_aqi(assembled_dt, location=selected_location)
            category, precautions = suggest_precautions(predicted_aqi)
            color                 = get_aqi_color(predicted_aqi)
            label                 = get_aqi_label(predicted_aqi)
            emoji, mood_msg       = get_mood(predicted_aqi)
            glow_soft, glow_strong, border_col = get_glow(predicted_aqi)
            skel_ph.empty()

            # ── Hero + KPIs ────────────────────────────────────────────────────
            hero_col, kpi_col = st.columns([1, 2], gap="large")
            with hero_col:
                st.markdown(
                    f"<div class='aqi-hero' style='"
                    f"--glow-soft:{glow_soft};--glow-strong:{glow_strong};"
                    f"border-color:{border_col};"
                    f"animation:pulse-glow 3s ease-in-out infinite,fadeSlideUp 0.7s ease both'>"
                    f"  <span class='mood-emoji'>{emoji}</span>"
                    f"  <div class='aqi-subtitle'>AIR QUALITY INDEX</div>"
                    f"  <div class='aqi-number' style='color:{color};-webkit-text-fill-color:unset'>{round(predicted_aqi,1)}</div>"
                    f"  <div class='aqi-category' style='color:{color}'>{label}</div>"
                    f"  <div class='aqi-subtitle' style='margin-top:14px'>{selected_location} · {hour_label}</div>"
                    f"  <div style='margin-top:10px;font-size:0.78rem;color:{color};opacity:0.8;padding:0 8px'>{mood_msg}</div>"
                    f"</div>",
                    unsafe_allow_html=True)
            with kpi_col:
                k1, k2 = st.columns(2)
                k3, k4 = st.columns(2)
                k1.metric("📍 Location",      selected_location)
                k2.metric("🔢 Predicted AQI", f"{round(predicted_aqi, 2)}")
                k3.metric("🏷️ Category",      category)
                k4.metric("⚠️ Risk Level",    label)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # ── AQI Trend Sparkline ────────────────────────────────────────────
            try:
                from model import get_model_for_location
                df_loc, *_ = get_model_for_location(selected_location)
                input_dt   = pd.to_datetime(assembled_dt)
                spark_df   = df_loc[df_loc["Datetime"] < input_dt].tail(24).copy()
                if len(spark_df) >= 6:
                    st.markdown("<div class='section-header'>📉 24-Hour Historical AQI Trend — leading up to prediction</div>", unsafe_allow_html=True)

                    # hex color → rgb for fill
                    h = color.lstrip("#")
                    r_val, g_val, b_val = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

                    fig_spark = go.Figure()
                    fig_spark.add_trace(go.Scatter(
                        x=spark_df["Datetime"].tolist(), y=spark_df["AQI"].tolist(),
                        mode="lines", line=dict(color=color, width=2),
                        fill="tozeroy", fillcolor=f"rgba({r_val},{g_val},{b_val},0.10)",
                        showlegend=False,
                    ))
                    fig_spark.add_trace(go.Scatter(
                        x=[input_dt], y=[predicted_aqi], mode="markers",
                        marker=dict(color=color, size=10, symbol="diamond",
                                    line=dict(color="#fff", width=2)),
                        name="Predicted", showlegend=True,
                    ))
                    for thresh, bc, bl, _ in AQI_BANDS[:-1]:
                        fig_spark.add_hline(
                            y=thresh, line_dash="dot", line_color=bc, opacity=0.20,
                            annotation_text=bl, annotation_font_size=9, annotation_font_color=bc,
                        )
                    sl = plotly_dark_layout()
                    sl["margin"] = dict(l=10, r=10, t=10, b=30)
                    fig_spark.update_layout(
                        height=175,
                        xaxis=dict(showgrid=False, color="#4a5568", tickformat="%H:%M"),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#4a5568"),
                        legend=dict(orientation="h", x=1, xanchor="right", y=1.2,
                                    bgcolor="rgba(0,0,0,0)", font=dict(color="#8099c0", size=11)),
                        **sl,
                    )
                    st.plotly_chart(fig_spark, use_container_width=True)
            except Exception:
                pass

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # ── Gauge + Risk Bar ───────────────────────────────────────────────
            gauge_col, risk_col = st.columns([3, 2], gap="large")
            with gauge_col:
                st.markdown("<div class='section-header'>📡 AQI Gauge</div>", unsafe_allow_html=True)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=predicted_aqi,
                    title={"text": f"<b>{selected_location}</b>",
                           "font": {"size": 16, "color": "#8099c0", "family": "Syne"}},
                    number={"font": {"size": 52, "color": color, "family": "Syne"}, "suffix": " AQI"},
                    gauge={
                        "axis": {"range":[0,500],"tickvals":[0,50,100,200,300,400,500],
                                 "ticktext":["0","50","100","200","300","400","500"],
                                 "tickcolor":"#3a4a60","tickfont":{"color":"#3a4a60","size":11}},
                        "bar": {"color":"rgba(255,255,255,0.08)","thickness":0.25},
                        "bgcolor":"rgba(0,0,0,0)","bordercolor":"rgba(0,0,0,0)",
                        "steps":[
                            {"range":[0,50],  "color":"rgba(52,211,153,0.55)"},   # green
                            {"range":[50,100], "color":"rgba(163,230,53,0.50)"},  # yellow-green
                            {"range":[100,200],"color":"rgba(251,191,36,0.50)"},  # amber
                            {"range":[200,300],"color":"rgba(251,113,56,0.52)"},  # orange
                            {"range":[300,400],"color":"rgba(220,60,60,0.55)"},   # red
                            {"range":[400,500],"color":"rgba(160,50,180,0.55)"},  # purple
                        ],
                        "threshold":{"line":{"color":"#ffffff","width":3},"thickness":0.85,"value":predicted_aqi},
                    },
                ))
                gl = plotly_dark_layout()
                gl["margin"] = dict(l=30, r=30, t=60, b=10)
                fig_gauge.update_layout(height=360, **gl)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with risk_col:
                st.markdown("<div class='section-header'>📊 Health Risk Breakdown</div>", unsafe_allow_html=True)
                risk_labels = ["Safe Margin","Moderate Risk","High Risk"]
                risk_values = [round(max(0,150-predicted_aqi),1), round(predicted_aqi/2,1), round(predicted_aqi/3,1)]
                fig_risk = go.Figure(go.Bar(
                    x=risk_labels, y=risk_values,
                    marker=dict(color=["#34d399","#fb923c","#f87171"], line=dict(width=0)),
                    text=[f"{v:.0f}" for v in risk_values], textposition="outside",
                    textfont=dict(color="#8099c0", family="DM Sans"), width=0.5,
                ))
                fig_risk.update_layout(
                    yaxis_title="Score",
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#4a5568"),
                    xaxis=dict(color="#4a5568"), height=360, **plotly_dark_layout(),
                )
                st.plotly_chart(fig_risk, use_container_width=True)

            # ── Precaution Accordions (native st.expander) ───────────────────
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>🛡️ Group-Specific Health Advisories</div>", unsafe_allow_html=True)

            # Style st.expander to match the dark theme
            st.markdown(f"""
            <style>
            div[data-testid="stExpander"] {{
                background: var(--bg-glass);
                border: 1px solid {color}44 !important;
                border-radius: 16px !important;
                margin-bottom: 10px;
                overflow: hidden;
                animation: fadeSlideUp 0.5s ease both;
                transition: border-color 0.2s;
            }}
            div[data-testid="stExpander"]:hover {{
                border-color: {color}88 !important;
            }}
            div[data-testid="stExpander"] summary {{
                font-family: 'Syne', sans-serif;
                font-weight: 700;
                font-size: 0.92rem;
                letter-spacing: 0.03em;
                color: {color} !important;
                padding: 14px 20px;
            }}
            div[data-testid="stExpander"] summary:hover {{
                background: rgba(255,255,255,0.02);
            }}
            div[data-testid="stExpander"] svg {{
                stroke: {color} !important;
            }}
            div[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {{
                padding: 0 20px 14px 20px;
                border-top: 1px solid {color}22;
            }}
            </style>
            """, unsafe_allow_html=True)

            prec_dict = normalise_precautions(precautions)
            if prec_dict:
                cols = st.columns(2)
                for i, (group, advice_list) in enumerate(prec_dict.items()):
                    icon = GROUP_ICONS.get(group, "👤")
                    label_with_badge = (
                        f"{icon}  {group}  ·  {len(advice_list)} tip{'s' if len(advice_list) != 1 else ''}"
                    )
                    with cols[i % 2].expander(label_with_badge, expanded=False):
                        for advice in advice_list:
                            st.markdown(
                                f"<div class='prec-item'>"
                                f"  <div class='prec-dot' style='background:{color}'></div>"
                                f"  <span>{advice}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
            else:
                st.markdown("<div class='glass-card' style='text-align:center;color:#6b82a8'>✅ No specific precautions required at this AQI level.</div>", unsafe_allow_html=True)

            # ── Export Card + Download ─────────────────────────────────────────
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>📤 Export Report</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='export-card'>"
                f"  <div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px'>"
                f"    <div>"
                f"      <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;margin-bottom:4px'>AirSense Prediction Report</div>"
                f"      <div style='color:#6b82a8;font-size:0.85rem'>{selected_date.strftime('%A, %d %B %Y')} · {hour_label} · {selected_location}</div>"
                f"    </div>"
                f"    <div style='text-align:right'>"
                f"      <div style='font-size:2rem;font-weight:800;font-family:Syne,sans-serif;color:{color}'>{round(predicted_aqi,1)}</div>"
                f"      <div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:{color}'>{label}</div>"
                f"    </div>"
                f"  </div>"
                f"  <div style='margin-top:16px;display:flex;gap:12px;flex-wrap:wrap'>"
                f"    <div style='flex:1;min-width:100px;background:rgba(255,255,255,0.03);border-radius:12px;padding:12px 16px'>"
                f"      <div style='font-size:0.72rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.08em'>Category</div>"
                f"      <div style='font-weight:600;margin-top:2px'>{category}</div></div>"
                f"    <div style='flex:1;min-width:100px;background:rgba(255,255,255,0.03);border-radius:12px;padding:12px 16px'>"
                f"      <div style='font-size:0.72rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.08em'>Groups Advised</div>"
                f"      <div style='font-weight:600;margin-top:2px'>{len(prec_dict)} groups</div></div>"
                f"    <div style='flex:1;min-width:100px;background:rgba(255,255,255,0.03);border-radius:12px;padding:12px 16px'>"
                f"      <div style='font-size:0.72rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.08em'>Mood</div>"
                f"      <div style='font-weight:600;margin-top:2px'>{emoji} {label}</div></div>"
                f"  </div>"
                f"  <div style='margin-top:14px;font-size:0.8rem;color:#3b4a6b;border-top:1px solid rgba(99,160,255,0.1);padding-top:12px'>"
                f"    Generated by AirSense · XGBoost model trained on historical AQI data · Final Year Project 2025"
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True)

            report_text = f"""AirSense AQI Prediction Report
==============================
Location      : {selected_location}
Date / Time   : {selected_date.strftime('%A, %d %B %Y')} at {hour_label}
Predicted AQI : {round(predicted_aqi, 2)}
Category      : {category}
Risk Level    : {label}
Mood          : {emoji} {mood_msg}

Group-Specific Advisories:
"""
            for grp, tips in prec_dict.items():
                report_text += f"\n{GROUP_ICONS.get(grp,'•')} {grp}:\n"
                for tip in tips:
                    report_text += f"   - {tip}\n"
            report_text += "\n---\nGenerated by AirSense · Final Year Project 2025"

            st.download_button(
                "⬇️ Download Report (.txt)", data=report_text,
                file_name=f"airsense_{selected_location}_{selected_date}.txt",
                mime="text/plain",
            )

        except Exception as e:
            skel_ph.empty()
            st.error(f"⚠️ Prediction error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2 — CITY COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
elif "Comparison" in page:
    st.markdown("<div class='page-wrapper'>", unsafe_allow_html=True)
    st.markdown("""<div style='display:flex;align-items:center;gap:14px;margin-bottom:6px'><svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="cityGrad" x1="0" y1="0" x2="38" y2="38" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#60efff"/>
      <stop offset="100%" stop-color="#3b82f6"/>
    </linearGradient>
  </defs>
  <!-- City A skyline -->
  <rect x="2"  y="18" width="5"  height="16" rx="1" fill="url(#cityGrad)" opacity="0.9"/>
  <rect x="4"  y="12" width="5"  height="22" rx="1" fill="url(#cityGrad)"/>
  <rect x="10" y="20" width="4"  height="14" rx="1" fill="url(#cityGrad)" opacity="0.7"/>
  <!-- VS divider -->
  <line x1="19" y1="6" x2="19" y2="34" stroke="rgba(96,239,255,0.3)" stroke-width="1.2" stroke-dasharray="3 2"/>
  <!-- City B skyline -->
  <rect x="24" y="16" width="5"  height="18" rx="1" fill="url(#cityGrad)" opacity="0.9"/>
  <rect x="26" y="10" width="5"  height="24" rx="1" fill="url(#cityGrad)"/>
  <rect x="32" y="22" width="4"  height="12" rx="1" fill="url(#cityGrad)" opacity="0.7"/>
  <!-- Ground line -->
  <line x1="1" y1="34" x2="37" y2="34" stroke="rgba(96,239,255,0.25)" stroke-width="1.2"/>
</svg><span class='page-title' style='margin-bottom:0'>City Comparison</span></div>""", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Compare AQI predictions for two cities side-by-side at the same date and hour.</div>", unsafe_allow_html=True)

    cc1, cc2, cc3, cc4 = st.columns([2, 2, 1.5, 1])
    with cc1:
        st.markdown("<div style='font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;color:#6b82a8;margin-bottom:6px'>🏙️ City A</div>", unsafe_allow_html=True)
        city_a = st.selectbox("City A", LOCATION_LIST, index=0, label_visibility="collapsed", key="cmp_a")
    with cc2:
        st.markdown("<div style='font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;color:#6b82a8;margin-bottom:6px'>🏙️ City B</div>", unsafe_allow_html=True)
        city_b = st.selectbox("City B", LOCATION_LIST, index=min(1,len(LOCATION_LIST)-1), label_visibility="collapsed", key="cmp_b")
    with cc3:
        st.markdown("<div style='font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;color:#6b82a8;margin-bottom:6px'>📅 Date</div>", unsafe_allow_html=True)
        cmp_date = st.date_input("CmpDate", value=date(2025,1,20), label_visibility="collapsed", key="cmp_date")
    with cc4:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        compare_clicked = st.button("Compare", use_container_width=True)

    cmp_hour = st.slider("Hour of Day", 0, 23, 15, format="%d:00", key="cmp_hour")
    cmp_hour_label = "12:00 noon" if cmp_hour == 12 else f"{cmp_hour}:00 {'AM' if cmp_hour < 12 else 'PM'}"

    if compare_clicked:
        skel_ph = st.empty()
        with skel_ph.container():
            render_skeleton(2, True)
        try:
            cmp_dt = f"{cmp_date} {cmp_hour:02d}:00:00"
            aqi_a  = predict_aqi(cmp_dt, location=city_a)
            aqi_b  = predict_aqi(cmp_dt, location=city_b)
            skel_ph.empty()

            col_a, col_vs, col_b = st.columns([5, 1, 5], gap="small")

            def render_compare_card(col, city, aqi):
                c = get_aqi_color(aqi); lbl = get_aqi_label(aqi)
                em, _ = get_mood(aqi); gs, gb, bc = get_glow(aqi)
                col.markdown(
                    f"<div class='compare-card' style='--glow-soft:{gs};--glow-strong:{gb};"
                    f"border-color:{bc};animation:pulse-glow 3s ease-in-out infinite,fadeSlideUp 0.5s ease both'>"
                    f"  <div style='font-size:2.5rem'>{em}</div>"
                    f"  <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;margin:6px 0 2px 0'>{city}</div>"
                    f"  <div style='font-size:2.6rem;font-weight:800;font-family:Syne,sans-serif;color:{c};line-height:1'>{round(aqi,1)}</div>"
                    f"  <div style='font-size:0.85rem;text-transform:uppercase;letter-spacing:0.1em;color:{c};margin-top:4px'>{lbl}</div>"
                    f"  <div style='margin-top:12px;font-size:0.8rem;color:#4a5568'>{cmp_date.strftime('%d %b %Y')} · {cmp_hour_label}</div>"
                    f"</div>",
                    unsafe_allow_html=True)

            render_compare_card(col_a, city_a, aqi_a)
            col_vs.markdown(
                "<div style='display:flex;align-items:center;justify-content:center;height:100%;font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#3b4a6b;padding-top:60px'>VS</div>",
                unsafe_allow_html=True)
            render_compare_card(col_b, city_b, aqi_b)

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>📊 Side-by-Side AQI Comparison</div>", unsafe_allow_html=True)

            fig_cmp = go.Figure()
            for city, aqi in [(city_a, aqi_a), (city_b, aqi_b)]:
                c = get_aqi_color(aqi)
                fig_cmp.add_trace(go.Bar(
                    name=city, x=[city], y=[round(aqi,2)],
                    marker=dict(color=c, line=dict(width=0)),
                    text=[f"{round(aqi,1)}"], textposition="outside",
                    textfont=dict(color=c, family="Syne", size=16), width=0.35,
                ))
            prev = 0
            for thresh, bc2, _ in [(50,"rgba(52,211,153,0.06)",""), (100,"rgba(251,191,36,0.06)",""),
                                    (200,"rgba(251,146,60,0.06)",""), (300,"rgba(248,113,113,0.06)",""),
                                    (400,"rgba(167,139,250,0.06)",""), (500,"rgba(244,114,182,0.06)","")]:
                fig_cmp.add_hrect(y0=prev, y1=thresh, fillcolor=bc2, layer="below", line_width=0)
                prev = thresh
            fig_cmp.update_layout(
                barmode="group",
                yaxis=dict(range=[0,520], showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#4a5568", title="AQI"),
                xaxis=dict(color="#4a5568"),
                legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)", font=dict(color="#8099c0")),
                height=360, **plotly_dark_layout(),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            better_city  = city_a if aqi_a <= aqi_b else city_b
            diff         = abs(aqi_a - aqi_b)
            better_color = get_aqi_color(min(aqi_a, aqi_b))
            st.markdown(
                f"<div class='glass-card' style='text-align:center;padding:20px'>"
                f"  <div style='font-size:1.5rem'>🏆</div>"
                f"  <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;margin-top:6px'>"
                f"    <span style='color:{better_color}'>{better_city}</span> has cleaner air"
                f"    <span style='color:#4a5568'> by </span>"
                f"    <span style='color:{better_color}'>{diff:.1f} AQI points</span>"
                f"  </div>"
                f"  <div style='color:#4a5568;font-size:0.82rem;margin-top:4px'>at {cmp_hour_label} on {cmp_date.strftime('%d %b %Y')}</div>"
                f"</div>",
                unsafe_allow_html=True)

        except Exception as e:
            skel_ph.empty()
            st.error(f"⚠️ Comparison error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3 — MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
elif "Performance" in page:
    st.markdown("<div class='page-wrapper'>", unsafe_allow_html=True)
    st.markdown("""<div style='display:flex;align-items:center;gap:14px;margin-bottom:6px'><svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="perfGrad" x1="0" y1="0" x2="38" y2="38" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#60efff"/>
      <stop offset="100%" stop-color="#3b82f6"/>
    </linearGradient>
  </defs>
  <!-- Bar chart bars -->
  <rect x="4"  y="24" width="7" height="10" rx="1.5" fill="url(#perfGrad)" opacity="0.6"/>
  <rect x="15" y="16" width="7" height="18" rx="1.5" fill="url(#perfGrad)" opacity="0.8"/>
  <rect x="26" y="8"  width="7" height="26" rx="1.5" fill="url(#perfGrad)"/>
  <!-- Trend line -->
  <polyline points="7,26 18,18 30,10" stroke="#60efff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  <circle cx="30" cy="10" r="2.5" fill="#60efff"/>
  <!-- Axis -->
  <line x1="2" y1="34" x2="36" y2="34" stroke="rgba(96,239,255,0.25)" stroke-width="1.2"/>
  <line x1="2" y1="4"  x2="2"  y2="34" stroke="rgba(96,239,255,0.25)" stroke-width="1.2"/>
</svg><span class='page-title' style='margin-bottom:0'>Model Performance</span></div>""", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>XGBoost evaluation on the held-out test set (last 20% of data).</div>", unsafe_allow_html=True)

    perf_location = st.selectbox("Select location to evaluate:", LOCATION_LIST, index=0)

    skel_ph = st.empty()
    with skel_ph.container():
        render_skeleton(4, True)

    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        actual, predicted = get_actual_vs_predicted(location=perf_location)
        mae  = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2   = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / np.where(actual==0,1,actual))) * 100
        skel_ph.empty()

        # ── Animated count-up metric cards ────────────────────────────────────
        cards_html = (
            "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px'>"
            + metric_card_html("MAE",  f"{mae:.2f}",   "#60efff", "Mean Absolute Error — lower is better", 0.0)
            + metric_card_html("RMSE", f"{rmse:.2f}",  "#a78bfa", "Root Mean Squared Error", 0.1)
            + metric_card_html("R²",   f"{r2:.4f}",    "#34d399", "Coefficient of determination — closer to 1.0 is better", 0.2)
            + metric_card_html("MAPE", f"{mape:.2f}%", "#fb923c", "Mean Absolute Percentage Error", 0.3)
            + "</div>"
        )
        countup_js = """
        <script>
        (function(){
            document.querySelectorAll('.cu-val').forEach(function(el){
                var target=parseFloat(el.getAttribute('data-target'));
                var suffix=el.getAttribute('data-suffix')||'';
                var dec=parseInt(el.getAttribute('data-dec'))||2;
                var start=0,duration=1200,startTime=null;
                function step(ts){
                    if(!startTime)startTime=ts;
                    var prog=Math.min((ts-startTime)/duration,1);
                    var ease=1-Math.pow(1-prog,3);
                    el.textContent=(start+(target-start)*ease).toFixed(dec)+suffix;
                    if(prog<1)requestAnimationFrame(step);
                }
                requestAnimationFrame(step);
            });
        })();
        </script>
        """
        st.markdown(cards_html + countup_js, unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Actual vs Predicted + Confidence Band ─────────────────────────────
        st.markdown("<div class='section-header'>📈 Actual vs Predicted AQI</div>", unsafe_allow_html=True)
        n_points = len(actual)
        sample   = min(n_points, 800)
        idx      = np.linspace(0, n_points-1, sample, dtype=int)
        act_s    = actual[idx]
        pred_s   = predicted[idx]
        xs       = list(range(sample))

        # rolling std of residuals as confidence band
        roll_std   = pd.Series(actual - predicted).rolling(20, min_periods=1).std().fillna(0).values
        roll_std_s = roll_std[idx]
        upper = pred_s + roll_std_s
        lower = pred_s - roll_std_s

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=xs + xs[::-1],
            y=upper.tolist() + lower[::-1].tolist(),
            fill="toself", fillcolor="rgba(249,115,22,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="±1σ Confidence Band",
        ))
        fig_line.add_trace(go.Scatter(
            x=xs, y=act_s.tolist(), mode="lines", name="Actual AQI",
            line=dict(color="#60efff", width=1.6),
            fill="tozeroy", fillcolor="rgba(96,239,255,0.04)",
        ))
        fig_line.add_trace(go.Scatter(
            x=xs, y=pred_s.tolist(), mode="lines", name="Predicted AQI",
            line=dict(color="#f97316", width=1.6, dash="dot"),
        ))
        fig_line.update_layout(
            xaxis_title="Test Data Points (sampled)", yaxis_title="AQI",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)", font=dict(color="#8099c0")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#4a5568"),
            xaxis=dict(showgrid=False, color="#4a5568"),
            height=380, **plotly_dark_layout(),
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # ── Residuals + Scatter ───────────────────────────────────────────────
        rc1, rc2 = st.columns(2, gap="large")
        with rc1:
            st.markdown("<div class='section-header'>📉 Residuals Distribution</div>", unsafe_allow_html=True)
            residuals = actual - predicted
            fig_hist  = px.histogram(x=residuals, nbins=60,
                labels={"x":"Residual (Actual − Predicted)","y":"Count"},
                color_discrete_sequence=["#6366f1"])
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#f87171", line_width=2)
            fig_hist.update_layout(
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#4a5568"),
                xaxis=dict(color="#4a5568"), height=320, **plotly_dark_layout())
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown("<div style='color:#4a5568;font-size:0.8rem;margin-top:-10px'>Residuals centred near zero → low systematic bias.</div>", unsafe_allow_html=True)

        with rc2:
            st.markdown("<div class='section-header'>🎯 Prediction Accuracy Scatter</div>", unsafe_allow_html=True)
            fig_scatter = px.scatter(x=actual, y=predicted,
                labels={"x":"Actual AQI","y":"Predicted AQI"},
                opacity=0.45, color_discrete_sequence=["#3b82f6"])
            mn = float(min(actual.min(), predicted.min()))
            mx = float(max(actual.max(), predicted.max()))
            fig_scatter.add_trace(go.Scatter(
                x=[mn,mx], y=[mn,mx], mode="lines", name="Perfect",
                line=dict(color="#f87171", dash="dash", width=2)))
            fig_scatter.update_layout(
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#4a5568"),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#4a5568"),
                height=320, **plotly_dark_layout())
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown("<div style='color:#4a5568;font-size:0.8rem;margin-top:-10px'>Points near the diagonal indicate accurate predictions.</div>", unsafe_allow_html=True)

    except Exception as e:
        skel_ph.empty()
        st.error(f"Error loading model data: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 4 — METHODOLOGY
# ──────────────────────────────────────────────────────────────────────────────
elif "Methodology" in page:
    st.markdown("<div class='page-wrapper'>", unsafe_allow_html=True)
    st.markdown("""<div style='display:flex;align-items:center;gap:14px;margin-bottom:6px'><svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="methGrad" x1="0" y1="0" x2="38" y2="38" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#60efff"/>
      <stop offset="100%" stop-color="#3b82f6"/>
    </linearGradient>
  </defs>
  <!-- Document shape -->
  <path d="M8 4 H24 L32 12 V34 H8 Z" fill="rgba(59,130,246,0.15)" stroke="url(#methGrad)" stroke-width="1.5"/>
  <!-- Folded corner -->
  <path d="M24 4 L24 12 L32 12" fill="none" stroke="url(#methGrad)" stroke-width="1.5"/>
  <!-- Text lines -->
  <line x1="13" y1="18" x2="27" y2="18" stroke="#60efff" stroke-width="1.8" stroke-linecap="round" opacity="0.9"/>
  <line x1="13" y1="23" x2="27" y2="23" stroke="#60efff" stroke-width="1.8" stroke-linecap="round" opacity="0.7"/>
  <line x1="13" y1="28" x2="22" y2="28" stroke="#60efff" stroke-width="1.8" stroke-linecap="round" opacity="0.5"/>
</svg><span class='page-title' style='margin-bottom:0'>Methodology & About</span></div>""", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Architecture, features, and design decisions behind AirSense.</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🌐 Project Overview</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='glass-card'>This system predicts the Air Quality Index (AQI) at a given datetime and location "
        "using a machine learning model trained on historical AQI observations. Based on the predicted AQI, "
        "the system provides evidence-based health recommendations tailored to different population groups "
        "including children, elderly individuals, pregnant women, athletes, and patients with chronic "
        "respiratory or cardiovascular conditions.</div>",
        unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📍 Supported Locations</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"City":list(LOCATION_FILES.keys()),"Data File":list(LOCATION_FILES.values())}), use_container_width=True, hide_index=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🗄️ Dataset & Preprocessing</div>", unsafe_allow_html=True)
    md1, md2 = st.columns(2, gap="large")
    with md1:
        st.markdown(
            "<div class='glass-card'>"
            "<div style='font-family:Syne,sans-serif;font-weight:700;margin-bottom:10px'>Source</div>"
            "<div style='color:#8099c0;font-size:0.9rem'>Historical AQI readings stored per city in <code>.xlsx</code> files.</div>"
            "<div style='font-family:Syne,sans-serif;font-weight:700;margin:14px 0 8px 0'>Format</div>"
            "<div style='color:#8099c0;font-size:0.9rem'>Wide format — rows are dates, columns are hourly time slots. Melted to long format during preprocessing.</div>"
            "</div>", unsafe_allow_html=True)
    with md2:
        steps_html = "".join(f"<div class='prec-item'><div class='prec-dot'></div><span>{s}</span></div>"
            for s in ["Melt wide → long format","Parse Datetime from Date + Time columns",
                      "Coerce non-numeric AQI values to NaN and drop","Sort chronologically and reset index"])
        st.markdown(f"<div class='glass-card'><div style='font-family:Syne,sans-serif;font-weight:700;margin-bottom:12px'>Preprocessing Steps</div>{steps_html}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🔧 Feature Engineering</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Feature":["hour","day","dayofweek","lag1","lag2","lag24","lag48","lag72","rolling_mean_3","rolling_mean_6","rolling_mean_12"],
        "Type":["Temporal"]*3+["Lag"]*5+["Rolling"]*3,
        "Description":["Hour of day (0–23)","Day of month (1–31)","Day of week (0=Monday)",
            "AQI 1 hour prior","AQI 2 hours prior","AQI 24 hours prior","AQI 48 hours prior","AQI 72 hours prior",
            "3-hour rolling mean","6-hour rolling mean","12-hour rolling mean"],
    }), use_container_width=True, hide_index=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🤖 Model Configuration</div>", unsafe_allow_html=True)
    mm1, mm2 = st.columns(2, gap="large")
    with mm1:
        info_html = "".join(
            f"<div style='margin-bottom:12px'><div style='font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:#4a5568;margin-bottom:2px'>{k}</div><div style='color:#e2eaf8;font-weight:500'>{v}</div></div>"
            for k,v in [("Algorithm","XGBoost Regressor"),("Train/Test Split","80% / 20% (chronological)"),("Evaluation Metrics","MAE, RMSE, R², MAPE"),("Model Caching","Each city's model is trained once and cached")])
        st.markdown(f"<div class='glass-card'>{info_html}</div>", unsafe_allow_html=True)
    with mm2:
        params_html = "".join(
            f"<div class='prec-item'><div class='prec-dot' style='background:#6366f1'></div><span><code style='color:#a78bfa'>{k}</code><span style='color:#4a5568'> = </span><code style='color:#60efff'>{v}</code></span></div>"
            for k,v in {"n_estimators":100,"learning_rate":0.05,"max_depth":6,"subsample":0.8,"colsample_bytree":0.8,"tree_method":"hist"}.items())
        st.markdown(f"<div class='glass-card'><div style='font-family:Syne,sans-serif;font-weight:700;margin-bottom:12px'>Hyperparameters</div>{params_html}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🏥 AQI Health Classification</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "AQI Range":["0–50","51–100","101–200","201–300","301–400","401–500"],
        "Category": ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"],
        "Primary Health Concern":["Minimal impact","Minor breathing discomfort for sensitive individuals",
            "Breathing discomfort for people with lung/heart disease","Breathing discomfort for most people on prolonged exposure",
            "Respiratory illness on prolonged exposure","Serious respiratory and cardiovascular effects"],
    }), use_container_width=True, hide_index=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>⚠️ Limitations</div>", unsafe_allow_html=True)
    lim_html = "".join(f"<div class='prec-item'><div class='prec-dot' style='background:#f87171'></div><span>{l}</span></div>" for l in [
        "The model does not incorporate real-time meteorological data (wind speed, humidity, temperature).",
        "Predictions are based solely on historical AQI patterns; sudden events cannot be anticipated.",
        "The 80/20 chronological split means the model was not evaluated against unseen seasonal periods.",
        "Health advisory thresholds follow general Indian AQI classification guidelines.",
        "Lag features require at least 72 hours of prior historical data.",
    ])
    st.markdown(f"<div class='glass-card'>{lim_html}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🛠️ Technologies Used</div>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3, gap="large")
    for title, items, col in [
        ("⚙️ Machine Learning",["XGBoost","scikit-learn","pandas / NumPy"], t1),
        ("📊 Visualisation",   ["Plotly","Streamlit"], t2),
        ("🖥️ Environment",     ["Python 3.x","Streamlit web app","Excel datasets (.xlsx)"], t3),
    ]:
        items_html = "".join(f"<div class='prec-item'><div class='prec-dot' style='background:#3b82f6'></div><span>{it}</span></div>" for it in items)
        col.markdown(f"<div class='glass-card'><div style='font-family:Syne,sans-serif;font-weight:700;margin-bottom:10px'>{title}</div>{items_html}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
