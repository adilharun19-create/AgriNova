import html
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# ============================================================
# AGRINOVA — APPLE-INSPIRED LIQUID GLASS FRONTEND
# Backend contract is intentionally unchanged:
#   GET  http://127.0.0.1:8000/health
#   POST http://127.0.0.1:8000/predict
# ============================================================

st.set_page_config(
    page_title="AgriNova | Smart Irrigation",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_BASE}/predict"
HEALTH_URL = f"{API_BASE}/health"

# Unsplash image: aerial rice terraces in Sidemen, Bali.
# A dark overlay keeps foreground text readable; CSS gradient is the fallback.
RICE_BACKGROUND = (
    "https://images.unsplash.com/photo-1646928998297-5fb2a10aca27"
    "?auto=format&fit=crop&w=2400&q=85"
)

if "history" not in st.session_state:
    st.session_state.history = []
if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None
if "api_online" not in st.session_state:
    st.session_state.api_online = False


# ============================================================
# GLOBAL CSS — FROSTED / LIQUID GLASS
# ============================================================

st.markdown(
f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {{
    --ink: #17271d;
    --ink-soft: #34483a;
    --muted: #6f8074;
    --green: #4f805b;
    --green-dark: #244c34;
    --line: rgba(255,255,255,.52);
    --glass: rgba(247,251,247,.78);
    --glass-strong: rgba(250,253,250,.88);
}}

html, body, [class*="css"] {{
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

.stApp {{
    background-color: #1c3828;
    background-image:
        linear-gradient(115deg, rgba(8,25,16,.68) 0%, rgba(10,35,20,.45) 48%, rgba(8,27,17,.56) 100%),
        url('{RICE_BACKGROUND}');
    background-size: cover;
    background-position: center center;
    background-attachment: fixed;
}}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {{
    background: transparent !important;
}}

[data-testid="stHeader"] {{
    background: rgba(10,28,17,.12) !important;
}}

.block-container {{
    max-width: 1480px;
    padding: 2rem 2.8rem 3.5rem;
}}

/* Sidebar as a softly frosted pane */
[data-testid="stSidebar"] {{
    background: rgba(245,250,245,.88) !important;
    border-right: 1px solid rgba(255,255,255,.55);
    backdrop-filter: blur(28px) saturate(160%);
    -webkit-backdrop-filter: blur(28px) saturate(160%);
}}
[data-testid="stSidebar"] > div:first-child {{
    background: transparent !important;
}}
[data-testid="stSidebar"] * {{
    color: #20382a;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    font-size: 13px;
    font-weight: 600;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {{
    gap: 7px;
}}

/* Typography */
h1, h2, h3 {{
    font-family: 'Manrope', 'DM Sans', sans-serif !important;
    color: #fff !important;
    letter-spacing: -.7px;
}}
p, label {{
    color: inherit;
}}
[data-testid="stMarkdownContainer"] p {{
    line-height: 1.55;
}}

/* Brand / page header */
.brand-row {{
    display: flex;
    align-items: center;
    gap: 11px;
    margin: 3px 0 30px;
}}
.brand-mark {{
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 14px;
    background: linear-gradient(145deg, rgba(255,255,255,.94), rgba(218,237,220,.76));
    border: 1px solid rgba(255,255,255,.85);
    box-shadow: 0 8px 24px rgba(13,40,22,.12), inset 0 1px 0 rgba(255,255,255,.95);
    color: #2e6740;
    font: 800 20px 'Manrope', sans-serif;
}}
.brand-name {{
    color: #fff;
    font: 800 17px 'Manrope', sans-serif;
    letter-spacing: -.4px;
}}
.brand-sub {{
    color: rgba(255,255,255,.72);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.8px;
    margin-top: 2px;
}}
.eyebrow {{
    color: rgba(255,255,255,.78);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}}
.page-title {{
    color: #fff;
    font: 800 clamp(30px, 4vw, 48px)/1.08 'Manrope', sans-serif;
    letter-spacing: -1.8px;
    text-shadow: 0 3px 24px rgba(0,0,0,.18);
}}
.page-subtitle {{
    color: rgba(255,255,255,.82);
    font-size: 14px;
    margin-top: 10px;
    max-width: 680px;
}}

/* Liquid glass cards */
.glass-card {{
    background: var(--glass);
    border: 1px solid rgba(255,255,255,.72);
    border-radius: 23px;
    padding: 21px 22px;
    color: var(--ink);
    box-shadow:
        0 16px 45px rgba(7,28,15,.15),
        inset 0 1px 0 rgba(255,255,255,.9),
        inset 0 -1px 0 rgba(255,255,255,.22);
    backdrop-filter: blur(24px) saturate(165%);
    -webkit-backdrop-filter: blur(24px) saturate(165%);
}}
.glass-card-strong {{
    background: var(--glass-strong);
    border: 1px solid rgba(255,255,255,.82);
    border-radius: 23px;
    padding: 22px;
    color: var(--ink);
    box-shadow: 0 16px 45px rgba(7,28,15,.14), inset 0 1px 0 rgba(255,255,255,.96);
    backdrop-filter: blur(28px) saturate(170%);
    -webkit-backdrop-filter: blur(28px) saturate(170%);
}}
.card-label {{
    color: #718276;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.35px;
    text-transform: uppercase;
}}
.card-value {{
    color: #203a29;
    font: 800 29px 'Manrope', sans-serif;
    letter-spacing: -1.1px;
    margin-top: 9px;
}}
.card-note {{
    color: #718176;
    font-size: 11px;
    margin-top: 3px;
}}
.card-icon {{
    width: 34px;
    height: 34px;
    border-radius: 12px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(91,139,100,.13);
    color: #477653;
    font-size: 16px;
    font-weight: 700;
}}
.section-kicker {{
    color: rgba(255,255,255,.76);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin: 8px 0 12px;
}}
.panel-heading {{
    color: #203a29;
    font: 700 16px 'Manrope', sans-serif;
}}
.panel-caption {{
    color: #718176;
    font-size: 11px;
    margin-top: 5px;
}}

/* Hero glass */
.hero-glass {{
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    background: linear-gradient(115deg, rgba(249,253,249,.84), rgba(239,248,239,.65));
    border: 1px solid rgba(255,255,255,.82);
    border-radius: 28px;
    padding: 27px 30px;
    box-shadow: 0 22px 55px rgba(5,25,13,.17), inset 0 1px 0 rgba(255,255,255,.96);
    backdrop-filter: blur(30px) saturate(175%);
    -webkit-backdrop-filter: blur(30px) saturate(175%);
}}
.hero-glass:after {{
    content: '';
    position: absolute;
    width: 260px;
    height: 260px;
    right: -95px;
    top: -155px;
    border-radius: 50%;
    background: rgba(255,255,255,.35);
    filter: blur(10px);
    pointer-events: none;
}}
.hero-title {{
    color: #203a29;
    font: 800 clamp(25px, 3vw, 37px)/1.12 'Manrope', sans-serif;
    letter-spacing: -1.2px;
}}
.hero-copy {{
    color: #687b6d;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 8px;
    max-width: 570px;
}}
.status-chip {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 8px 12px;
    color: #2f6b40;
    background: rgba(225,241,225,.9);
    border: 1px solid rgba(109,158,116,.18);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .8px;
}}
.status-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4b9b5d;
    box-shadow: 0 0 0 4px rgba(75,155,93,.13);
}}

/* Circular gauge */
.gauge-wrap {{
    min-width: 128px;
    text-align: center;
    position: relative;
    z-index: 1;
}}
.gauge {{
    --gauge-value: 0;
    --gauge-color: #54865f;
    width: 116px;
    height: 116px;
    border-radius: 50%;
    margin: 0 auto 7px;
    display: grid;
    place-items: center;
    background: conic-gradient(var(--gauge-color) calc(var(--gauge-value) * 1%), rgba(83,126,91,.14) 0);
    box-shadow: 0 7px 22px rgba(37,79,46,.10), inset 0 1px 2px rgba(255,255,255,.8);
    position: relative;
}}
.gauge:before {{
    content: '';
    position: absolute;
    inset: 8px;
    border-radius: 50%;
    background: rgba(248,252,248,.95);
    border: 1px solid rgba(255,255,255,.95);
    box-shadow: inset 0 2px 7px rgba(31,65,38,.06);
}}
.gauge-inner {{
    position: relative;
    z-index: 1;
    color: #24452e;
    font: 800 25px 'Manrope', sans-serif;
    letter-spacing: -1px;
}}
.gauge-label {{
    color: #718176;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}}

/* Native Streamlit controls */
div[data-testid="stForm"] {{
    background: rgba(247,251,247,.86);
    border: 1px solid rgba(255,255,255,.8);
    border-radius: 22px;
    padding: 23px;
    box-shadow: 0 16px 40px rgba(7,28,15,.14), inset 0 1px 0 rgba(255,255,255,.95);
    backdrop-filter: blur(25px) saturate(160%);
    -webkit-backdrop-filter: blur(25px) saturate(160%);
}}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {{
    color: #344b39 !important;
    font-weight: 600;
    font-size: 12px;
}}
div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {{
    background: rgba(255,255,255,.88) !important;
    border-radius: 11px !important;
    border-color: rgba(86,119,91,.20) !important;
}}
.stSlider [data-baseweb="slider"] {{
    padding-top: 4px;
}}
.stButton > button,
[data-testid="stFormSubmitButton"] > button {{
    background: linear-gradient(135deg, #477b53, #315f3d) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,.35) !important;
    border-radius: 999px !important;
    min-height: 44px;
    font-weight: 700;
    box-shadow: 0 7px 18px rgba(42,91,54,.19);
    transition: transform .18s ease, box-shadow .18s ease;
}}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 10px 23px rgba(42,91,54,.25);
}}
[data-testid="stExpander"] {{
    background: rgba(247,251,247,.82);
    border: 1px solid rgba(255,255,255,.72);
    border-radius: 14px;
    color: #203a29;
}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    background: rgba(255,255,255,.9);
    border-radius: 14px;
}}
[data-testid="stAlert"] {{
    border-radius: 14px;
    backdrop-filter: blur(16px);
}}
[data-testid="stVegaLiteChart"] {{
    background: rgba(250,253,250,.88);
    border: 1px solid rgba(255,255,255,.72);
    border-radius: 18px;
    padding: 10px;
}}

.footer {{
    color: rgba(255,255,255,.72);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-align: center;
    padding: 28px 0 5px;
}}

@media (max-width: 850px) {{
    .block-container {{ padding: 1.2rem 1rem 2rem; }}
    .hero-glass {{ align-items: flex-start; flex-direction: column; padding: 22px; }}
    .gauge-wrap {{ align-self: center; }}
}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def safe(value):
    return html.escape(str(value))


def metric_card(label, value, note, symbol):
    st.markdown(
        f"""
        <div class="glass-card" style="min-height:142px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div class="card-label">{safe(label)}</div>
                <div class="card-icon">{safe(symbol)}</div>
            </div>
            <div class="card-value">{safe(value)}</div>
            <div class="card-note">{safe(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def gauge_card(value, label, suffix="%", color="#54865f"):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    numeric = max(0.0, min(100.0, numeric))
    display = f"{numeric:.1f}{suffix}" if suffix else f"{numeric:.1f}"
    st.markdown(
        f"""
        <div class="gauge-wrap">
            <div class="gauge" style="--gauge-value:{numeric};--gauge-color:{color};">
                <div class="gauge-inner">{display}</div>
            </div>
            <div class="gauge-label">{safe(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text):
    st.markdown(f'<div class="section-kicker">{safe(text)}</div>', unsafe_allow_html=True)


def recommendation_panel(result):
    pred = result.get("hasil_prediksi", {})
    probability = float(pred.get("probabilitas", 0) or 0)
    threshold = float(pred.get("threshold", 0.5) or 0.5)
    needs_water = bool(pred.get("perlu_siram", False))

    if needs_water:
        title = "Irrigation recommended"
        detail = "The model indicates that current conditions meet the irrigation decision threshold."
        accent = "#a66b37"
        background = "rgba(250,242,230,.88)"
    else:
        title = "No irrigation required"
        detail = "The model indicates that current conditions are below the irrigation decision threshold."
        accent = "#376b46"
        background = "rgba(235,246,235,.90)"

    st.markdown(
        f"""
        <div class="glass-card-strong" style="border-left:5px solid {accent};background:{background};">
            <div class="card-label">MODEL RECOMMENDATION</div>
            <div style="font:800 23px 'Manrope',sans-serif;color:#203a29;margin-top:8px;">
                {safe(title)}
            </div>
            <div class="card-note" style="font-size:12px;margin-top:7px;">
                {safe(detail)}
            </div>
            <div style="display:flex;gap:24px;flex-wrap:wrap;margin-top:16px;">
                <div><div class="card-label">Probability</div><strong style="color:#203a29;font-size:17px;">{probability * 100:.2f}%</strong></div>
                <div><div class="card-label">Threshold</div><strong style="color:#203a29;font-size:17px;">{threshold * 100:.0f}%</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    reasons = result.get("alasan", [])
    if reasons:
        with st.expander("View model explanation"):
            for reason in reasons:
                st.write(f"• {reason}")


def fetch_api_status():
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


# ============================================================
# API STATUS
# ============================================================

st.session_state.api_online = fetch_api_status()
api_online = st.session_state.api_online


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-mark">A</div>
            <div>
                <div class="brand-name" style="color:#203a29;">AgriNova</div>
                <div class="brand-sub" style="color:#758679;">SMART IRRIGATION</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("WORKSPACE")
    page = st.radio(
        "Navigation",
        ["Overview", "New prediction", "History", "System"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if api_online:
        st.markdown(
            '<div class="status-chip"><span class="status-dot"></span> API CONNECTED</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#9a5a3e;font-size:11px;font-weight:700;">API DISCONNECTED</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("MODEL DEPLOYMENT")
    st.caption("Streamlit · FastAPI · Joblib")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown('<div class="eyebrow">AGRICULTURE INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)

if page == "Overview":
    page_title = "A clearer view of your field."
    page_subtitle = "Environmental readings and machine-learning irrigation insights in one place."
elif page == "New prediction":
    page_title = "Understand what your crop needs."
    page_subtitle = "Enter the latest plant and environmental conditions to request a prediction."
elif page == "History":
    page_title = "Your prediction history."
    page_subtitle = "Review the sensor inputs and model outputs recorded in this session."
else:
    page_title = "System status."
    page_subtitle = "Application components and backend connectivity."

st.markdown(f'<div class="page-title">{safe(page_title)}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="page-subtitle">{safe(page_subtitle)}</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":
    latest = st.session_state.latest_prediction

    if latest:
        sensor = latest.get("data_sensor", {})
        pred = latest.get("hasil_prediksi", {})
        soil = sensor.get("kelembaban_tanah_pct", "—")
        temp = sensor.get("suhu_udara_C", "—")
        humidity = sensor.get("kelembaban_udara_pct", "—")
        light = sensor.get("cahaya_lux", "—")
        plant = str(sensor.get("tanaman", "Plant")).title()
        probability = float(pred.get("probabilitas", 0) or 0) * 100
        soil_type = str(sensor.get("jenis_tanah", "—")).title()
        age = sensor.get("umur_tanaman_hari", "—")
        decision = bool(pred.get("perlu_siram", False))
    else:
        soil = temp = humidity = light = "—"
        plant, soil_type, age = "No reading", "—", "—"
        probability = 0.0
        decision = False

    status_text = "CONNECTED · MODEL READY" if api_online else "API CONNECTION REQUIRED"
    hero_title = "Your field, understood." if latest else "A smarter way to care for every crop."
    hero_copy = (
        f"Latest analysis for {plant}: soil moisture and environmental conditions have been evaluated by your trained model."
        if latest else
        "Connect your analysis workflow with environmental readings and receive a clear irrigation recommendation."
    )

    st.markdown(
        f"""
        <div class="hero-glass">
            <div style="position:relative;z-index:1;max-width:690px;">
                <div class="status-chip"><span class="status-dot"></span>{safe(status_text)}</div>
                <div class="hero-title" style="margin-top:16px;">{safe(hero_title)}</div>
                <div class="hero-copy">{safe(hero_copy)}</div>
            </div>
            <div class="gauge-wrap">
                <div class="gauge" style="--gauge-value:{max(0,min(100,probability))};--gauge-color:#4f805b;">
                    <div class="gauge-inner">{probability:.1f}%</div>
                </div>
                <div class="gauge-label">AI probability</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_label("01 / Environmental readings")
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        metric_card("Soil moisture", f"{soil}%" if soil != "—" else "—", "Latest reading", "◌")
    with m2:
        metric_card("Air temperature", f"{temp}°C" if temp != "—" else "—", "Ambient temperature", "°")
    with m3:
        metric_card("Air humidity", f"{humidity}%" if humidity != "—" else "—", "Relative humidity", "≈")
    with m4:
        metric_card("Light intensity", f"{light} lux" if light != "—" else "—", "Measured illumination", "✧")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.25, 1], gap="large")

    with left:
        section_label("02 / Sensor history")
        st.markdown(
            '<div class="glass-card-strong"><div class="panel-heading">Soil moisture over time</div><div class="panel-caption">Only readings submitted during this app session are shown.</div></div>',
            unsafe_allow_html=True,
        )
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            chart_df = history_df[["time", "soil_moisture"]].set_index("time")
            st.line_chart(chart_df, height=235, color="#4f805b")
        else:
            st.markdown(
                '<div class="glass-card" style="margin-top:8px;color:#536b59;font-size:12px;">No sensor history yet. Submit a prediction to populate this chart with actual readings.</div>',
                unsafe_allow_html=True,
            )

    with right:
        section_label("03 / Field overview")
        st.markdown(
            f"""
            <div class="glass-card-strong">
                <div class="panel-heading">{safe(plant)}</div>
                <div class="panel-caption">Latest crop profile</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:19px;">
                    <div><div class="card-label">Soil type</div><div style="font:700 17px 'Manrope';color:#203a29;margin-top:5px;">{safe(soil_type)}</div></div>
                    <div><div class="card-label">Plant age</div><div style="font:700 17px 'Manrope';color:#203a29;margin-top:5px;">{safe(age)} days</div></div>
                </div>
                <div style="height:1px;background:rgba(69,105,76,.14);margin:18px 0;"></div>
                <div class="card-label">Decision status</div>
                <div style="font:700 16px 'Manrope';color:{'#96663a' if decision else '#376b46'};margin-top:6px;">{'Irrigation recommended' if latest and decision else 'No irrigation required' if latest else 'Awaiting first prediction'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    section_label("04 / Latest model recommendation")
    if latest:
        recommendation_panel(latest)
    else:
        st.markdown(
            '<div class="glass-card-strong"><div class="panel-heading">No prediction yet</div><div class="panel-caption">Open New prediction to submit the first set of sensor readings. The dashboard will populate using the actual API response.</div></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# NEW PREDICTION PAGE
# ============================================================

elif page == "New prediction":
    section_label("01 / Plant and soil conditions")
    st.markdown(
        '<div class="glass-card-strong" style="margin-bottom:14px;"><div class="panel-heading">Sensor input</div><div class="panel-caption">The field names and JSON payload remain compatible with the existing FastAPI endpoint.</div></div>',
        unsafe_allow_html=True,
    )

    with st.form("irrigation_prediction_form"):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**Plant profile**")
            tanaman = st.selectbox("Plant type", ["selada", "cabai", "tomat"])
            jenis_tanah = st.selectbox("Soil type", ["pasir", "lempung", "liat"])
            umur = st.number_input("Plant age (days)", min_value=0, max_value=365, value=30)
            kelembaban_tanah = st.slider("Soil moisture (%)", min_value=0.1, max_value=100.0, value=38.0, step=0.1)
            curah_hujan = st.number_input("Current rainfall (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
            prakiraan_hujan = st.number_input("Rain forecast — next 6 hours (mm)", min_value=0.0, max_value=200.0, value=0.0, step=0.5)

        with col2:
            st.markdown("**Environmental conditions**")
            suhu = st.slider("Air temperature (°C)", min_value=0.0, max_value=50.0, value=24.0, step=0.5)
            kelembaban_udara = st.slider("Air humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            cahaya = st.number_input("Light intensity (lux)", min_value=0.0, max_value=150000.0, value=40000.0, step=500.0)
            jam = st.slider("Observation hour", min_value=0, max_value=23, value=11)

        submitted = st.form_submit_button("Analyze irrigation needs", use_container_width=True)

    if submitted:
        if not api_online:
            st.error("FastAPI is not connected. Start the backend first, then retry.")
        else:
            # Keep the existing API payload keys exactly as defined in src/api.py.
            payload = {
                "tanaman": tanaman,
                "jenis_tanah": jenis_tanah,
                "umur_tanaman_hari": int(umur),
                "kelembaban_tanah_pct": float(kelembaban_tanah),
                "suhu_udara_C": float(suhu),
                "kelembaban_udara_pct": float(kelembaban_udara),
                "cahaya_lux": float(cahaya),
                "curah_hujan_mm": float(curah_hujan),
                "prakiraan_hujan_6jam_mm": float(prakiraan_hujan),
                "jam": int(jam),
            }
            try:
                with st.spinner("Analyzing field conditions..."):
                    response = requests.post(PREDICT_URL, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.latest_prediction = result
                    prediction = result.get("hasil_prediksi", {})
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "plant": tanaman,
                        "soil_type": jenis_tanah,
                        "soil_moisture": float(kelembaban_tanah),
                        "temperature": float(suhu),
                        "air_humidity": float(kelembaban_udara),
                        "light_lux": float(cahaya),
                        "probability": float(prediction.get("probabilitas", 0) or 0),
                        "irrigation_needed": bool(prediction.get("perlu_siram", False)),
                    })
                    st.success("Prediction completed. The overview has been updated.")
                else:
                    st.error(f"API error {response.status_code}: {response.text}")
            except requests.RequestException as exc:
                st.error(f"Could not reach the prediction API: {exc}")

    if st.session_state.latest_prediction:
        st.markdown("<br>", unsafe_allow_html=True)
        section_label("02 / Latest prediction")
        recommendation_panel(st.session_state.latest_prediction)
        latest_pred = st.session_state.latest_prediction.get("hasil_prediksi", {})
        gauge_cols = st.columns(3)
        with gauge_cols[0]:
            gauge_card(float(latest_pred.get("probabilitas", 0) or 0) * 100, "Irrigation probability")
        with gauge_cols[1]:
            gauge_card(float(latest_pred.get("threshold", 0.5) or 0.5) * 100, "Decision threshold", color="#7896a4")
        with gauge_cols[2]:
            latest_sensor = st.session_state.latest_prediction.get("data_sensor", {})
            gauge_card(float(latest_sensor.get("kelembaban_tanah_pct", 0) or 0), "Soil moisture")


# ============================================================
# HISTORY PAGE
# ============================================================

elif page == "History":
    section_label("01 / Recorded predictions")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.markdown(
            '<div class="glass-card-strong"><div class="panel-heading">Soil moisture trend</div><div class="panel-caption">Recorded readings in the current Streamlit session.</div></div>',
            unsafe_allow_html=True,
        )
        st.line_chart(history_df.set_index("time")[["soil_moisture"]], height=250, color="#4f805b")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="glass-card-strong"><div class="panel-heading">Prediction records</div><div class="panel-caption">Inputs and outputs recorded during this session.</div></div>',
            unsafe_allow_html=True,
        )
        display_df = history_df.rename(columns={
            "time": "Time",
            "plant": "Plant",
            "soil_type": "Soil type",
            "soil_moisture": "Soil moisture (%)",
            "temperature": "Temperature (°C)",
            "air_humidity": "Air humidity (%)",
            "light_lux": "Light (lux)",
            "probability": "Probability",
            "irrigation_needed": "Irrigation needed",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div class="glass-card-strong"><div class="panel-heading">No history available</div><div class="panel-caption">Prediction records will appear here after you submit sensor inputs.</div></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# SYSTEM PAGE
# ============================================================

else:
    section_label("01 / Service health")
    status_text = "Connected" if api_online else "Disconnected"
    status_color = "#376b46" if api_online else "#a14e3d"
    st.markdown(
        f"""
        <div class="glass-card-strong">
            <div class="panel-heading">FastAPI service</div>
            <div style="font:800 24px 'Manrope';color:{status_color};margin-top:10px;">{status_text}</div>
            <div class="panel-caption">Health endpoint: GET /health</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    section_label("02 / Application architecture")
    arch_cols = st.columns(3)
    with arch_cols[0]:
        metric_card("Frontend", "Streamlit", "Liquid Glass interface", "01")
    with arch_cols[1]:
        metric_card("Backend", "FastAPI", "Existing REST API", "02")
    with arch_cols[2]:
        metric_card("Model artifact", "Joblib", "Existing trained model", "03")
    st.markdown(
        '<div class="glass-card" style="margin-top:16px;"><div class="card-label">PREDICTION ENDPOINT</div><div class="card-value" style="font-size:19px;">POST /predict</div><div class="card-note">The frontend sends the existing sensor JSON schema to the current API.</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">AGRINOVA · SMART IRRIGATION INTELLIGENCE · MACHINE LEARNING DEPLOYMENT</div>',
    unsafe_allow_html=True,
)
