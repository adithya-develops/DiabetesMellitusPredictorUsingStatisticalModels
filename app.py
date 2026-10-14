from pathlib import Path
import sys
import html

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data import load_data
from bayes import MixedBayesianPredictor
from stats import proportion_ci, chi_square, one_way_anova

st.set_page_config(
    page_title="Diabetes Study",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = ROOT / "data" / "diabetes_prediction_dataset.csv"
df = load_data(DATA_PATH)
predictor = MixedBayesianPredictor(df)

# -----------------------------------------------------------------------------
# Visual system — dark, high-contrast research workspace
# -----------------------------------------------------------------------------
st.markdown(
r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg: #0b0e11;
  --sidebar: #0e1216;
  --surface: #12171c;
  --surface-raised: #171e24;
  --surface-hover: #1b232a;
  --line: #252d34;
  --line-strong: #35414a;
  --ink: #edf1f4;
  --muted: #9aa5ad;
  --subtle: #65717a;
  --lime: #d8f56c;
  --mint: #73d8b4;
  --blue: #8caeff;
  --coral: #ff987f;
  --shadow: 0 22px 70px rgba(0,0,0,.23);
}

html { scroll-behavior: smooth; }
html, body, [class*="css"], button, input, textarea, select {
  font-family: 'Manrope', sans-serif !important;
  color: var(--ink);
}
.stApp {
  color: var(--ink);
  background:
    radial-gradient(ellipse at 88% 0%, rgba(96,121,73,.10), transparent 27%),
    radial-gradient(ellipse at 35% 48%, rgba(69,93,115,.055), transparent 40%),
    var(--bg);
}
.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  opacity: .12;
  background-image: radial-gradient(rgba(220,235,232,.18) .55px, transparent .55px);
  background-size: 5px 5px;
  mask-image: linear-gradient(to bottom, black, transparent 82%);
}
.block-container {
  position: relative;
  z-index: 1;
  max-width: 1580px;
  padding: 1.65rem clamp(1.2rem, 3vw, 3.25rem) 3.5rem;
}
[data-testid="stHeader"] {
  background: rgba(11,14,17,.72);
  backdrop-filter: blur(22px);
}
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(14,18,22,.96);
  border-right: 1px solid #20272d;
}
section[data-testid="stSidebar"] > div { padding: 1.35rem .95rem 1.2rem; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .68rem; }
.brand-lockup {
  display: flex; align-items: center; gap: 12px;
  padding: 5px 8px 22px; margin-bottom: 13px;
  border-bottom: 1px solid var(--line);
}
.brand-mark {
  display: grid; place-items: center; flex: 0 0 auto;
  width: 38px; height: 38px; border-radius: 12px;
  color: #10140a; background: var(--lime);
  font: 700 19px 'Space Grotesk', sans-serif;
  box-shadow: 0 0 30px rgba(216,245,108,.09);
}
.brand-name { color: var(--ink); font-size: .92rem; font-weight: 800; letter-spacing: -.035em; }
.brand-caption { color: var(--subtle); font-size: .67rem; margin-top: 3px; letter-spacing: .02em; }
.sidebar-label {
  margin: 4px 0 8px 9px; color: #66727b;
  font: 500 .63rem 'DM Mono', monospace;
  letter-spacing: .13em; text-transform: uppercase;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] {
  display: flex; flex-direction: column; gap: 5px; width: 100%;
  padding: 0; border: 0; background: transparent;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
  display: flex; align-items: center; gap: 12px;
  width: 100%; min-height: 44px; padding: 0 12px !important;
  border: 1px solid transparent !important; border-radius: 10px !important;
  background: transparent !important; color: #8f9aa2 !important;
  font-size: .83rem !important; font-weight: 600 !important;
  transition: background .22s ease, color .22s ease, border-color .22s ease, transform .22s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label::before {
  content: ""; width: 5px; height: 5px; flex: 0 0 5px;
  border-radius: 50%; background: #4c5962; transition: all .22s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
  background: #171e23 !important; color: #f1f4f5 !important; transform: translateX(2px);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
  color: var(--lime) !important; background: rgba(216,245,108,.075) !important;
  border-color: rgba(216,245,108,.12) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"]::before {
  width: 6px; height: 18px; flex-basis: 6px; border-radius: 5px;
  background: var(--lime); box-shadow: 0 0 12px rgba(216,245,108,.28);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
.sidebar-spacer { height: 1px; background: var(--line); margin: 20px 2px 14px; }
.sidebar-meta {
  padding: 14px 12px 13px; border: 1px solid #242d34; border-radius: 13px;
  background: linear-gradient(145deg, rgba(255,255,255,.025), rgba(255,255,255,.008));
}
.sidebar-meta-label { color: #6c7982; font: 500 .61rem 'DM Mono', monospace; text-transform: uppercase; letter-spacing: .1em; }
.sidebar-meta-value { color: #e9eff2; font-size: .81rem; font-weight: 700; margin-top: 5px; }
.sidebar-meta-note { color: #77838b; font-size: .68rem; margin-top: 3px; }
.sidebar-status { display: flex; align-items: center; gap: 8px; color: #6e7d85; font-size: .68rem; margin: 16px 7px 0; }
.sidebar-status span { width: 6px; height: 6px; border-radius: 50%; background: var(--mint); box-shadow: 0 0 0 4px rgba(115,216,180,.09); }

/* Top line and page heading */
.topbar {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  margin: 0 0 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,.075);
  color: #71808a; font: 500 .64rem 'DM Mono', monospace;
  letter-spacing: .105em; text-transform: uppercase;
  animation: enter .55s cubic-bezier(.2,.8,.2,1) both;
}
.topbar-right { display: flex; align-items: center; gap: 9px; color: #9aa7ae; letter-spacing: .025em; }
.live-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--mint); box-shadow: 0 0 0 4px rgba(115,216,180,.085); }
.page-heading { margin: 0 0 26px; animation: enter .62s cubic-bezier(.2,.8,.2,1) both; }
.page-heading-row { display:flex; align-items:flex-end; justify-content:space-between; gap:20px; }
.page-heading h1 {
  margin: 0; color: #f0f4f6; font: 600 clamp(2.1rem, 4vw, 3.4rem)/1.02 'Space Grotesk', sans-serif;
  letter-spacing: -.065em;
}
.page-heading p { color: #87949c; margin: 10px 0 0; font-size: .86rem; line-height: 1.6; }
.page-index {
  display:inline-flex; align-items:center; gap:9px; flex:0 0 auto;
  padding: 8px 10px; border:1px solid #29343b; border-radius:8px;
  background: rgba(255,255,255,.018); color:#7e8b92;
  font: 500 .65rem 'DM Mono',monospace; letter-spacing:.04em;
}
.page-index b { color:var(--lime); font-weight:500; }

/* Panels and section headings */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid #263039 !important; border-radius: 15px !important;
  background: linear-gradient(145deg, rgba(23,30,36,.91), rgba(17,22,27,.94)) !important;
  box-shadow: 0 12px 38px rgba(0,0,0,.09);
  backdrop-filter: blur(18px);
  transition: border-color .25s ease, box-shadow .25s ease, transform .25s ease;
  animation: enter .5s cubic-bezier(.2,.8,.2,1) both;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: #35434b !important; box-shadow: 0 18px 46px rgba(0,0,0,.17);
}
.section-head { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin: 0 0 17px; }
.section-title { color:#e9eef1; font:600 1.08rem/1.3 'Space Grotesk',sans-serif; letter-spacing:-.035em; margin:0; }
.section-sub { color:#75828a; font-size:.75rem; line-height:1.55; margin-top:5px; }
.eyebrow { color:#98a9b0; font:500 .61rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.12em; margin-bottom:8px; }
.section-tag { flex:0 0 auto; color:#7d8b92; border:1px solid #303b43; border-radius:6px; padding:4px 7px; font:500 .59rem 'DM Mono',monospace; }

/* Metric strips */
.metric {
  position:relative; overflow:hidden; min-height:119px; padding:17px 18px 16px;
  border:1px solid #263039; border-radius:13px;
  background:linear-gradient(145deg, rgba(23,30,36,.95), rgba(16,21,26,.94));
  transition: transform .23s ease, border-color .23s ease, background .23s ease;
  animation: enter .55s cubic-bezier(.2,.8,.2,1) both;
}
.metric::before { content:""; position:absolute; left:0; top:17px; bottom:17px; width:2px; border-radius:2px; background:var(--metric-accent,var(--lime)); opacity:.9; }
.metric::after { content:""; position:absolute; width:90px; height:90px; right:-52px; top:-56px; border-radius:50%; background:var(--metric-accent,var(--lime)); filter:blur(37px); opacity:.10; pointer-events:none; }
.metric:hover { transform:translateY(-3px); border-color:#3a474f; background:linear-gradient(145deg, #1a2228, #12181d); }
.metric .k { color:#8d9aa2; font-size:.69rem; font-weight:600; letter-spacing:.01em; }
.metric .v { color:#eff3f5; font:600 clamp(1.65rem,2.3vw,2.15rem)/1.2 'Space Grotesk',sans-serif; letter-spacing:-.06em; margin-top:11px; font-variant-numeric:tabular-nums; }
.metric .s { color:#67757e; font-size:.66rem; line-height:1.45; margin-top:5px; }

/* Prediction result */
.result-shell {
  position:relative; overflow:hidden; min-height:320px; padding:24px;
  border:1px solid #35433f; border-radius:17px;
  background: radial-gradient(ellipse at 90% 8%, rgba(216,245,108,.13), transparent 36%),
              radial-gradient(ellipse at 4% 100%, rgba(115,216,180,.08), transparent 36%),
              linear-gradient(145deg, #151d20, #10161a 72%);
  box-shadow:0 22px 60px rgba(0,0,0,.19);
  animation: enter .65s cubic-bezier(.2,.8,.2,1) both;
}
.result-shell::after { content:""; position:absolute; inset:0; border-radius:inherit; pointer-events:none; background:linear-gradient(115deg,rgba(255,255,255,.025),transparent 34%); }
.result-label { position:relative; z-index:1; color:#9ba9a9; font:500 .64rem 'DM Mono',monospace; letter-spacing:.12em; text-transform:uppercase; }
.result-content { position:relative; z-index:1; display:flex; align-items:center; gap:24px; margin-top:21px; }
.prob-ring {
  position:relative; display:grid; place-items:center; flex:0 0 auto; width:139px; height:139px; border-radius:50%;
  background:conic-gradient(var(--lime) var(--ring-value), #29343a 0deg);
  box-shadow:0 0 35px rgba(216,245,108,.055);
  animation:ringIn .95s cubic-bezier(.16,1,.3,1) both;
}
.prob-ring::before { content:""; position:absolute; inset:8px; border-radius:50%; background:#12191c; border:1px solid rgba(255,255,255,.055); }
.prob-ring-inner { position:relative; z-index:1; text-align:center; }
.prob-ring-value { color:#f2f6ed; font:600 2.1rem/1 'Space Grotesk',sans-serif; letter-spacing:-.07em; }
.prob-ring-unit { color:#78877e; font:500 .61rem 'DM Mono',monospace; margin-top:6px; }
.result-text { min-width:0; }
.result-level { color:var(--lime); font:600 1.14rem/1.3 'Space Grotesk',sans-serif; letter-spacing:-.035em; }
.result-description { max-width:220px; color:#89969b; font-size:.75rem; line-height:1.6; margin-top:8px; }
.result-chip { display:inline-flex; align-items:center; gap:7px; margin-top:17px; border:1px solid #35433f; border-radius:7px; padding:6px 8px; color:#9caaa8; background:rgba(255,255,255,.025); font-size:.64rem; }
.result-chip::before { content:""; width:5px; height:5px; border-radius:50%; background:var(--mint); }
.probability-list { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:19px; position:relative; z-index:1; }
.prob-mini { border:1px solid rgba(255,255,255,.075); border-radius:10px; padding:12px 13px; background:rgba(255,255,255,.025); }
.prob-mini-label { color:#819097; font-size:.67rem; }
.prob-mini-value { color:#edf2f4; font:600 1.2rem 'Space Grotesk',sans-serif; margin-top:4px; }
.empty-result { display:flex; flex-direction:column; justify-content:center; min-height:325px; padding:23px; border:1px dashed #354149; border-radius:15px; background:linear-gradient(145deg,rgba(23,30,36,.57),rgba(13,18,22,.52)); }
.empty-mark { display:grid; place-items:center; width:43px; height:43px; border:1px solid #36434a; border-radius:13px; color:var(--lime); background:rgba(216,245,108,.06); font:500 1.3rem 'Space Grotesk',sans-serif; }
.empty-title { color:#dfe6e8; font:600 1.12rem 'Space Grotesk',sans-serif; letter-spacing:-.03em; margin-top:18px; }
.empty-copy { max-width:250px; color:#78868e; font-size:.76rem; line-height:1.65; margin-top:7px; }

/* Controls */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
  min-height:43px; border-radius:9px !important; border:1px solid #36434a !important;
  background:#1a2228 !important; color:#e5ecef !important; font-size:.79rem !important;
  font-weight:700 !important; transition:transform .2s ease, box-shadow .2s ease, background .2s ease, border-color .2s ease !important;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
  transform:translateY(-2px); background:#242e35 !important; border-color:#64716e !important;
  box-shadow:0 9px 24px rgba(0,0,0,.17); color:#fff !important;
}
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
  background:var(--lime) !important; border-color:var(--lime) !important; color:#151a10 !important;
  box-shadow:0 5px 22px rgba(216,245,108,.09);
}
[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover { background:#e5ff8b !important; border-color:#e5ff8b !important; box-shadow:0 8px 26px rgba(216,245,108,.15); color:#10140a !important; }
[data-testid="stForm"] { border:0; padding:0; }
.stNumberInput input, .stTextInput input, div[data-baseweb="select"] > div {
  border-radius:9px !important; background:#10161b !important; border-color:#2d3840 !important;
  color:#e9eef1 !important; transition:border-color .2s ease, box-shadow .2s ease;
}
.stNumberInput input:focus, .stTextInput input:focus, div[data-baseweb="select"] > div:focus-within {
  border-color:rgba(216,245,108,.65) !important; box-shadow:0 0 0 3px rgba(216,245,108,.08) !important;
}
[data-testid="stWidgetLabel"] p, .stSlider label, .stSelectbox label, .stNumberInput label {
  color:#9ba7ad !important; font-size:.73rem !important; font-weight:600 !important;
}
[data-testid="stCaptionContainer"] { color:#78858d; }
.stRadio label, .stCheckbox label { color:#aab4b8 !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:var(--lime) !important; border-color:var(--lime) !important; }

/* Tables, expanders, charts */
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid #2a343b; border-radius:11px; overflow:hidden; }
[data-testid="stDataFrame"] [role="grid"] { background:#13191e; }
[data-testid="stPlotlyChart"] { transition:transform .24s ease, filter .24s ease; }
[data-testid="stPlotlyChart"]:hover { transform:translateY(-1px); filter:brightness(1.04); }
[data-testid="stAlert"] { border-radius:10px; border:1px solid #34414a; }
[data-testid="stExpander"] { border:1px solid #273139 !important; border-radius:11px !important; background:rgba(18,23,28,.6) !important; }
[data-testid="stExpander"] summary { color:#cbd4d8 !important; }
hr { border-color:#28323a !important; }
.footer {
  display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px 20px;
  margin-top:32px; padding-top:14px; border-top:1px solid rgba(255,255,255,.07);
  color:#5f6d75; font:400 .63rem 'DM Mono',monospace; letter-spacing:.015em;
}
.footer strong { color:#8b989f; font-weight:500; }
.equation {
  padding:20px 22px; border:1px solid #29353b; border-radius:12px;
  background:linear-gradient(110deg,rgba(216,245,108,.045),rgba(115,216,180,.025));
  color:#dfe8e9; font:500 clamp(1.05rem,2.2vw,1.55rem)/1.5 'Space Grotesk',sans-serif; letter-spacing:-.035em;
}
.conclusion { padding:15px 17px; border-radius:10px; border:1px solid #2e3d37; background:rgba(115,216,180,.045); color:#c8d8d4; font-size:.82rem; line-height:1.7; }

@keyframes enter { from { opacity:0; transform:translateY(9px); } to { opacity:1; transform:translateY(0); } }
@keyframes ringIn { from { opacity:.3; transform:scale(.9) rotate(-16deg); } to { opacity:1; transform:scale(1) rotate(0deg); } }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; transition-duration:.01ms !important; }
}
@media (max-width: 850px) {
  .block-container { padding:1.1rem 1rem 2rem; }
  .page-heading-row { align-items:flex-start; }
  .page-index { display:none; }
  .result-content { gap:16px; }
  .prob-ring { width:112px; height:112px; }
  .prob-ring-value { font-size:1.7rem; }
  .result-shell { padding:20px; }
}
@media (max-width: 520px) {
  .result-content { align-items:flex-start; flex-direction:column; }
  .probability-list { grid-template-columns:1fr 1fr; }
  .topbar-right { font-size:.58rem; }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def fmt_pct(value):
    return f"{value * 100:.2f}%"


def card_metric(label, value, caption, accent="lime"):
    accents = {"lime": "#d8f56c", "mint": "#73d8b4", "blue": "#8caeff", "coral": "#ff987f"}
    color = accents.get(accent, accents["lime"])
    st.markdown(
        f'<div class="metric" style="--metric-accent:{color}">'
        f'<div class="k">{html.escape(str(label))}</div>'
        f'<div class="v">{html.escape(str(value))}</div>'
        f'<div class="s">{html.escape(str(caption))}</div></div>',
        unsafe_allow_html=True,
    )


def section_heading(title, subtitle=None, eyebrow=None, tag=None):
    eyebrow_html = f'<div class="eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    subtitle_html = f'<div class="section-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    tag_html = f'<div class="section-tag">{html.escape(tag)}</div>' if tag else ""
    st.markdown(
        f'<div class="section-head"><div>{eyebrow_html}<div class="section-title">{html.escape(title)}</div>{subtitle_html}</div>{tag_html}</div>',
        unsafe_allow_html=True,
    )


def plotly_theme(fig, height=370):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color="#a8b3b8", size=11),
        title=dict(font=dict(family="Space Grotesk, sans-serif", size=15, color="#e6ecee"), x=0.02, xanchor="left"),
        margin=dict(l=12, r=16, t=48, b=12),
        hoverlabel=dict(bgcolor="#202a30", bordercolor="#3c494f", font_color="#f1f5f5", font_family="Manrope"),
        legend=dict(
            bgcolor="rgba(16,22,26,.72)", bordercolor="rgba(255,255,255,.07)", borderwidth=1,
            font=dict(size=10, color="#b6c0c4"),
        ),
        transition=dict(duration=420, easing="cubic-in-out"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(255,255,255,.10)", tickfont=dict(color="#819098"), title_font=dict(color="#a6b1b6"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,.065)", zeroline=False, tickfont=dict(color="#819098"), title_font=dict(color="#a6b1b6"))
    return fig


def page_header(title, subtitle, index, section):
    st.markdown(
        f'<div class="topbar"><span>DIABETES STUDY&nbsp;&nbsp; / &nbsp;&nbsp;{html.escape(section.upper())}</span>'
        f'<span class="topbar-right"><span class="live-dot"></span>{len(df):,} records</span></div>'
        f'<div class="page-heading"><div class="page-heading-row"><div><h1>{html.escape(title)}</h1>'
        f'<p>{html.escape(subtitle)}</p></div><div class="page-index"><b>{index}</b><span>/ 05</span></div></div></div>',
        unsafe_allow_html=True,
    )


def add_footer():
    st.markdown(
        '<div class="footer"><span>DIABETES STUDY <strong>·</strong> Statistical analysis workspace</span>'
        '<span>Educational use only <strong>·</strong> Not a medical diagnosis</span></div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Session state and first-step sex selection
# -----------------------------------------------------------------------------
if "gender_selected" not in st.session_state:
    st.session_state.gender_selected = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if st.session_state.gender_selected is None:
    st.markdown(
        '<div class="topbar"><span>DIABETES STUDY&nbsp;&nbsp; / &nbsp;&nbsp;PROFILE SETUP</span>'
        '<span class="topbar-right"><span class="live-dot"></span>Dataset available</span></div>'
        '<div class="page-heading"><div class="page-heading-row"><div><h1>Diabetes Study</h1>'
        '<p>Probability estimates and statistical analysis.</p></div></div></div>',
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([.65, 1.45, .65])
    with center:
        with st.container(border=True):
            section_heading("Select sex", "This field is used by the probability model.", "Profile setup", "STEP 01")
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                if st.button("Male", use_container_width=True, key="start_male"):
                    st.session_state.gender_selected = "Male"
                    st.rerun()
            with c2:
                if st.button("Female", use_container_width=True, key="start_female"):
                    st.session_state.gender_selected = "Female"
                    st.rerun()
    add_footer()
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><div class="brand-mark">D</div><div>'
        '<div class="brand-name">Diabetes Study</div><div class="brand-caption">Probability / statistics</div>'
        '</div></div><div class="sidebar-label">Workspace</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Workspace",
        ["Prediction", "Probability", "Statistics", "ANOVA", "Visuals"],
        label_visibility="collapsed",
        key="workspace_page",
    )
    st.markdown(
        '<div class="sidebar-spacer"></div>'
        f'<div class="sidebar-meta"><div class="sidebar-meta-label">Current profile</div>'
        f'<div class="sidebar-meta-value">{html.escape(st.session_state.gender_selected)}</div>'
        f'<div class="sidebar-meta-note">Active model input</div></div>'
        f'<div style="height:9px"></div>'
        f'<div class="sidebar-meta"><div class="sidebar-meta-label">Dataset</div>'
        f'<div class="sidebar-meta-value">{len(df):,} <span style="font-size:.7rem;font-weight:500;color:#78858d">rows</span></div>'
        f'<div class="sidebar-meta-note">All records used in analysis</div></div>'
        '<div class="sidebar-status"><span></span>Model and dataset ready</div>',
        unsafe_allow_html=True,
    )

page_indexes = {"Prediction": "01", "Probability": "02", "Statistics": "03", "ANOVA": "04", "Visuals": "05"}

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
if page == "Prediction":
    page_header("Patient profile", "Enter the measurements used by the probability model.", page_indexes[page], "Prediction")
    left, right = st.columns([1.02, .98], gap="large")

    with left:
        with st.container(border=True):
            section_heading("Patient details", "Set the input variables for this estimate.", "Input variables", "8 FEATURES")
            with st.form("patient_form"):
                a, b = st.columns(2, gap="medium")
                with a:
                    age = st.number_input("Age", min_value=0.1, max_value=100.0, value=40.0, step=1.0)
                    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=27.3, step=0.1, format="%.1f")
                    hba1c = st.number_input("HbA1c level", min_value=3.0, max_value=10.0, value=5.7, step=0.1, format="%.1f")
                with b:
                    glucose = st.number_input("Blood glucose level", min_value=40, max_value=300, value=120, step=1)
                    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
                    heart = st.selectbox("Heart disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
                smoking = st.selectbox("Smoking history", ["never", "No Info", "former", "current", "not current", "ever"], index=0)
                submitted = st.form_submit_button("Calculate probability", use_container_width=True, type="primary")

        col_a, col_b = st.columns([1, 1], gap="small")
        with col_a:
            if st.button("Change sex", key="change_gender", use_container_width=True):
                st.session_state.gender_selected = None
                st.session_state.prediction = None
                st.rerun()
        with col_b:
            st.markdown('<div style="padding:12px 0 0;color:#65727a;font-size:.67rem;text-align:right">Profile: ' + html.escape(st.session_state.gender_selected) + '</div>', unsafe_allow_html=True)

    with right:
        if submitted:
            sample = {
                "gender": st.session_state.gender_selected,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart,
                "smoking_history": smoking,
                "bmi": bmi,
                "HbA1c_level": hba1c,
                "blood_glucose_level": glucose,
            }
            p1, p0 = predictor.predict(sample)
            st.session_state.prediction = (sample, p1, p0)

        if st.session_state.prediction:
            sample, p1, p0 = st.session_state.prediction
            if p1 < .10:
                level = "Lower probability"
                description = "The model estimates a lower probability from the supplied measurements."
            elif p1 < .30:
                level = "Elevated probability"
                description = "The model estimates an elevated probability from the supplied measurements."
            elif p1 < .60:
                level = "Higher probability"
                description = "The model estimates a higher probability from the supplied measurements."
            else:
                level = "Very high probability"
                description = "The model estimates a very high probability from the supplied measurements."
            degrees = max(0, min(360, p1 * 360))
            st.markdown(
                f'<div class="result-shell"><div class="result-label">Estimated diabetes probability</div>'
                f'<div class="result-content"><div class="prob-ring" style="--ring-value:{degrees:.2f}deg">'
                f'<div class="prob-ring-inner"><div class="prob-ring-value">{p1 * 100:.1f}<span style="font-size:.9rem;letter-spacing:-.02em">%</span></div>'
                f'<div class="prob-ring-unit">POSTERIOR</div></div></div>'
                f'<div class="result-text"><div class="result-level">{html.escape(level)}</div>'
                f'<div class="result-description">{html.escape(description)}</div>'
                f'<div class="result-chip">{html.escape(sample["gender"])} · Bayesian estimate</div></div></div>'
                f'<div class="probability-list"><div class="prob-mini"><div class="prob-mini-label">Diabetes</div><div class="prob-mini-value">{fmt_pct(p1)}</div></div>'
                f'<div class="prob-mini"><div class="prob-mini-label">Non-diabetes</div><div class="prob-mini-value">{fmt_pct(p0)}</div></div></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:13px'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                section_heading("Probability split", "Posterior probabilities returned by the model.", "Model output")
                fig = go.Figure(go.Bar(
                    x=[p1 * 100, p0 * 100], y=["Diabetes", "Non-diabetes"], orientation="h",
                    text=[fmt_pct(p1), fmt_pct(p0)], textposition="outside", cliponaxis=False,
                    marker=dict(color=["#d8f56c", "#45535c"], line=dict(width=0)),
                ))
                fig.update_layout(xaxis_title="Probability (%)", yaxis_title="", xaxis=dict(range=[0, 110]), showlegend=False, bargap=.52)
                st.plotly_chart(plotly_theme(fig, 225), use_container_width=True, config={"displayModeBar": False})
            st.caption("Educational estimate only. This output is not a diagnosis or validated clinical risk score.")
        else:
            st.markdown(
                '<div class="empty-result"><div class="empty-mark">%</div><div class="eyebrow" style="margin-top:20px">Model output</div>'
                '<div class="empty-title">Probability estimate</div>'
                '<div class="empty-copy">Enter the profile variables and calculate to view the posterior probability.</div>'
                '<div style="height:32px;max-width:280px;margin-top:22px;background:linear-gradient(90deg,#29343a 0 30%,#1d272d 30% 100%);border-radius:4px;overflow:hidden"><div style="width:24%;height:100%;background:#65727a;opacity:.45"></div></div></div>',
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# Conditional probability
# -----------------------------------------------------------------------------
elif page == "Probability":
    page_header("Probability", "Conditional probabilities calculated from the dataset.", page_indexes[page], "Probability")
    overall = df.diabetes.mean()
    male = df.loc[df.gender == "Male", "diabetes"].mean()
    female = df.loc[df.gender == "Female", "diabetes"].mean()

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: card_metric("P(Diabetes)", fmt_pct(overall), "Full dataset", "lime")
    with c2: card_metric("P(D | Male)", fmt_pct(male), "Conditional on sex", "mint")
    with c3: card_metric("P(D | Female)", fmt_pct(female), "Conditional on sex", "blue")
    with c4: card_metric("Observations", f"{len(df):,}", "Records analysed", "coral")

    st.markdown("<div style='height:17px'></div>", unsafe_allow_html=True)
    l, r = st.columns(2, gap="large")
    with l:
        with st.container(border=True):
            section_heading("Probability by sex", "Share of each sex group with diabetes.", "Comparison", "P(D | GENDER)")
            gender_rates = df[df.gender.isin(["Male", "Female"])].groupby("gender", as_index=False).diabetes.mean()
            gender_rates["rate"] = gender_rates.diabetes * 100
            fig = px.bar(gender_rates, x="gender", y="rate", text=gender_rates["rate"].map(lambda x: f"{x:.2f}%"),
                         labels={"gender": "Sex", "rate": "Probability (%)"}, color="gender",
                         color_discrete_map={"Male": "#d8f56c", "Female": "#73d8b4"})
            fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0, width=.42)
            fig.update_layout(showlegend=False, yaxis_title="Probability (%)", xaxis_title="", yaxis=dict(range=[0, max(gender_rates.rate) * 1.22]))
            st.plotly_chart(plotly_theme(fig), use_container_width=True, config={"displayModeBar": False})
    with r:
        with st.container(border=True):
            section_heading("Existing conditions", "Observed diabetes rates by condition status.", "Comparison", "CONDITIONAL RATE")
            risk_rows = []
            for feature, label in [("hypertension", "Hypertension"), ("heart_disease", "Heart disease")]:
                for value in [0, 1]:
                    probability = df[df[feature] == value].diabetes.mean()
                    risk_rows.append({"Factor": label, "State": "Yes" if value else "No", "Probability": probability * 100})
            risk_frame = pd.DataFrame(risk_rows)
            fig = px.bar(risk_frame, x="Factor", y="Probability", color="State", barmode="group",
                         labels={"Probability": "P(Diabetes | condition) (%)", "Factor": "", "State": "Condition"},
                         color_discrete_map={"No": "#56666f", "Yes": "#ff987f"})
            fig.update_layout(legend_title_text="")
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(plotly_theme(fig), use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        section_heading("Bayes’ rule", "The model combines the prior probability with the likelihood of observed features.", "Method")
        st.markdown('<div class="equation">P(D | X) ∝ P(D) × P(Gender | D) × P(Age | D) × ···</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub" style="margin-top:12px">Numerical variables use Gaussian likelihoods. Categorical variables use smoothed frequency likelihoods. Calculations are performed in log-space for numerical stability.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Descriptive statistics and chi-square tests
# -----------------------------------------------------------------------------
elif page == "Statistics":
    page_header("Statistics", "Descriptive measures, confidence intervals and independence tests.", page_indexes[page], "Statistics")
    n = len(df)
    successes = int(df.diabetes.sum())
    p, lo, hi = proportion_ci(successes, n)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1: card_metric("Diabetes prevalence", fmt_pct(p), f"95% CI · {lo * 100:.2f}%–{hi * 100:.2f}%", "lime")
    with c2: card_metric("Mean age", f"{df.age.mean():.1f}", f"SD {df.age.std():.1f} years", "mint")
    with c3: card_metric("Mean glucose", f"{df.blood_glucose_level.mean():.1f}", f"SD {df.blood_glucose_level.std():.1f}", "blue")

    st.markdown("<div style='height:17px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        section_heading("Descriptive summary", "Numeric variables across the full dataset.", "Overview", "N = ALL ROWS")
        summary = df[["age", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"]].describe().T.round(3)
        st.dataframe(summary, use_container_width=True)

    st.markdown("<div style='height:17px'></div>", unsafe_allow_html=True)
    section_heading("Chi-square independence tests", "Association between each categorical feature and diabetes.", "Tests")
    for feature, label in [("gender", "Sex"), ("hypertension", "Hypertension"), ("heart_disease", "Heart disease"), ("smoking_history", "Smoking history")]:
        table, chi2, pval, dof = chi_square(df, feature)
        significance = "Significant at α = 0.05" if pval < 0.05 else "Not significant at α = 0.05"
        with st.expander(f"{label}  ·  χ² = {chi2:.3f}  ·  p = {pval:.3e}", expanded=(feature == "gender")):
            st.markdown(f'<div class="section-sub" style="margin:0 0 9px">Degrees of freedom: {dof} &nbsp; / &nbsp; {html.escape(significance)}</div>', unsafe_allow_html=True)
            st.dataframe(table, use_container_width=True)

# -----------------------------------------------------------------------------
# One-way ANOVA
# -----------------------------------------------------------------------------
elif page == "ANOVA":
    page_header("One-way ANOVA", "Compare a numerical measure across groups.", page_indexes[page], "ANOVA")
    nums = {"Age": "age", "BMI": "bmi", "HbA1c level": "HbA1c_level", "Blood glucose level": "blood_glucose_level"}
    groups = {"Sex": "gender", "Smoking history": "smoking_history", "Hypertension": "hypertension", "Heart disease": "heart_disease"}

    with st.container(border=True):
        section_heading("Test configuration", "Choose a numeric variable and a grouping variable.", "Parameters", "ONE-WAY")
        a, b = st.columns(2, gap="medium")
        with a: numerical_label = st.selectbox("Numerical variable", list(nums), index=2)
        with b: group_label = st.selectbox("Grouping variable", list(groups), index=0)
        min_size = st.slider("Minimum observations per group", 2, 100, 20)

    try:
        fstat, pval, dfb, dfw, eta, summary = one_way_anova(df, nums[numerical_label], groups[group_label], min_size)
        significant = pval < 0.05
        conclusion = (
            f"There is statistically significant evidence that mean {numerical_label.lower()} differs across {group_label.lower()} groups."
            if significant else
            f"There is not enough evidence at α = 0.05 to say that mean {numerical_label.lower()} differs across {group_label.lower()} groups."
        )

        st.markdown("<div style='height:17px'></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        with m1: card_metric("F statistic", f"{fstat:.3f}", f"df {dfb} / {dfw}", "lime")
        with m2: card_metric("p-value", f"{pval:.3e}", "α = 0.05", "mint")
        with m3: card_metric("η² effect size", f"{eta:.3f}", "Variance explained", "blue")
        with m4: card_metric("Groups", str(len(summary)), "Included in test", "coral")

        with st.container(border=True):
            section_heading("Interpretation", "H₀: all group means are equal. H₁: at least one mean differs.", "Test result", "α = 0.05")
            st.markdown(f'<div class="conclusion">{html.escape(conclusion)}</div>', unsafe_allow_html=True)

        l, r = st.columns([.95, 1.2], gap="large")
        with l:
            with st.container(border=True):
                section_heading("Group summary", "Sample size, mean and standard deviation.", "Output")
                st.dataframe(summary.round({"Mean": 3, "SD": 3}), use_container_width=True, hide_index=True)
        with r:
            with st.container(border=True):
                section_heading(f"{numerical_label} by {group_label.lower()}", "Distribution across selected groups.", "Distribution")
                plot = df[[nums[numerical_label], groups[group_label]]].dropna()
                if len(plot) > 12000:
                    plot = plot.sample(12000, random_state=42)
                fig = px.box(plot, x=groups[group_label], y=nums[numerical_label], points=False,
                             color_discrete_sequence=["#d8f56c"], labels={nums[numerical_label]: numerical_label, groups[group_label]: group_label})
                fig.update_traces(line_color="#d8f56c", fillcolor="rgba(216,245,108,.13)")
                st.plotly_chart(plotly_theme(fig, 400), use_container_width=True, config={"displayModeBar": False})
    except Exception as exc:
        st.error(f"ANOVA could not be calculated: {exc}")

# -----------------------------------------------------------------------------
# Dataset visualisations
# -----------------------------------------------------------------------------
else:
    page_header("Visual analysis", "Distributions and relationships in the dataset.", page_indexes[page], "Visuals")
    visual1, visual2 = st.columns(2, gap="large")
    with visual1:
        with st.container(border=True):
            section_heading("Blood glucose distribution", "Distribution split by diabetes outcome.", "01 / Distribution")
            fig = px.histogram(df, x="blood_glucose_level", color="diabetes", nbins=45, marginal="box",
                               labels={"blood_glucose_level": "Blood glucose", "diabetes": "Diabetes"},
                               color_discrete_map={0: "#4b5962", 1: "#d8f56c"}, opacity=.88)
            fig.update_layout(legend_title_text="Diabetes")
            st.plotly_chart(plotly_theme(fig, 390), use_container_width=True, config={"displayModeBar": False})
    with visual2:
        with st.container(border=True):
            section_heading("BMI vs blood glucose", "Patient-level relationship, sampled for readability.", "02 / Relationship")
            scatter_sample = df.sample(min(5000, len(df)), random_state=7)
            fig = px.scatter(scatter_sample, x="bmi", y="blood_glucose_level", color="diabetes", opacity=.58,
                             labels={"bmi": "BMI", "blood_glucose_level": "Blood glucose", "diabetes": "Diabetes"},
                             color_discrete_map={0: "#6c7c85", 1: "#d8f56c"})
            fig.update_traces(marker=dict(size=5, line=dict(width=0)))
            st.plotly_chart(plotly_theme(fig, 390), use_container_width=True, config={"displayModeBar": False})

    visual3, visual4 = st.columns(2, gap="large")
    with visual3:
        with st.container(border=True):
            section_heading("Smoking history", "Observed diabetes percentage in each category.", "03 / Conditional rate")
            smoking_rates = df.groupby("smoking_history", as_index=False).diabetes.mean()
            smoking_rates["rate"] = smoking_rates.diabetes * 100
            smoking_rates = smoking_rates.sort_values("rate")
            fig = px.bar(smoking_rates, x="rate", y="smoking_history", orientation="h",
                         text=smoking_rates["rate"].map(lambda x: f"{x:.2f}%"),
                         labels={"rate": "Probability (%)", "smoking_history": "Smoking history"})
            fig.update_traces(marker_color="#73d8b4", textposition="outside", cliponaxis=False, marker_line_width=0)
            fig.update_layout(xaxis=dict(range=[0, max(smoking_rates.rate) * 1.2]))
            st.plotly_chart(plotly_theme(fig, 350), use_container_width=True, config={"displayModeBar": False})
    with visual4:
        with st.container(border=True):
            section_heading("Numerical correlation", "Pearson correlation between numeric variables.", "04 / Correlation")
            correlation = df[["age", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"]].corr()
            fig = px.imshow(correlation, text_auto=".2f", aspect="auto",
                            color_continuous_scale=["#182027", "#5b7a79", "#d8f56c"], zmin=-1, zmax=1)
            fig.update_traces(textfont=dict(color="#edf2f2", size=10))
            fig.update_layout(coloraxis_colorbar=dict(tickfont=dict(color="#a3afb4"), outlinewidth=0, thickness=8))
            st.plotly_chart(plotly_theme(fig, 350), use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        section_heading("Dataset sample", "Twenty rows for inspection. Statistical calculations use the full dataset.", "Raw records", "20 ROWS")
        st.dataframe(df.sample(20, random_state=42), use_container_width=True, hide_index=True)

add_footer()
