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
# Visual system
# -----------------------------------------------------------------------------
st.markdown(
r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');

:root {
    --ink: #252821;
    --muted: #77796f;
    --paper: #f5f3ed;
    --panel: rgba(255, 254, 250, .72);
    --panel-strong: rgba(255, 254, 250, .9);
    --line: rgba(43, 46, 38, .11);
    --line-strong: rgba(43, 46, 38, .18);
    --accent: #b76545;
    --accent-dark: #8e482f;
    --green: #2f5549;
    --green-dark: #233f36;
    --shadow: 0 18px 50px rgba(39, 39, 30, .055);
}

html, body, [class*="css"], button, input, textarea, select {
    font-family: 'Instrument Sans', sans-serif;
    color: var(--ink);
}
html { scroll-behavior: smooth; }
.stApp {
    color: var(--ink);
    background:
      radial-gradient(ellipse at 78% 2%, rgba(202, 178, 145, .17), transparent 36%),
      radial-gradient(ellipse at 5% 36%, rgba(166, 187, 174, .12), transparent 32%),
      var(--paper);
}
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: .22;
    background-image: radial-gradient(rgba(38, 41, 33, .16) .45px, transparent .45px);
    background-size: 5px 5px;
    mask-image: linear-gradient(to bottom, black, transparent 85%);
}
.block-container {
    position: relative;
    z-index: 1;
    max-width: 1390px;
    padding: 2.15rem clamp(1.2rem, 3vw, 3.25rem) 4rem;
}
[data-testid="stHeader"] {
    background: rgba(245, 243, 237, .72);
    backdrop-filter: blur(20px);
}
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* Sidebar workspace */
section[data-testid="stSidebar"] {
    background: rgba(248, 247, 242, .78);
    border-right: 1px solid var(--line);
    backdrop-filter: blur(24px);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.6rem;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: .8rem;
}
.brand-lockup {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 4px 6px 22px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 5px;
}
.brand-mark {
    width: 30px;
    height: 30px;
    border-radius: 10px;
    display: grid;
    place-items: center;
    color: #fbf8f1;
    background: var(--green-dark);
    font: 600 16px 'Newsreader', serif;
    box-shadow: 0 5px 14px rgba(35, 63, 54, .16);
}
.brand-name { font-size: .93rem; font-weight: 700; letter-spacing: -.025em; }
.brand-caption { color: var(--muted); font-size: .68rem; margin-top: 1px; }
.sidebar-label {
    margin: 5px 0 1px 7px;
    color: #96958a;
    font-size: .63rem;
    font-weight: 700;
    letter-spacing: .15em;
    text-transform: uppercase;
}
section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 4px;
    width: 100%;
    padding: 0;
    border: 0;
    background: transparent;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    display: flex;
    align-items: center;
    gap: 11px;
    width: 100%;
    min-height: 43px;
    padding: 0 12px !important;
    border: 1px solid transparent !important;
    border-radius: 11px !important;
    background: transparent !important;
    color: #686b61 !important;
    font-size: .88rem !important;
    transition: background .22s ease, color .22s ease, border-color .22s ease, transform .22s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(255,255,255,.62) !important;
    color: var(--ink) !important;
    transform: translateX(2px);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
    color: var(--green-dark) !important;
    background: rgba(47, 85, 73, .085) !important;
    border-color: rgba(47, 85, 73, .10) !important;
    font-weight: 700;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
.sidebar-foot {
    border-top: 1px solid var(--line);
    padding: 16px 7px 0;
    margin-top: 18px;
    color: var(--muted);
    font-size: .75rem;
    line-height: 1.55;
}
.sidebar-foot strong { color: var(--ink); font-weight: 600; }

/* Page header */
.page-topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin: 0 0 24px;
    color: var(--muted);
    font-size: .69rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    animation: riseIn .55s cubic-bezier(.2,.75,.25,1) both;
}
.topline-right {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #73766b;
    font-size: .72rem;
    font-weight: 500;
    letter-spacing: 0;
    text-transform: none;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #65896d;
    box-shadow: 0 0 0 4px rgba(101,137,109,.10);
}
.page-heading {
    margin: 0 0 26px;
    padding-bottom: 25px;
    border-bottom: 1px solid var(--line);
    animation: riseIn .58s cubic-bezier(.2,.75,.25,1) both;
}
.page-heading h1 {
    margin: 0;
    color: #292b25;
    font: 500 clamp(2.25rem, 4.7vw, 3.7rem)/.99 'Newsreader', serif;
    letter-spacing: -.055em;
}
.page-heading p {
    color: var(--muted);
    margin: 12px 0 0;
    font-size: .93rem;
    line-height: 1.6;
}
.eyebrow {
    color: var(--accent-dark);
    font-size: .64rem;
    font-weight: 700;
    letter-spacing: .15em;
    text-transform: uppercase;
    margin-bottom: 9px;
}
.section-title {
    color: var(--ink);
    font: 500 1.45rem/1.15 'Newsreader', serif;
    letter-spacing: -.025em;
    margin: 2px 0 5px;
}
.section-sub {
    color: var(--muted);
    font-size: .82rem;
    line-height: 1.55;
    margin-bottom: 16px;
}

/* Glass panels */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(50, 52, 43, .105) !important;
    border-radius: 19px !important;
    background: var(--panel) !important;
    box-shadow: 0 11px 38px rgba(35, 37, 29, .035);
    backdrop-filter: blur(18px);
    transition: border-color .25s ease, box-shadow .25s ease, transform .25s ease;
    animation: riseIn .48s cubic-bezier(.2,.75,.25,1) both;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(47, 85, 73, .2) !important;
    box-shadow: 0 16px 42px rgba(35, 37, 29, .055);
}
.metric {
    min-height: 120px;
    padding: 20px 21px 18px;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: rgba(255, 254, 250, .72);
    box-shadow: 0 9px 26px rgba(35, 37, 29, .025);
    backdrop-filter: blur(16px);
    transition: transform .25s cubic-bezier(.2,.75,.25,1), box-shadow .25s ease, border-color .25s ease;
    animation: riseIn .48s cubic-bezier(.2,.75,.25,1) both;
}
.metric:hover {
    transform: translateY(-3px);
    border-color: rgba(47, 85, 73, .2);
    box-shadow: 0 16px 32px rgba(35, 37, 29, .065);
}
.metric .k {
    color: #77796f;
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: .11em;
    text-transform: uppercase;
}
.metric .v {
    color: #282b25;
    font: 500 2rem/.98 'Newsreader', serif;
    letter-spacing: -.045em;
    margin-top: 11px;
}
.metric .s { color: #8b8b81; font-size: .72rem; margin-top: 7px; line-height: 1.4; }
.result {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    padding: 28px;
    border-radius: 19px;
    border: 1px solid rgba(255,255,255,.15);
    background: linear-gradient(138deg, #2d5146 0%, #27463d 55%, #203b33 100%);
    color: #fbf8f0;
    box-shadow: 0 18px 42px rgba(31, 62, 51, .15);
    animation: riseIn .58s cubic-bezier(.2,.75,.25,1) both;
}
.result::before {
    content: "";
    position: absolute;
    z-index: -1;
    width: 300px;
    height: 300px;
    top: -170px;
    right: -105px;
    border: 1px solid rgba(255,255,255,.17);
    border-radius: 50%;
    box-shadow: 0 0 0 25px rgba(255,255,255,.025), 0 0 0 58px rgba(255,255,255,.018), 0 0 0 91px rgba(255,255,255,.012);
    animation: slowOrbit 18s linear infinite;
}
.result .small {
    color: rgba(251,248,240,.67);
    font-size: .66rem;
    font-weight: 700;
    letter-spacing: .13em;
    text-transform: uppercase;
}
.result .big {
    margin: 17px 0 12px;
    font: 400 clamp(3.6rem, 7vw, 5.5rem)/.86 'Newsreader', serif;
    letter-spacing: -.07em;
    color: #fffaf0;
}
.result .label { color: rgba(251,248,240,.78); font-size: .86rem; line-height: 1.5; }
.pill {
    display: inline-block;
    margin-top: 20px;
    padding: 7px 10px;
    border: 1px solid rgba(255,255,255,.15);
    border-radius: 999px;
    background: rgba(255,255,255,.075);
    color: rgba(255,255,255,.85);
    font-size: .71rem;
}
.form-kicker {
    color: var(--muted);
    font-size: .75rem;
    margin: 1px 0 14px;
}

/* Inputs and actions */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    min-height: 43px;
    border-radius: 11px;
    border: 1px solid rgba(43,46,38,.13);
    background: rgba(255,254,250,.75);
    color: var(--ink);
    font-size: .84rem;
    font-weight: 600;
    transition: transform .22s cubic-bezier(.2,.75,.25,1), box-shadow .22s ease, background .22s ease, border-color .22s ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(35,37,29,.09);
    border-color: rgba(47,85,73,.3);
    background: #fffefa;
    color: var(--green-dark);
}
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
    background: var(--green-dark);
    border-color: var(--green-dark);
    color: #fffdf7;
}
[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
    background: #31594b;
    border-color: #31594b;
    color: white;
}
div[data-testid="stForm"] { border: 0; padding: 0; }
.stNumberInput input, .stTextInput input, div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    background: rgba(255,254,250,.78) !important;
    border-color: rgba(43,46,38,.15) !important;
    transition: border-color .2s ease, box-shadow .2s ease;
}
.stNumberInput input:focus, .stTextInput input:focus, div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(47,85,73,.55) !important;
    box-shadow: 0 0 0 3px rgba(47,85,73,.08) !important;
}
[data-testid="stWidgetLabel"] p, .stSlider label, .stSelectbox label, .stNumberInput label {
    color: #55584f;
    font-size: .79rem;
    font-weight: 600;
}
[data-testid="stCaptionContainer"] { color: var(--muted); }
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--line);
    border-radius: 13px;
    overflow: hidden;
}
[data-testid="stPlotlyChart"] { transition: transform .25s ease; }
[data-testid="stPlotlyChart"]:hover { transform: translateY(-1px); }
[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: var(--line); }
.footer {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px 20px;
    margin-top: 34px;
    padding-top: 15px;
    border-top: 1px solid var(--line);
    color: #939287;
    font-size: .7rem;
    line-height: 1.5;
}

@keyframes riseIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes slowOrbit { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; scroll-behavior: auto !important; transition-duration: .01ms !important; }
}
@media (max-width: 760px) {
    .block-container { padding: 1.35rem 1rem 2.5rem; }
    .page-heading { margin-bottom: 19px; padding-bottom: 19px; }
    .page-heading h1 { font-size: 2.65rem; }
    .result { padding: 23px; }
    .metric { padding: 16px; min-height: 108px; }
    .metric .v { font-size: 1.7rem; }
    .brand-lockup { padding-bottom: 16px; }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def fmt_pct(value):
    return f"{value * 100:.2f}%"


def card_metric(label, value, caption):
    st.markdown(
        f'<div class="metric"><div class="k">{html.escape(str(label))}</div>'
        f'<div class="v">{html.escape(str(value))}</div>'
        f'<div class="s">{html.escape(str(caption))}</div></div>',
        unsafe_allow_html=True,
    )


def section_heading(title, subtitle=None, eyebrow=None):
    eyebrow_html = f'<div class="eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    subtitle_html = f'<div class="section-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'{eyebrow_html}<div class="section-title">{html.escape(title)}</div>{subtitle_html}',
        unsafe_allow_html=True,
    )


def plotly_theme(fig, height=410):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans, sans-serif", color="#55584f", size=12),
        title=dict(font=dict(family="Newsreader, serif", size=21, color="#292b25"), x=0.02, xanchor="left"),
        margin=dict(l=12, r=16, t=58, b=12),
        hoverlabel=dict(bgcolor="#252821", font_color="#fffdf7", font_family="Instrument Sans"),
        legend=dict(
            bgcolor="rgba(255,254,250,.65)",
            bordercolor="rgba(43,46,38,.08)",
            borderwidth=1,
            font=dict(size=11),
        ),
        transition=dict(duration=450, easing="cubic-in-out"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(43,46,38,.12)", tickfont=dict(color="#77796f"))
    fig.update_yaxes(gridcolor="rgba(43,46,38,.07)", zeroline=False, tickfont=dict(color="#77796f"))
    return fig


def page_header(title, subtitle, index):
    st.markdown(
        f'<div class="page-topline"><span>DIABETES STUDY / {index}</span>'
        f'<span class="topline-right"><span class="status-dot"></span>{len(df):,} records loaded</span></div>'
        f'<div class="page-heading"><h1>{html.escape(title)}</h1>'
        f'<p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Session state and first-step gender selection
# -----------------------------------------------------------------------------
if "gender_selected" not in st.session_state:
    st.session_state.gender_selected = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if st.session_state.gender_selected is None:
    st.markdown(
        '<div class="page-topline"><span>DIABETES STUDY</span>'
        '<span class="topline-right"><span class="status-dot"></span>Probability & statistics</span></div>'
        '<div class="page-heading"><h1>Diabetes study</h1>'
        '<p>Probability estimates and statistical analysis across the dataset.</p></div>',
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([0.65, 1.7, 0.65])
    with center:
        with st.container(border=True):
            section_heading("Select sex", "Used as the first categorical variable in the probability model.", "Patient profile")
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                if st.button("Male", use_container_width=True, key="male_start"):
                    st.session_state.gender_selected = "Male"
                    st.rerun()
            with col2:
                if st.button("Female", use_container_width=True, key="female_start"):
                    st.session_state.gender_selected = "Female"
                    st.rerun()
        st.markdown(
            '<div class="footer"><span>Statistical and educational use only.</span>'
            '<span>Not a medical diagnosis.</span></div>',
            unsafe_allow_html=True,
        )
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><div class="brand-mark">D</div><div>'
        '<div class="brand-name">Diabetes Study</div><div class="brand-caption">Probability & statistics</div>'
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
        f'<div class="sidebar-foot"><strong>Current profile</strong><br>{html.escape(st.session_state.gender_selected)}'
        f'<br><br><strong>Dataset</strong><br>{len(df):,} observations</div>',
        unsafe_allow_html=True,
    )

page_indexes = {"Prediction": "01", "Probability": "02", "Statistics": "03", "ANOVA": "04", "Visuals": "05"}

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
if page == "Prediction":
    page_header("Patient profile", "Enter the measurements used by the probability model.", page_indexes[page])
    left, right = st.columns([1.06, .94], gap="large")

    with left:
        with st.container(border=True):
            section_heading("Profile details", "", "Patient inputs")
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

        if st.button("Change sex", key="change_gender", use_container_width=True):
            st.session_state.gender_selected = None
            st.session_state.prediction = None
            st.rerun()

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
            elif p1 < .30:
                level = "Elevated probability"
            elif p1 < .60:
                level = "Higher probability"
            else:
                level = "Very high probability"

            st.markdown(
                f'<div class="result"><div class="small">Estimated diabetes probability</div>'
                f'<div class="big">{p1 * 100:.1f}%</div><div class="label">{html.escape(level)}</div>'
                f'<div class="pill">{html.escape(sample["gender"])} · Bayesian estimate</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                card_metric("Diabetes", fmt_pct(p1), "Posterior probability")
            with c2:
                card_metric("Non-diabetes", fmt_pct(p0), "Posterior probability")

            with st.container(border=True):
                fig = go.Figure(
                    go.Bar(
                        x=[p1 * 100, p0 * 100],
                        y=["Diabetes", "Non-diabetes"],
                        orientation="h",
                        text=[fmt_pct(p1), fmt_pct(p0)],
                        textposition="auto",
                        marker=dict(color=["#2f5549", "#d8d8ce"]),
                    )
                )
                fig.update_layout(xaxis_title="Probability (%)", yaxis_title="", xaxis=dict(range=[0, 100]))
                st.plotly_chart(plotly_theme(fig, 270), use_container_width=True, config={"displayModeBar": False})
            st.caption("Educational estimate only. Not a diagnosis or clinical risk score.")
        else:
            with st.container(border=True):
                st.markdown(
                    '<div style="min-height:342px;display:flex;flex-direction:column;justify-content:center;">'
                    '<div class="eyebrow">Result</div><div class="section-title">Probability estimate</div>'
                    '<div class="section-sub" style="margin-top:8px;max-width:310px">Complete the profile and calculate to view the result.</div></div>',
                    unsafe_allow_html=True,
                )

# -----------------------------------------------------------------------------
# Conditional probability
# -----------------------------------------------------------------------------
elif page == "Probability":
    page_header("Probability", "Conditional probabilities calculated from the dataset.", page_indexes[page])
    overall = df.diabetes.mean()
    male = df.loc[df.gender == "Male", "diabetes"].mean()
    female = df.loc[df.gender == "Female", "diabetes"].mean()

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        card_metric("P(Diabetes)", fmt_pct(overall), "Entire dataset")
    with c2:
        card_metric("P(D | Male)", fmt_pct(male), "Conditional probability")
    with c3:
        card_metric("P(D | Female)", fmt_pct(female), "Conditional probability")
    with c4:
        card_metric("Observations", f"{len(df):,}", "Records analysed")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    l, r = st.columns(2, gap="large")
    with l:
        with st.container(border=True):
            gender_rates = df[df.gender.isin(["Male", "Female"])].groupby("gender", as_index=False).diabetes.mean()
            gender_rates["rate"] = gender_rates.diabetes * 100
            fig = px.bar(
                gender_rates, x="gender", y="rate",
                text=gender_rates["rate"].map(lambda x: f"{x:.2f}%"),
                title="Diabetes probability by sex",
                labels={"gender": "Sex", "rate": "Probability (%)"}, template="simple_white",
            )
            fig.update_traces(marker_color="#2f5549", textposition="outside", cliponaxis=False)
            st.plotly_chart(plotly_theme(fig), use_container_width=True)
    with r:
        with st.container(border=True):
            risk_rows = []
            for feature, label in [("hypertension", "Hypertension"), ("heart_disease", "Heart disease")]:
                for value in [0, 1]:
                    probability = df[df[feature] == value].diabetes.mean()
                    risk_rows.append({"Factor": label, "State": "Yes" if value else "No", "Probability": probability * 100})
            risk_frame = pd.DataFrame(risk_rows)
            fig = px.bar(
                risk_frame, x="Factor", y="Probability", color="State", barmode="group",
                title="Probability by existing condition",
                labels={"Probability": "P(Diabetes | condition) (%)"},
                color_discrete_sequence=["#2f5549", "#c58b68"],
            )
            st.plotly_chart(plotly_theme(fig), use_container_width=True)

    with st.container(border=True):
        section_heading("Bayes’ rule", "The model combines the prior probability with the likelihood of each observed feature.")
        st.markdown(
            '<div style="font:500 clamp(1.2rem,2.4vw,1.7rem)/1.35 Newsreader,serif;letter-spacing:-.02em;padding:8px 0 13px;color:#34372f">'
            'P(D | X) ∝ P(D) × P(Gender | D) × P(Age | D) × …</div>'
            '<div style="color:#77796f;font-size:.82rem;line-height:1.65">Numerical variables use Gaussian likelihoods. Categorical variables use smoothed frequency likelihoods. Calculations are performed in log-space for numerical stability.</div>',
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Descriptive statistics and chi-square tests
# -----------------------------------------------------------------------------
elif page == "Statistics":
    page_header("Statistics", "Descriptive measures, confidence intervals and independence tests.", page_indexes[page])
    n = len(df)
    successes = int(df.diabetes.sum())
    p, lo, hi = proportion_ci(successes, n)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        card_metric("Diabetes prevalence", fmt_pct(p), f"95% CI: {lo * 100:.2f}%–{hi * 100:.2f}%")
    with c2:
        card_metric("Mean age", f"{df.age.mean():.1f}", f"SD {df.age.std():.1f} years")
    with c3:
        card_metric("Mean glucose", f"{df.blood_glucose_level.mean():.1f}", f"SD {df.blood_glucose_level.std():.1f}")

    st.markdown("<div style='height:17px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        section_heading("Descriptive summary", "Numeric variables across the complete dataset.")
        st.dataframe(
            df[["age", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"]].describe().T.round(3),
            use_container_width=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    for feature, label in [
        ("gender", "Sex"),
        ("hypertension", "Hypertension"),
        ("heart_disease", "Heart disease"),
        ("smoking_history", "Smoking history"),
    ]:
        table, chi2, pval, dof = chi_square(df, feature)
        significance = "Statistically significant" if pval < 0.05 else "Not statistically significant"
        with st.container(border=True):
            section_heading(f"{label} × diabetes", f"χ² = {chi2:.3f}  ·  df = {dof}  ·  p = {pval:.3e}  ·  {significance} at α = 0.05")
            st.dataframe(table, use_container_width=True)

# -----------------------------------------------------------------------------
# One-way ANOVA
# -----------------------------------------------------------------------------
elif page == "ANOVA":
    page_header("ANOVA", "Compare a numerical measure across groups.", page_indexes[page])
    nums = {"Age": "age", "BMI": "bmi", "HbA1c level": "HbA1c_level", "Blood glucose level": "blood_glucose_level"}
    groups = {"Sex": "gender", "Smoking history": "smoking_history", "Hypertension": "hypertension", "Heart disease": "heart_disease"}

    with st.container(border=True):
        section_heading("Test configuration", "Choose a numerical variable and the groups to compare.")
        a, b = st.columns(2, gap="medium")
        with a:
            numerical_label = st.selectbox("Numerical variable", list(nums), index=2)
        with b:
            group_label = st.selectbox("Grouping variable", list(groups), index=0)
        min_size = st.slider("Minimum observations per group", 2, 100, 20)

    try:
        fstat, pval, dfb, dfw, eta, summary = one_way_anova(df, nums[numerical_label], groups[group_label], min_size)
        significant = pval < 0.05
        conclusion = (
            f"There is statistically significant evidence that mean {numerical_label.lower()} differs across {group_label.lower()} groups."
            if significant else
            f"There is not enough evidence at α = 0.05 to say that mean {numerical_label.lower()} differs across {group_label.lower()} groups."
        )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        with m1:
            card_metric("F statistic", f"{fstat:.3f}", f"{dfb} and {dfw} df")
        with m2:
            card_metric("p-value", f"{pval:.3e}", "α = 0.05")
        with m3:
            card_metric("η² effect size", f"{eta:.3f}", "Variance explained")
        with m4:
            card_metric("Groups", str(len(summary)), "Included in test")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            section_heading("Test result", "H₀: all group means are equal.  H₁: at least one group mean differs.", "Interpretation")
            st.markdown(f'<div style="font-size:.9rem;line-height:1.6;color:#41443b">{html.escape(conclusion)}</div>', unsafe_allow_html=True)

        l, r = st.columns([1.0, 1.15], gap="large")
        with l:
            with st.container(border=True):
                section_heading("Group summary", "Sample size, mean and standard deviation.")
                st.dataframe(summary.round({"Mean": 3, "SD": 3}), use_container_width=True, hide_index=True)
        with r:
            with st.container(border=True):
                plot = df[[nums[numerical_label], groups[group_label]]].dropna()
                if len(plot) > 12000:
                    plot = plot.sample(12000, random_state=42)
                fig = px.box(
                    plot, x=groups[group_label], y=nums[numerical_label], points=False,
                    title=f"{numerical_label} distribution by {group_label}",
                )
                fig.update_traces(marker_color="#2f5549", line_color="#2f5549")
                st.plotly_chart(plotly_theme(fig, 430), use_container_width=True)

    except Exception as exc:
        st.error(f"ANOVA could not be calculated: {exc}")

# -----------------------------------------------------------------------------
# Dataset visualisations
# -----------------------------------------------------------------------------
else:
    page_header("Visuals", "Distributions and relationships in the dataset.", page_indexes[page])
    v1, v2 = st.columns(2, gap="large")
    with v1:
        with st.container(border=True):
            fig = px.histogram(
                df, x="blood_glucose_level", color="diabetes", nbins=45, marginal="box",
                title="Blood glucose distribution",
                labels={"blood_glucose_level": "Blood glucose", "diabetes": "Diabetes"},
                color_discrete_map={0: "#d3d4ca", 1: "#2f5549"},
            )
            st.plotly_chart(plotly_theme(fig), use_container_width=True)
    with v2:
        with st.container(border=True):
            scatter_sample = df.sample(min(5000, len(df)), random_state=7)
            fig = px.scatter(
                scatter_sample, x="bmi", y="blood_glucose_level", color="diabetes", opacity=.55,
                title="BMI vs blood glucose",
                labels={"bmi": "BMI", "blood_glucose_level": "Blood glucose"},
                color_discrete_map={0: "#c8cbc2", 1: "#2f5549"},
            )
            st.plotly_chart(plotly_theme(fig), use_container_width=True)

    v3, v4 = st.columns(2, gap="large")
    with v3:
        with st.container(border=True):
            smoking_rates = df.groupby("smoking_history", as_index=False).diabetes.mean()
            smoking_rates["rate"] = smoking_rates.diabetes * 100
            smoking_rates = smoking_rates.sort_values("rate")
            fig = px.bar(
                smoking_rates, x="rate", y="smoking_history", orientation="h",
                text=smoking_rates["rate"].map(lambda x: f"{x:.2f}%"),
                title="Probability by smoking history",
                labels={"rate": "Probability (%)", "smoking_history": "Smoking history"},
            )
            fig.update_traces(marker_color="#bd7958", textposition="outside", cliponaxis=False)
            st.plotly_chart(plotly_theme(fig), use_container_width=True)
    with v4:
        with st.container(border=True):
            correlation = df[["age", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"]].corr()
            fig = px.imshow(
                correlation, text_auto=".2f", aspect="auto", title="Numerical correlation matrix",
                color_continuous_scale=["#f4f1e8", "#afc1b5", "#2f5549"],
            )
            st.plotly_chart(plotly_theme(fig), use_container_width=True)

    with st.container(border=True):
        section_heading("Dataset sample", "20 rows shown for inspection. Statistical calculations use the complete dataset.")
        st.dataframe(df.sample(20, random_state=42), use_container_width=True, hide_index=True)

st.markdown(
    '<div class="footer"><span>Diabetes Probability Lab</span>'
    '<span>Statistical and educational use only · Not a medical diagnosis</span></div>',
    unsafe_allow_html=True,
)
