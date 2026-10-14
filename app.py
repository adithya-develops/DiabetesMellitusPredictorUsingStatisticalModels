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

st.set_page_config(page_title="Diabetes — Probability Lab", page_icon="◉", layout="wide", initial_sidebar_state="collapsed")

DATA_PATH = ROOT / "data" / "diabetes_prediction_dataset.csv"
df = load_data(DATA_PATH)
predictor = MixedBayesianPredictor(df)

# -------------------- Theme --------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#171a19; --muted:#737773; --paper:#f4f1ea; --card:rgba(255,253,248,.86); --line:rgba(23,26,25,.10); --green:#214b42; --green2:#2c6256; --gold:#b28a55; }
html,body,[class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
.stApp { background:linear-gradient(135deg,#f7f4ed 0%,#efede6 52%,#f8f6f0 100%); }
.block-container { max-width:1180px; padding-top:1.1rem; padding-bottom:3.5rem; }
[data-testid="stHeader"] { background:rgba(247,244,237,.78); backdrop-filter:blur(18px); }
[data-testid="stToolbar"] { display:none; }
/* Quiet navigation */
div[role="radiogroup"] { width:100%; display:flex; gap:3px; padding:4px; border:1px solid var(--line); border-radius:14px; background:rgba(255,255,255,.55); }
div[role="radiogroup"] label { flex:1; justify-content:center; border:0!important; border-radius:10px!important; padding:9px 12px!important; background:transparent!important; color:#696e69!important; font-size:.84rem!important; transition:all .18s ease; }
div[role="radiogroup"] label:hover { color:var(--ink)!important; transform:none!important; }
div[role="radiogroup"] label[data-checked="true"] { background:#214b42!important; color:white!important; box-shadow:0 5px 18px rgba(33,75,66,.16); }
.hero { padding:31px 34px 29px; border:1px solid var(--line); border-radius:26px; background:rgba(255,253,248,.78); box-shadow:0 18px 55px rgba(32,34,30,.07); animation:fadeUp .5s ease both; }
.hero .eyebrow,.eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.67rem; font-weight:800; color:var(--green); }
.hero h1 { font-family:'Manrope',sans-serif; font-size:clamp(2.05rem,4vw,3.5rem); line-height:1.02; margin:.42rem 0 .55rem; letter-spacing:-.055em; font-weight:800; }
.hero p { max-width:650px; color:#686d68; font-size:.94rem; line-height:1.55; margin:0; }
.hero .badge { display:none; }
.card { border:1px solid var(--line); background:var(--card); border-radius:21px; padding:22px; box-shadow:0 12px 35px rgba(32,34,30,.045); animation:fadeUp .38s ease both; }
.metric { padding:19px 20px; border-radius:18px; background:rgba(255,253,248,.82); border:1px solid var(--line); box-shadow:0 9px 27px rgba(32,34,30,.04); transition:transform .2s ease,box-shadow .2s ease; }
.metric:hover { transform:translateY(-2px); box-shadow:0 15px 32px rgba(32,34,30,.08); }
.metric .k { color:#777b76;font-size:.66rem;text-transform:uppercase;letter-spacing:.12em;font-weight:800; }
.metric .v { font-family:'Manrope',sans-serif;font-size:1.85rem;font-weight:800;letter-spacing:-.04em;margin-top:5px; }
.metric .s { color:#858984;font-size:.76rem;margin-top:2px; }
.section-title { font-family:'Manrope',sans-serif; font-size:1.3rem; font-weight:800; letter-spacing:-.035em; margin:4px 0 3px; }
.section-sub { color:#777b76; font-size:.83rem; margin-bottom:15px; line-height:1.45; }
.question { font-size:.68rem; font-weight:800; text-transform:uppercase; letter-spacing:.11em; color:#626762; margin:3px 0 8px; }
.result { padding:25px; border-radius:23px; background:linear-gradient(135deg,#214b42,#2c6256); color:#fff; box-shadow:0 20px 48px rgba(33,75,66,.18); position:relative; overflow:hidden; animation:fadeUp .45s ease both; }
.result:after { content:""; position:absolute; width:260px;height:260px;right:-110px;top:-150px;border-radius:50%;border:1px solid rgba(255,255,255,.12); box-shadow:0 0 0 34px rgba(255,255,255,.025),0 0 0 68px rgba(255,255,255,.018); }
.result .small { color:rgba(255,255,255,.62); text-transform:uppercase; letter-spacing:.11em;font-size:.66rem;font-weight:800; }
.result .big { font-family:'Manrope',sans-serif; font-size:clamp(3rem,6vw,5rem); line-height:.92; font-weight:800; letter-spacing:-.07em; margin:9px 0; }
.result .label { font-size:.88rem; color:rgba(255,255,255,.78); }
.pill { display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.11);font-size:.72rem;font-weight:600;margin-top:13px; }
.footer { margin-top:30px; padding:15px 1px; color:#90938e; font-size:.72rem; border-top:1px solid var(--line); }
.stButton > button { border-radius:12px; border:1px solid rgba(23,26,25,.12); font-weight:600; min-height:43px; transition:all .18s ease; background:rgba(255,253,248,.78); }
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 8px 22px rgba(32,34,30,.09); border-color:rgba(33,75,66,.25); }
div[data-testid="stForm"] { border:0; padding:0; }
div[data-baseweb="select"] > div,.stNumberInput input,.stTextInput input { border-radius:11px!important; }
[data-testid="stDataFrame"] { border-radius:16px; overflow:hidden; }
[data-testid="stMetric"] { background:transparent; }
@keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
@media(max-width:760px){.hero{padding:25px 21px;border-radius:21px}.card{padding:18px;border-radius:18px}.block-container{padding:1rem 1rem 2.5rem}.result{padding:21px;border-radius:20px}}
</style>
""", unsafe_allow_html=True)

# -------------------- Helpers --------------------
def fmt_pct(x): return f"{x*100:.2f}%"
def card_metric(k, v, s):
    st.markdown(f'<div class="metric"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div><div class="s">{html.escape(s)}</div></div>', unsafe_allow_html=True)

def plotly_theme(fig, height=410):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#33383d"), margin=dict(l=10,r=10,t=48,b=10),
        hoverlabel=dict(bgcolor="#121817", font_color="#fff"),
        legend=dict(bgcolor="rgba(255,255,255,.65)", bordercolor="rgba(0,0,0,.06)", borderwidth=1)
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(18,20,23,.07)", zeroline=False)
    return fig

# -------------------- Session flow --------------------
if "gender_selected" not in st.session_state: st.session_state.gender_selected = None
if "prediction" not in st.session_state: st.session_state.prediction = None

# -------------------- Header --------------------
st.markdown("""<div class="hero"><div class="eyebrow">Probability & Statistics</div><h1>Diabetes Study</h1><p>Probability, statistical tests and distributions from a 100,000-record dataset.</p></div>""", unsafe_allow_html=True)

# -------------------- Gender-first gate --------------------
if st.session_state.gender_selected is None:
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="eyebrow">Patient</div><div class="section-title">Select sex</div><div class="section-sub">This is the first variable used in the probability calculation.</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,1,2])
    with c1:
        if st.button("Male", use_container_width=True, key="male_start"):
            st.session_state.gender_selected = "Male"; st.rerun()
    with c2:
        if st.button("Female", use_container_width=True, key="female_start"):
            st.session_state.gender_selected = "Female"; st.rerun()
    with c3:
        st.markdown('', unsafe_allow_html=True)
    st.markdown('<div class="footer">Statistical/educational use only · Not a medical diagnosis.</div>', unsafe_allow_html=True)
    st.stop()

# -------------------- Navigation --------------------
page = st.radio("", ["Prediction", "Probability", "Statistics", "ANOVA", "Visuals"], horizontal=True, label_visibility="collapsed")

# -------------------- Prediction --------------------
if page == "Prediction":
    st.markdown(f'<div class="nav"><span class="eyebrow">Patient</span> &nbsp; <strong>{html.escape(st.session_state.gender_selected)}</strong></div>', unsafe_allow_html=True)
    left,right = st.columns([1.08,.92], gap="large")
    with left:
        st.markdown('<div class="card"><div class="section-title">Complete the profile</div><div class="section-sub">These measurements are passed into the probability model.</div>', unsafe_allow_html=True)
        with st.form("patient_form"):
            a,b = st.columns(2)
            with a:
                age = st.number_input("Age", min_value=0.1, max_value=100.0, value=40.0, step=1.0)
                bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=27.3, step=0.1, format="%.1f")
                hba1c = st.number_input("HbA1c level", min_value=3.0, max_value=10.0, value=5.7, step=0.1, format="%.1f")
            with b:
                glucose = st.number_input("Blood glucose level", min_value=40, max_value=300, value=120, step=1)
                hypertension = st.selectbox("Hypertension", [0,1], format_func=lambda x: "Yes" if x else "No")
                heart = st.selectbox("Heart disease", [0,1], format_func=lambda x: "Yes" if x else "No")
            smoking = st.selectbox("Smoking history", ["never","No Info","former","current","not current","ever"], index=0)
            submitted = st.form_submit_button("Calculate probability  →", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("← Choose a different gender", key="change_gender"):
            st.session_state.gender_selected=None; st.session_state.prediction=None; st.rerun()
    with right:
        if submitted:
            sample={"gender":st.session_state.gender_selected,"age":age,"hypertension":hypertension,"heart_disease":heart,"smoking_history":smoking,"bmi":bmi,"HbA1c_level":hba1c,"blood_glucose_level":glucose}
            p1,p0=predictor.predict(sample)
            st.session_state.prediction=(sample,p1,p0)
        if st.session_state.prediction:
            sample,p1,p0=st.session_state.prediction
            if p1 < .10: level="Lower probability"
            elif p1 < .30: level="Elevated probability"
            elif p1 < .60: level="Higher probability"
            else: level="Very high probability"
            st.markdown(f'<div class="result"><div class="small">Estimated diabetes probability</div><div class="big">{p1*100:.1f}%</div><div class="label">{html.escape(level)} · based on this dataset and profile</div><div class="pill">{html.escape(sample["gender"])} · Bayesian estimate</div></div>', unsafe_allow_html=True)
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1: card_metric("Diabetes",fmt_pct(p1),"posterior probability")
            with c2: card_metric("Non-diabetes",fmt_pct(p0),"posterior probability")
            fig=go.Figure(go.Bar(x=[p1*100,p0*100],y=["Diabetes","Non-diabetes"],orientation="h",text=[fmt_pct(p1),fmt_pct(p0)],textposition="auto",marker=dict(color=["#174a43","#d9ddd9"])))
            fig.update_layout(xaxis_title="Probability (%)", yaxis_title="", xaxis=dict(range=[0,100]))
            st.plotly_chart(plotly_theme(fig,300),use_container_width=True,config={"displayModeBar":False})
            st.caption("Educational estimate only — not a diagnosis or clinical risk score.")
        else:
            st.markdown('<div class="card" style="min-height:420px;display:flex;align-items:center;justify-content:center;text-align:center"><div><div class="eyebrow">Waiting for profile</div><div class="section-title">Your probability result will appear here.</div><div class="section-sub">Complete the profile and run the calculation to reveal the posterior probability.</div></div></div>',unsafe_allow_html=True)

# -------------------- Probability --------------------
elif page == "Probability":
    st.markdown('<div class="section-title">Probability</div><div class="section-sub">Conditional probabilities calculated from the dataset.</div>',unsafe_allow_html=True)
    overall=df.diabetes.mean()
    male=df.loc[df.gender=="Male","diabetes"].mean(); female=df.loc[df.gender=="Female","diabetes"].mean()
    c1,c2,c3,c4=st.columns(4)
    with c1: card_metric("P(Diabetes)",fmt_pct(overall),"entire dataset")
    with c2: card_metric("P(D | Male)",fmt_pct(male),"conditional probability")
    with c3: card_metric("P(D | Female)",fmt_pct(female),"conditional probability")
    with c4: card_metric("Observations",f"{len(df):,}","records analysed")
    st.markdown('<div style="height:18px"></div>',unsafe_allow_html=True)
    l,r=st.columns(2,gap="large")
    with l:
        gender_rates=df[df.gender.isin(["Male","Female"])].groupby("gender",as_index=False).diabetes.mean(); gender_rates["rate"]=gender_rates.diabetes*100
        fig=px.bar(gender_rates,x="gender",y="rate",text=gender_rates["rate"].map(lambda x:f"{x:.2f}%"),title="Diabetes probability by gender",labels={"gender":"Gender","rate":"Probability (%)"},template="simple_white")
        fig.update_traces(marker_color="#174a43",textposition="outside")
        st.plotly_chart(plotly_theme(fig),use_container_width=True)
    with r:
        risk_rows=[]
        for f,label in [("hypertension","Hypertension"),("heart_disease","Heart disease")]:
            for v in [0,1]:
                s=df[df[f]==v].diabetes.mean(); risk_rows.append({"Factor":label,"State":"Yes" if v else "No","Probability":s*100})
        rr=pd.DataFrame(risk_rows)
        fig=px.bar(rr,x="Factor",y="Probability",color="State",barmode="group",title="Conditional probability by existing condition",labels={"Probability":"P(Diabetes | condition) (%)"},color_discrete_sequence=["#174a43","#c78b42"])
        st.plotly_chart(plotly_theme(fig),use_container_width=True)
    st.markdown('<div class="card"><div class="section-title">Bayes intuition</div><div class="section-sub">For a profile X, the model combines the prior probability with the likelihood of each observed feature.</div><div style="font-family:Manrope;font-size:1.25rem;font-weight:700;padding:12px 0">P(D | X) ∝ P(D) × P(Gender | D) × P(Age | D) × …</div><div style="color:#777;line-height:1.6">Numerical variables use Gaussian likelihoods. Categorical variables use smoothed frequency likelihoods. Calculations are performed in log-space for numerical stability.</div></div>',unsafe_allow_html=True)

# -------------------- Statistics --------------------
elif page == "Statistics":
    st.markdown('<div class="section-title">Statistics</div><div class="section-sub">Descriptive statistics, confidence intervals and independence tests.</div>',unsafe_allow_html=True)
    n=len(df); successes=int(df.diabetes.sum()); p,lo,hi=proportion_ci(successes,n)
    c1,c2,c3=st.columns(3)
    with c1: card_metric("Diabetes prevalence",fmt_pct(p),f"95% CI: {lo*100:.2f}%–{hi*100:.2f}%")
    with c2: card_metric("Mean age",f"{df.age.mean():.1f}",f"SD {df.age.std():.1f} years")
    with c3: card_metric("Mean glucose",f"{df.blood_glucose_level.mean():.1f}",f"SD {df.blood_glucose_level.std():.1f}")
    st.markdown('<div style="height:18px"></div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="section-title">Descriptive summary</div></div>',unsafe_allow_html=True)
    st.dataframe(df[["age","bmi","HbA1c_level","blood_glucose_level","diabetes"]].describe().T.round(3),use_container_width=True)
    st.markdown('<div style="height:18px"></div>',unsafe_allow_html=True)
    for feature,label in [("gender","Gender"),("hypertension","Hypertension"),("heart_disease","Heart disease"),("smoking_history","Smoking history")]:
        table,chi2,pval,dof=chi_square(df,feature)
        sig="statistically significant" if pval<0.05 else "not statistically significant"
        st.markdown(f'<div class="card"><div class="section-title">{label} × diabetes</div><div class="section-sub">Chi-square = {chi2:.3f} · df = {dof} · p-value = {pval:.3e} · <strong>{sig}</strong> at α = 0.05</div></div>',unsafe_allow_html=True)
        st.dataframe(table,use_container_width=True)

# -------------------- ANOVA --------------------
elif page == "ANOVA":
    st.markdown('<div class="section-title">ANOVA</div><div class="section-sub">Compare a numerical measure across groups.</div>', unsafe_allow_html=True)
    nums={"Age":"age","BMI":"bmi","HbA1c level":"HbA1c_level","Blood glucose level":"blood_glucose_level"}
    groups={"Gender":"gender","Smoking history":"smoking_history","Hypertension":"hypertension","Heart disease":"heart_disease"}
    a,b=st.columns(2,gap="large")
    with a: nl=st.selectbox("Numerical variable",list(nums),index=2)
    with b: gl=st.selectbox("Grouping variable",list(groups),index=0)
    min_size=st.slider("Minimum observations per group",2,100,20)
    try:
        fstat,pval,dfb,dfw,eta,summary=one_way_anova(df,nums[nl],groups[gl],min_size)
        sig=pval<0.05
        conclusion=f"There is statistically significant evidence that mean {nl.lower()} differs across {gl.lower()} groups." if sig else f"There is not enough evidence at α = 0.05 to say that mean {nl.lower()} differs across {gl.lower()} groups."
        m1,m2,m3,m4=st.columns(4)
        with m1: card_metric("F statistic",f"{fstat:.3f}",f"{dfb} and {dfw} df")
        with m2: card_metric("p-value",f"{pval:.3e}","α = 0.05")
        with m3: card_metric("η² effect size",f"{eta:.3f}","variance explained")
        with m4: card_metric("Groups",str(len(summary)),"included")
        st.markdown(f'<div class="card"><div class="eyebrow">Result</div><div class="section-title">H₀: all group means are equal</div><div class="section-sub">H₁: at least one group mean is different.</div><strong>{html.escape(conclusion)}</strong></div>',unsafe_allow_html=True)
        l,r=st.columns([1.0,1.15],gap="large")
        with l:
            st.markdown('<div class="card"><div class="section-title">Group summary</div><div class="section-sub">Sample size, mean and standard deviation.</div></div>',unsafe_allow_html=True)
            st.dataframe(summary.round({"Mean":3,"SD":3}),use_container_width=True,hide_index=True)
        with r:
            plot=df[[nums[nl],groups[gl]]].dropna()
            if len(plot)>12000: plot=plot.sample(12000,random_state=42)
            fig=px.box(plot,x=groups[gl],y=nums[nl],points=False,title=f"{nl} distribution by {gl}")
            fig.update_traces(marker_color="#174a43",line_color="#174a43")
            st.plotly_chart(plotly_theme(fig,430),use_container_width=True)

    except Exception as exc:
        st.error(f"ANOVA could not be calculated: {exc}")

# -------------------- Visuals --------------------
else:
    st.markdown('<div class="section-title">Visuals</div><div class="section-sub">Distributions and relationships in the dataset.</div>',unsafe_allow_html=True)
    v1,v2=st.columns(2,gap="large")
    with v1:
        fig=px.histogram(df,x="blood_glucose_level",color="diabetes",nbins=45,marginal="box",title="Blood glucose distribution",labels={"blood_glucose_level":"Blood glucose","diabetes":"Diabetes"},color_discrete_map={0:"#cfd4d0",1:"#174a43"})
        st.plotly_chart(plotly_theme(fig),use_container_width=True)
    with v2:
        fig=px.scatter(df.sample(min(5000,len(df)),random_state=7),x="bmi",y="blood_glucose_level",color="diabetes",opacity=.55,title="BMI vs blood glucose",labels={"bmi":"BMI","blood_glucose_level":"Blood glucose"},color_discrete_map={0:"#cfd4d0",1:"#174a43"})
        st.plotly_chart(plotly_theme(fig),use_container_width=True)
    v3,v4=st.columns(2,gap="large")
    with v3:
        tmp=df.groupby("smoking_history",as_index=False).diabetes.mean(); tmp["rate"]=tmp.diabetes*100
        fig=px.bar(tmp.sort_values("rate"),x="rate",y="smoking_history",orientation="h",text=tmp.sort_values("rate")["rate"].map(lambda x:f"{x:.2f}%"),title="Diabetes probability by smoking history",labels={"rate":"Probability (%)","smoking_history":"Smoking history"})
        fig.update_traces(marker_color="#c78b42",textposition="outside")
        st.plotly_chart(plotly_theme(fig),use_container_width=True)
    with v4:
        corr=df[["age","bmi","HbA1c_level","blood_glucose_level","diabetes"]].corr()
        fig=px.imshow(corr,text_auto=".2f",aspect="auto",title="Numerical correlation matrix",color_continuous_scale=["#f2f3ef","#9ebbb4","#174a43"])
        st.plotly_chart(plotly_theme(fig),use_container_width=True)
    st.markdown('<div class="card"><div class="section-title">Dataset snapshot</div><div class="section-sub">A small sample is shown for readability; all statistical calculations use the complete 100,000-row dataset.</div></div>',unsafe_allow_html=True)
    st.dataframe(df.sample(20,random_state=42),use_container_width=True)

st.markdown('<div class="footer">Diabetes Probability Lab · Built for Probability & Statistics · Statistical/educational use only · The model does not provide medical diagnosis.</div>',unsafe_allow_html=True)
