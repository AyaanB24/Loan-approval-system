import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.model_utils import FEATURES, NUMERIC, CATEGORICAL, DATA_PATH, MODEL_PATH, train_model, load_model

st.set_page_config(page_title="LoanLens | ML Loan Analysis", page_icon="📊", layout="wide")
st.markdown("""<style>
.main {background:#f7f8fb;} .block-container{padding-top:1.5rem;max-width:1400px;}
.hero{padding:1.4rem 1.6rem;border-radius:16px;background:linear-gradient(120deg,#111827,#263a63);color:white;margin-bottom:1rem;}
.hero h1{color:white;margin:0;font-size:2.2rem;} .hero p{color:#dbeafe;margin:.5rem 0 0 0;}
.note{padding:12px 14px;border-left:4px solid #d97706;background:#fff7ed;border-radius:6px;color:#7c2d12;}
/* White dashboard boxes */
div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    padding: 15px;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
}

/* Black text inside the boxes */
div[data-testid="stMetric"] *,
div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>LoanLens</h1><p>Machine-learning loan application analysis · model evaluation · transparent decision support</p></div>', unsafe_allow_html=True)
st.markdown('<div class="note"><b>Important:</b> This is an educational prototype trained on synthetic demo data. It is not a lender, credit score, or a validated underwriting system. Never use its output to approve, reject, price, or deny real credit. Real lending requires validated data, fairness testing, regulatory review, explainability, security, and human oversight.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Control panel")
    model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
    st.caption("Choose a model, train it, then explore the evaluation and prediction pages.")
    if st.button("Train selected model", use_container_width=True):
        with st.spinner("Training and evaluating..."):
            try:
                _, m = train_model(model_choice)
                st.session_state["active_model"] = model_choice
                st.session_state["active_metrics"] = m
                st.success("Training completed.")
            except Exception as e:
                st.error(str(e))
    st.divider()
    page = st.radio("Navigate", ["Overview", "Predict (demo)", "Model evaluation", "Dataset explorer", "About & limitations"])

# ensure an initial model is available, without hiding training errors
loaded = load_model()
if loaded is None or loaded.get("model_name") != model_choice:
    try:
        _, _metrics = train_model(model_choice)
        loaded = load_model()
    except Exception as exc:
        st.error(f"Could not initialize model: {exc}")
        st.stop()
pipe = loaded["pipeline"]
metrics = loaded["metrics"]

if page == "Overview":
    st.subheader("Project dashboard")
    df = pd.read_csv(DATA_PATH)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Demo applications", f"{len(df):,}")
    c2.metric("Model", loaded["model_name"])
    c3.metric("Holdout accuracy", f"{metrics['accuracy']:.1%}")
    c4.metric("Holdout F1", f"{metrics['f1']:.1%}")
    st.caption("Metrics are measured on a held-out split of synthetic data. High scores here do not imply real-world validity.")
    left,right=st.columns([1.2,1])
    with left:
        st.markdown("#### Historical demo labels")
        counts=df["Loan_Status"].value_counts().rename_axis("Label").reset_index(name="Applications")
        fig=px.bar(counts,x="Label",y="Applications",color="Label",text="Applications",color_discrete_map={"Approved":"#16a34a","Rejected":"#dc2626"})
        fig.update_layout(showlegend=False,plot_bgcolor="white",paper_bgcolor="white",margin=dict(l=10,r=10,t=15,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown("#### What this app does")
        st.markdown("- Trains Logistic Regression or Random Forest\n- Reports holdout metrics and a confusion matrix\n- Estimates the model's class probabilities for a demo application\n- Lets you inspect the bundled dataset\n- Keeps the human-review warning visible")
        st.markdown("#### Input features")
        st.write(", ".join(FEATURES))
elif page == "Predict (demo)":
    st.subheader("Explore one application")
    st.write("Enter fictional values to see how the current model behaves. This is not an actual credit decision.")
    with st.form("prediction_form"):
        a,b,c=st.columns(3)
        with a:
            applicant_income=st.number_input("Applicant annual income (₹)",min_value=0,value=600000,step=10000)
            co_income=st.number_input("Co-applicant annual income (₹)",min_value=0,value=0,step=10000)
            loan_amount=st.number_input("Requested loan amount (₹)",min_value=1,value=1500000,step=25000)
        with b:
            term=st.selectbox("Loan term (months)",[120,180,240,300,360,420],index=4)
            credit=st.selectbox("Credit history flag (demo field)",[1,0],format_func=lambda x:"Positive / recorded (1)" if x==1 else "Negative / missing (0)")
            area=st.selectbox("Property area",["Urban","Semiurban","Rural"])
        with c:
            education=st.selectbox("Education",["Graduate","Not Graduate"])
            self_emp=st.selectbox("Self-employed",["No","Yes"])
            st.caption("Sensitive personal attributes such as gender and religion are not model inputs.")
        submitted=st.form_submit_button("Analyze demo application",type="primary",use_container_width=True)
    if submitted:
        sample=pd.DataFrame([{"ApplicantIncome":applicant_income,"CoapplicantIncome":co_income,"LoanAmount":loan_amount,"Loan_Amount_Term":term,"Credit_History":credit,"Property_Area":area,"Education":education,"Self_Employed":self_emp}])[FEATURES]
        pred=pipe.predict(sample)[0]
        probs=pipe.predict_proba(sample)[0]
        classes=list(pipe.named_steps["model"].classes_)
        approved_prob=float(probs[classes.index("Approved")]) if "Approved" in classes else 0.0
        rejected_prob=float(probs[classes.index("Rejected")]) if "Rejected" in classes else 0.0
        st.markdown("### Model output (not a credit decision)")
        x,y,z=st.columns(3)
        x.metric("Predicted demo label",pred)
        y.metric("Model probability: Approved",f"{approved_prob:.1%}")
        z.metric("Model probability: Rejected",f"{rejected_prob:.1%}")
        st.progress(min(max(approved_prob,0),1),text="Model-estimated probability of the historical 'Approved' label")
        st.warning("Do not interpret this as the probability that a real lender will approve this application. The model learned patterns from synthetic labels, which may not reflect real credit outcomes.")
        with st.expander("See submitted feature values"):
            st.dataframe(sample, use_container_width=True, hide_index=True)
elif page == "Model evaluation":
    st.subheader("Held-out evaluation")
    cols=st.columns(5)
    for col,(label,key) in zip(cols,[("Accuracy","accuracy"),("Precision","precision"),("Recall","recall"),("F1 score","f1"),("ROC AUC","roc_auc")]):
        col.metric(label, f"{metrics[key]:.3f}" if key in metrics else "N/A")
    st.caption(f"Train rows: {metrics['train_rows']} · Test rows: {metrics['test_rows']} · Split: 80/20, stratified, random_state=42")
    st.markdown("#### Confusion matrix")
    cm=pd.DataFrame(metrics["confusion_matrix"],index=["Actual Rejected","Actual Approved"],columns=["Predicted Rejected","Predicted Approved"])
    st.dataframe(cm, use_container_width=True)
    st.markdown("#### Classification report")
    report=pd.DataFrame(metrics["report"]).T
    st.dataframe(report.round(3),use_container_width=True)
    st.info("Accuracy alone is insufficient. For real credit use, evaluate calibration, subgroup error rates, stability over time, data leakage, drift, and the consequences of false positives/negatives.")
elif page == "Dataset explorer":
    st.subheader("Demo dataset")
    df=pd.read_csv(DATA_PATH)
    st.write("The included CSV is generated synthetic data for reproducible demos; it is not real customer or bank data.")
    a,b,c=st.columns(3); a.metric("Rows",len(df)); b.metric("Features",len(df.columns)-1); c.metric("Approval-label share",f"{(df.Loan_Status=='Approved').mean():.1%}")
    query=st.text_input("Search table (optional)",placeholder="e.g. Semiurban")
    view=df.copy()
    if query:
        mask=view.astype(str).apply(lambda col: col.str.contains(query,case=False,na=False)).any(axis=1); view=view[mask]
    st.dataframe(view.head(500),use_container_width=True,hide_index=True)
    st.download_button("Download demo CSV",df.to_csv(index=False).encode("utf-8"),file_name="loan_approval_demo.csv",mime="text/csv")
    st.markdown("#### Label distribution by property area")
    grouped=df.groupby(["Property_Area","Loan_Status"]).size().reset_index(name="Count")
    fig=px.bar(grouped,x="Property_Area",y="Count",color="Loan_Status",barmode="group",color_discrete_map={"Approved":"#16a34a","Rejected":"#dc2626"})
    st.plotly_chart(fig,use_container_width=True)
else:
    st.subheader("About this project")
    st.markdown("""**Stack:** Python, Pandas, scikit-learn, Streamlit, Plotly, Joblib.

**Workflow:** synthetic CSV → preprocessing pipeline → stratified train/test split → model training → evaluation → interactive demo inference.

**Why synthetic data?** It makes the project run immediately without sharing personal data or depending on an external dataset download. It is useful for learning and UI demonstrations, not evidence that the model can underwrite loans.

**Before any real-world pilot:** source a lawful, representative dataset; define a legitimate target; document consent and retention; remove unnecessary sensitive attributes; assess proxy discrimination and subgroup outcomes; validate probability calibration; test for leakage and drift; provide meaningful adverse-action explanations; secure the service; and require qualified human review and legal/compliance approval.

**Limitations:** the demo target was generated from a simple synthetic rule, so model scores largely reflect that rule. Do not publish the resulting metrics as real-world performance. The model is not trained on bank records and is not financial advice.

**Run locally:** `pip install -r requirements.txt` then `streamlit run app.py`.
""")
