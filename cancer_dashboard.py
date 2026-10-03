"""Streamlit dashboard for bladder-cancer risk predictions."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Bladder Cancer Predictor", page_icon="🧬", layout="wide")
st.title("🧬 Bladder Cancer Predictor")
st.caption("Research decision-support tool — not a clinical diagnosis.")


@st.cache_resource
def load_bundle() -> dict:
    path = ROOT / "model_bundle.pkl"
    if not path.exists():
        raise FileNotFoundError("model_bundle.pkl is missing. Run the training script first.")
    return joblib.load(path)


def risk_label(probability: float) -> str:
    if probability >= 0.9:
        return "High Risk"
    if probability >= 0.7:
        return "Moderate Risk"
    return "Low Risk"


try:
    bundle = load_bundle()
except (FileNotFoundError, ModuleNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

uploaded_file = st.file_uploader("Upload a CSV containing gene-expression features", type=["csv"])
if uploaded_file is None:
    st.info("Upload a patient or cohort CSV to begin.")
    st.stop()

raw = pd.read_csv(uploaded_file)
input_features = bundle.get("input_feature_names", bundle["feature_names"])
features = bundle["feature_names"]
missing = sorted(set(input_features).difference(raw.columns))
if missing:
    st.error(f"The upload is missing {len(missing)} required feature(s): {', '.join(missing[:10])}")
    st.stop()

try:
    input_values = raw[input_features].apply(pd.to_numeric, errors="raise")
except (TypeError, ValueError):
    st.error("All required gene-expression values must be numeric.")
    st.stop()
if input_values.isna().any().any():
    st.error("The upload contains blank values in required features.")
    st.stop()

st.subheader("Uploaded data")
st.dataframe(input_values.head(10), use_container_width=True)

if st.button("Predict and explain", type="primary"):
    transformed = bundle["selector"].transform(input_values)
    transformed = bundle["scaler"].transform(transformed)
    probabilities = bundle["model"].predict_proba(transformed)[:, 1]
    predictions = (probabilities >= bundle["threshold"]).astype(int)

    results = raw.copy()
    results["Prediction"] = pd.Series(predictions, index=results.index).map({0: "Non-Cancer", 1: "Cancer"})
    results["Cancer Probability"] = probabilities.round(4)
    results["Risk Level"] = [risk_label(value) for value in probabilities]
    st.success("Prediction complete.")
    st.dataframe(results[["Prediction", "Cancer Probability", "Risk Level"]], use_container_width=True)
    st.download_button(
        "Download prediction results",
        results.to_csv(index=False).encode("utf-8"),
        "bladder_cancer_predictions.csv",
        "text/csv",
    )

    st.subheader("SHAP explanation")
    sample_index = st.selectbox("Sample to explain", list(results.index))
    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = explainer(transformed)
    explanation = shap_values[sample_index]
    explanation.feature_names = features
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    st.pyplot(fig, clear_figure=True)

    if st.checkbox("Show global SHAP summary"):
        shap.summary_plot(shap_values, transformed, feature_names=features, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
