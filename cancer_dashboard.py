import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and feature names
model = joblib.load("XGBoost_best_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Cancer Prediction Dashboard", layout="wide")
st.title("🧬 Cancer Predictor Tool: Cancer Prediction from Gene Expression")

uploaded_file = st.file_uploader("📂 Upload gene expression CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if not set(feature_names).issubset(df.columns):
        st.error("❌ Uploaded file does not contain all required gene features.")
    else:
        X_input = df[feature_names]
        st.subheader("📊 Preview of Uploaded Data")
        st.dataframe(X_input.head())

        # Predict button
        if st.button("🔍 Predict and Explain"):
            preds = model.predict(X_input)
            probs = model.predict_proba(X_input)[:, 1]

            # Add prediction labels
            df['Prediction'] = preds
            df['Prediction'] = df['Prediction'].map({0: 'Non-Cancer', 1: 'Cancer'})
            df['Probability'] = probs

            # Add Risk Level column
            def label_risk(prob):
                if prob >= 0.9:
                    return "High Risk"
                elif prob >= 0.7:
                    return "Moderate Risk"
                else:
                    return "Low Risk"
            df['Risk Level'] = df['Probability'].apply(label_risk)

            st.success("✅ Prediction complete!")
            st.dataframe(df[['Prediction', 'Probability', 'Risk Level']].head())

            # Extract XGBoost classifier from pipeline
            xgb_model = model.named_steps['classifier']

            # Create SHAP explainer & compute values
            explainer = shap.Explainer(xgb_model, X_input)
            shap_values = explainer(X_input)

            # SHAP Explanation Section
            st.subheader("🧪 SHAP Explanation per Sample")
            sample_idx = st.selectbox("Select a Sample Index to Explain", df.index)

            st.markdown("### 🔬 SHAP Waterfall Plot")
            shap.plots.waterfall(shap_values[sample_idx], show=False)
            st.pyplot(bbox_inches='tight')

            if st.checkbox("📈 Show Global SHAP Summary Plot"):
                st.markdown("### SHAP Summary Plot (All Samples)")
                shap.summary_plot(shap_values, X_input, show=False)
                st.pyplot(bbox_inches='tight')

            # Download predictions
            st.download_button(
                label="📥 Download Prediction Results",
                data=df.to_csv(index=False),
                file_name="cancer_predictions.csv"
            )





