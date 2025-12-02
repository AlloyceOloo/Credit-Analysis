import streamlit as st
import pandas as pd
from pathlib import Path
from src.inference.predict import load_models, predict

st.set_page_config(page_title="CreditX Dashboard", layout="wide")
st.title("CreditX Score Dashboard")

uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    st.info("Loading models...")
    lgbm_model, lgbm_features, scorecard_artifact = load_models("models")

    st.info("Running predictions...")
    preds = predict(df, lgbm_model, lgbm_features, scorecard_artifact)
    df_result = pd.concat([df, preds], axis=1)

    st.success("Predictions complete!")
    st.dataframe(df_result.head())

    # Basic visualizations
    st.subheader("Score Distribution")
    st.bar_chart(df_result["scorecard_score"].value_counts().sort_index())

    st.subheader("LGBM Predicted Probability Distribution")
    st.bar_chart(df_result["lgbm_proba"].round(2).value_counts().sort_index())
