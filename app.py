import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# IMPORTANT: preprocessing import (pickle dependency)
# --------------------------------------------------
from preprocessing import map_columns  # DO NOT REMOVE

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Virat Kohli ODI Runs Predictor",
    layout="centered"
)

# --------------------------------------------------
# Title & Intro
# --------------------------------------------------
st.title("🏏 Virat Kohli – ODI Runs Prediction")

st.markdown(
    """
    Predict **runs scored by Virat Kohli in an ODI innings**  
    using historical match conditions and context.

    🔗 **ML pipeline, preprocessing & training code:**  
    [GitHub Repository](https://github.com/SharanKalyan)
    
    🔗 **Tableau Public Dashboard:**  
    [Tableau Dashboard](https://public.tableau.com/app/profile/sharankalyan/viz/ViratKohli-ODI/ViratKohliODIDashboard?publish=yes)
    """
)

st.info("🔒 Demo ML application. No data is stored.")

st.markdown("---")

# --------------------------------------------------
# Load Pipeline
# --------------------------------------------------
@st.cache_resource
def load_pipeline(path: Path):
    return joblib.load(path)

MODEL_PATH = Path("kohli_odi_pipeline.pkl")

if not MODEL_PATH.exists():
    st.error("❌ Model pipeline file not found.")
    st.stop()

pipeline = load_pipeline(MODEL_PATH)

# --------------------------------------------------
# Model Overview
# --------------------------------------------------
with st.expander("ℹ️ Model Overview"):
    st.markdown(
        """
        **Model:** Linear Regression  
        **Pipeline includes:**
        - Date feature engineering
        - Custom preprocessing via `FunctionTransformer`
        - Ordinal encoding
        - End-to-end sklearn Pipeline

        **Use case:** Analytical & educational ODI performance modeling
        """
    )

# --------------------------------------------------
# Single Prediction
# --------------------------------------------------
st.header("🧍 Single Match Prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        date = st.text_input("Match Date (MM/DD/YYYY)", "01/15/2023")
        innings = st.selectbox("Innings", ["1st", "2nd", "N/A - No Result"])
        captain = st.selectbox("Captain?", ["Yes", "No"])
        balls_faced = st.number_input("Balls Faced (B/F)", min_value=0, value=60)

    with col2:
        country = st.text_input("Match Country", "India")
        versus = st.text_input("Opponent", "Australia")
        sena = st.selectbox("SENA Match?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Runs")

if submitted:
    try:
        X_input = pd.DataFrame(
            [{
                "Date": date,
                "M/Inns": innings,
                "Captain": captain,
                "Country": country,
                "Versus": versus,
                "B/F": int(balls_faced),
                "SENA": 1 if sena == "Yes" else 0
            }]
        )

        # 🔒 SAFETY: always pass a DataFrame copy
        prediction = pipeline.predict(X_input.copy())[0]

        st.success(f"🏏 **Predicted Runs:** {round(prediction, 1)}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------
st.markdown("---")
st.header("📂 Batch Prediction (CSV Upload)")

st.info("Upload a CSV with the same schema used during training.")

# Sample CSV
sample_df = pd.DataFrame({
    "Date": ["01/15/2023", "07/20/2022"],
    "M/Inns": ["1st", "2nd"],
    "Captain": ["Yes", "No"],
    "Country": ["India", "England"],
    "Versus": ["Australia", "Pakistan"],
    "B/F": [75, 48],
    "SENA": [0, 1]
})

st.download_button(
    "⬇️ Download Sample CSV",
    sample_df.to_csv(index=False),
    "sample_kohli_odi_data.csv",
    "text/csv"
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        required_cols = sample_df.columns.tolist()
        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            st.stop()

        df = df.copy()  # 🔒 SAFETY

        df["Predicted_Runs"] = pipeline.predict(df)

        st.dataframe(df)

        st.download_button(
            "⬇️ Download Predictions",
            df.to_csv(index=False),
            "kohli_odi_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"❌ Batch prediction failed: {e}")
