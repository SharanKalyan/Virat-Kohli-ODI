import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# IMPORTANT: Import preprocessing function
# --------------------------------------------------
from preprocessing import map_columns  # must exist for pickle load

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
    This application predicts **runs scored by Virat Kohli in an ODI innings**
    using historical match context and conditions.

    🔗 **Complete ML pipeline, preprocessing & model code:**  
    [GitHub Repository](https://github.com/SharanKalyan)
    """
)

st.info("🔒 Demo ML application. No data is stored.")

st.markdown("---")

# --------------------------------------------------
# Model Overview
# --------------------------------------------------
with st.expander("ℹ️ Model Overview"):
    st.markdown(
        """
        **Model:** Linear Regression  
        **Pipeline Includes:**
        - Date feature engineering (Month, Year)
        - Match context encoding
        - Custom preprocessing via sklearn Pipeline
        - End-to-end inference safety

        **Use Case:**  
        Analytical & educational exploration of ODI batting performance.
        """
    )

# --------------------------------------------------
# Load Pipeline
# --------------------------------------------------
@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

model_path = Path("kohli_odi_pipeline.pkl")

if not model_path.exists():
    st.error("❌ Model pipeline file not found.")
    st.stop()

pipeline = load_pipeline(model_path)

# --------------------------------------------------
# Single Prediction
# --------------------------------------------------
st.header("🧍 Single Match Prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        date = st.text_input("Match Date (MM/DD/YYYY)", value="01/15/2023")
        innings = st.selectbox("Innings", ["1st", "2nd", "N/A - No Result"])
        captain = st.selectbox("Captain?", ["Yes", "No"])
        balls_faced = st.number_input(
            "Balls Faced (B/F)", min_value=0, value=60, step=1
        )

    with col2:
        country = st.text_input("Match Country", value="India")
        versus = st.text_input("Opponent", value="Australia")
        sena = st.selectbox("SENA Match?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Runs")

if submitted:
    try:
        X_input = pd.DataFrame([{
            "Date": date,
            "M/Inns": innings,
            "Captain": captain,
            "Country": country,
            "Versus": versus,
            "B/F": balls_faced,
            "SENA": 1 if sena == "Yes" else 0
        }])

        prediction = pipeline.predict(X_input)[0]

        st.success(f"🏏 **Predicted Runs:** {round(prediction, 1)}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------
st.markdown("---")
st.header("📂 Batch Prediction (CSV Upload)")

st.info(
    "Upload a CSV file with the same schema to predict runs for multiple matches."
)

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
