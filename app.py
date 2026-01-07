import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import base64
from datetime import date as dt_date

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
# Background Image
# --------------------------------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("data:image/jpeg;base64,%s");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        section[data-testid="stMain"] {
            background-color: rgba(255, 255, 255, 0.4);
            padding: 2rem;
            border-radius: 12px;
        }

        /* Page title only */
        div[data-testid="stTitle"] h1 {
            color: #000000 !important;
        }

        input, textarea {
            color: #FFFFFF !important;
        }

        div[data-baseweb="select"] span {
            color: #FFFFFF !important;
        }

        details summary {
            color: #FFFFFF !important;
        }
        </style>
        """ % encoded,
        unsafe_allow_html=True
    )

add_bg_from_local("vk.jpg")

# --------------------------------------------------
# Title & Intro
# --------------------------------------------------
st.title("🏏 Virat Kohli – ODI Runs Prediction")

st.markdown(
    """
    Predict **runs scored by Virat Kohli in an ODI innings**  
    using historical match conditions and context.
    """
)

st.info("🔒 Demo ML application. No data is stored.")

st.markdown("---")

# --------------------------------------------------
# Load Pipeline
# --------------------------------------------------
MODEL_PATH = Path("final_pipeline.pkl")

if not MODEL_PATH.exists():
    st.error("❌ Model pipeline file not found.")
    st.stop()

pipeline = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Constants (Dropdown values)
# --------------------------------------------------
COUNTRY_LIST = [
    'Sri Lanka', 'South Africa', 'India', 'Bangladesh', 'Zimbabwe',
    'Trinidad and Tobago', 'Antigua and Barbuda', 'Jamaica', 'England',
    'Wales', 'Australia', 'New Zealand', 'Guyana', 'Barbados', 'UAE'
]

OPPONENT_LIST = [
    'Sri Lanka', 'Pakistan', 'Australia', 'West Indies', 'Bangladesh',
    'South Africa', 'Zimbabwe', 'New Zealand', 'England', 'Ireland',
    'Netherlands', 'Afghanistan', 'United Arab Emirates', 'Nepal'
]

SENA_COUNTRIES = {'South Africa', 'England', 'Wales', 'New Zealand', 'Australia'}

# --------------------------------------------------
# Model Overview
# --------------------------------------------------
with st.expander("ℹ️ Model Overview"):
    st.markdown(
        """
        **Model:** Linear Regression  
        **Pipeline includes:**
        - Custom column mapping (`FunctionTransformer`)
        - Ordinal encoding for categorical variables
        - ColumnTransformer with explicit columns
        """
    )

# --------------------------------------------------
# Single Prediction
# --------------------------------------------------
st.header("🧍 Single Match Prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        match_date = st.date_input(
            "Match Date",
            value=dt_date(2023, 1, 15)
        )

        innings = st.selectbox(
            "Innings",
            ["1st", "2nd"]
        )

        captain_role = st.selectbox(
            "Captain / Player",
            ["Player", "Captain"],
            index=0
        )

        balls_faced = st.number_input(
            "Balls Faced (B/F)",
            min_value=0,
            value=60
        )

    with col2:
        country = st.selectbox(
            "Match Country",
            COUNTRY_LIST,
            index=COUNTRY_LIST.index("India")
        )

        versus = st.selectbox(
            "Opponent",
            OPPONENT_LIST,
            index=OPPONENT_LIST.index("Australia")
        )

        # Auto SENA logic
        sena_value = 1 if country in SENA_COUNTRIES else 0
        st.text_input(
            "SENA Match?",
            value="Yes" if sena_value == 1 else "No",
            disabled=True
        )

    submitted = st.form_submit_button("Predict Runs")

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if submitted:
    try:
        X_input = pd.DataFrame([{
            "Date": match_date.strftime("%m/%d/%Y"),
            "M/Inns": innings,
            "Captain": "Yes" if captain_role == "Captain" else "No",
            "Country": country,
            "Versus": versus,
            "B/F": int(balls_faced),
            "SENA": sena_value
        }])

        X_input["Date"] = pd.to_datetime(X_input["Date"], errors="coerce")

        prediction = int(pipeline.predict(X_input).ravel()[0])

        st.success(f"🏏 **Predicted Runs:** {prediction}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------
st.markdown("---")
st.header("📂 Batch Prediction (CSV Upload)")

sample_df = pd.DataFrame({
    "Date": ["01/15/2023", "07/20/2022"],
    "M/Inns": ["1st", "2nd"],
    "Captain": ["No", "Yes"],
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
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        df["Predicted_Runs"] = pipeline.predict(df).astype("int64")

        st.dataframe(df)

        st.download_button(
            "⬇️ Download Predictions",
            df.to_csv(index=False),
            "kohli_odi_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"❌ Batch prediction failed: {e}")
