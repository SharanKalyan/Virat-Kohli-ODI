# Virat Kohli – ODI Performance Analysis & Run Prediction

This project focuses on an in-depth analysis of One Day International (ODI) matches played by Virat Kohli. The repository contains end-to-end data science workflows, including data cleaning, exploratory analysis, visualization, and machine learning models to predict the number of runs he may score in upcoming matches based on multiple influencing factors.

# Tableau Public Dashboard

An interactive dashboard showcasing key insights and performance trends:
https://public.tableau.com/app/profile/sharankalyan/viz/ViratKohli-ODI/ViratKohliODIDashboard?publish=yes

# Model deployed on Streamlit: 
https://virat-kohli-odi-predictions.streamlit.app/

# Project Structure & Workflow
### 1. Cleaning-Virat-Kohli-ODI.ipynb

This notebook contains the complete data-cleaning pipeline.
The primary challenge was resolving inconsistencies across multiple columns. For example, the Runs column was stored as a string due to special characters such as *, which indicate not-out innings. Additional issues included missing values, inconsistent formats, and noisy data.
Another significant challenge was identifying and standardizing the geographical locations of cricket grounds. After addressing these issues, the cleaned and structured dataset was saved for downstream analysis.

### 2. Analysis-Virat-Kohli-ODI.ipynb

This notebook focuses on exploratory data analysis (EDA) and visualization. Most insights were derived and validated using the Tableau dashboard. Through this analysis, I identified:
Performance patterns against different oppositions
Strengths and weaknesses under varying conditions
Situational trends where performance peaks or declines
The finalized analytical dataset was then prepared and forwarded for machine learning modeling.

### 3. RegressionPipeline-Virat-Kohli-ODI.ipynb

This notebook implements the machine learning pipeline, including:
Data preprocessing and feature engineering pipelines
Model training and validation pipeline
Regression-based run prediction
Performance evaluation of the trained models
Saving trained model

The goal of this stage is to estimate the number of runs Virat Kohli is likely to score in future ODI matches based on historical and contextual parameters.


# Results of Predictions
# Metrics
r2 Score: 0.95 ; MAE: 7; RMSE: 10

<img width="875" height="556" alt="image" src="https://github.com/user-attachments/assets/e9a23965-b137-48e7-befe-51bde5e2fe4a" />

<img width="1187" height="696" alt="Preds-plot" src="https://github.com/user-attachments/assets/edd06441-f5b3-498d-9cf5-92998974a8af" />

# Here's the prediction for upcomming NZL ODIs

<img width="1256" height="485" alt="image" src="https://github.com/user-attachments/assets/5506628f-4960-4120-a558-8193f30ac7d4" />



