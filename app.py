import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Dropout Early Warning System", layout="wide")

# Load model and feature importance
model = joblib.load("dropout_pipeline.pkl")
feature_importance = pd.read_csv("feature_importance.csv")

st.title("🎓 Student Dropout Early Warning System")
st.write("Upload a student CSV file to see dropout risk and advisor suggestions.")

uploaded_file = st.file_uploader("Upload Student CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Predict risk
    probs = model.predict_proba(df)[:, 1]

    def risk_label(score):
        if score < 0.3:
            return "Low"
        elif score < 0.6:
            return "Medium"
        else:
            return "High"

    df["risk_score"] = probs
    df["risk_label"] = df["risk_score"].apply(risk_label)
    df["predicted_dropout"] = (df["risk_score"] >= 0.6).astype(int)

    st.subheader("🔴 Top 20 High-Risk Students")
    top_risk = df.sort_values("risk_score", ascending=False).head(20)
    st.dataframe(top_risk)

    st.subheader("🔍 Select a Student")
    idx = st.selectbox("Select student row index:", df.index)

    st.write("### 🎯 Risk Details")
    st.write(df.loc[idx, ["risk_score", "risk_label", "predicted_dropout"]])

    st.subheader("📊 Top Reasons (Feature Importance)")
    st.dataframe(feature_importance.head(10))


