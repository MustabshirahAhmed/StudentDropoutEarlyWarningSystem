import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Student Dropout Early Warning System", layout="wide")

st.title("🎓 Student Dropout Early Warning System")
st.write("Upload student CSV to predict dropout risk.")

uploaded_file = st.file_uploader("Upload xAPI-Edu-Data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Target
    df["target"] = df["Class"].apply(lambda x: 1 if x == "L" else 0)
    X = df.drop(["Class", "target"], axis=1)
    y = df["target"]

    cat_cols = X.select_dtypes(include="object").columns

    ct = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)],
        remainder="passthrough"
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", ct),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)

    probs = pipeline.predict_proba(X)[:, 1]

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

    st.subheader("🔴 Top 20 High Risk Students")
    st.dataframe(df.sort_values("risk_score", ascending=False).head(20))

    st.subheader("🔍 Select Student")
    idx = st.selectbox("Select Row Index", df.index)

    st.write("### 🎯 Risk Info")
    st.write(df.loc[idx, ["risk_score", "risk_label", "predicted_dropout"]])
