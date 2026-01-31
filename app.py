import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Student Dropout Early Warning System", layout="wide")

st.title("🎓 Student Dropout Early Warning System")
st.write("Upload student CSV to predict dropout risk and support advisors.")

uploaded_file = st.file_uploader("Upload xAPI-Edu-Data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # 🎯 Target: L = likely to dropout
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

    # 📊 Risk level
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

    # 🧠 Risk Reason + Advisor Suggestion
    def reason_and_suggestion(row):
        reasons = []
        suggestions = []

        if row.get("raisedhands", 0) < 10:
            reasons.append("Very low class participation")
            suggestions.append("Encourage class interaction and questions")

        if row.get("VisITedResources", 0) < 20:
            reasons.append("Rarely visits learning resources")
            suggestions.append("Guide student to use LMS materials regularly")

        if row.get("AnnouncementsView", 0) < 15:
            reasons.append("Does not follow announcements")
            suggestions.append("Ask student to check updates and deadlines")

        if row.get("Discussion", 0) < 5:
            reasons.append("Not active in discussions")
            suggestions.append("Involve student in group activities")

        if not reasons:
            return "General academic risk", "Monitor and provide regular support"

        return " | ".join(reasons), " | ".join(suggestions)

    df[["risk_reason", "advisor_suggestion"]] = df.apply(
        lambda row: pd.Series(reason_and_suggestion(row)), axis=1
    )

    # 🔴 Advisors Table (Only High Risk)
    advisors_df = df[df["risk_label"] == "High"].copy()
    advisors_df = advisors_df[[
        "risk_score", "risk_label", "risk_reason", "advisor_suggestion"
    ]]

    st.subheader("🧑‍🏫 Advisor Intervention Table (High Risk Students)")
    st.dataframe(advisors_df.reset_index(drop=True))

    # 🔍 Individual Student View
    st.subheader("🔍 Select a Student")
    idx = st.selectbox("Select Row Index", df.index)

    st.write("### 🎯 Risk Info")
    st.write(df.loc[idx, ["risk_score", "risk_label", "risk_reason", "advisor_suggestion"]])
