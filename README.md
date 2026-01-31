📌 Project Overview

This project is a Student Dropout Early Warning System.
It helps a university find students who are at risk of dropping out so advisors can help them early.

The system:
• Predicts which students may drop out
• Gives each student a risk score
• Flags high-risk students
• Explains why a student is at risk
• Gives suggestions for advisors

📊 Dataset Used

Dataset: xAPI-Edu-Data (from Kaggle)

It contains:
• Student background info
• Class activity
• Engagement behavior
• Attendance data

🧹 Data Cleaning

Steps I did:

• Loaded the CSV file
• Checked for missing values
• Created a new column:

target = 1 → Dropout (Class = L)  
target = 0 → Continue (Class = M / H)


• Dropped unused columns
• Separated features (X) and target (y)

⚙️ Features Used

The model uses:

• Gender
• Grade level
• Topic / subject
• Participation (raised hands)
• Resource visits
• Announcements view
• Absence days
• Relationship support

These features show student behavior and engagement, which are important for early risk detection.

🤖 Model Choice

Model used: Logistic Regression

Why?

✔ Easy to understand
✔ Works well for classification
✔ Gives clear feature importance
✔ Good for non-technical staff

📈 Training & Evaluation

• Split data into Train (75%) and Test (25%)
• Trained model using a Pipeline
• Evaluated with:

• Precision
• Recall
• F1-Score
• ROC-AUC

🎯 Risk Scoring

Each student gets:

• risk_score → Probability of dropout
• risk_label:

Low (< 0.3)

Medium (0.3 – 0.6)

High (≥ 0.6)

• predicted_dropout:

1 = High risk

0 = Not high risk

🚩 High-Risk Students for Advisors

A special file is created:

high_risk_students.csv


It contains only High-Risk students and includes:

• Risk score
• Risk label
• Reason for risk
• Advisor suggestions

🔍 Why Student Is At Risk

For each student, the system explains:

• Low participation
• Low resource usage
• Not checking announcements
• High absences

This is saved in:

risk_reason column

🧭 Advisor Suggestions

Each high-risk student also gets:

advisor_action column


Examples:

• Encourage participation
• Ask to use learning resources
• Discuss attendance issues
• Provide mentoring support

🖥 Streamlit App

A Streamlit app is created that:

• Uploads a student CSV
• Shows Top 20 High-Risk students
• Shows selected student’s risk
• Shows reasons & suggestions

📁 Files Generated

• student_dropout_predictions.csv → All students
• high_risk_students.csv → Only advisors list
• feature_importance.csv → Why model predicts risk
• dropout_pipeline.pkl → Trained model
• preprocessing_pipeline.pkl → Preprocessing steps
• app.py → Streamlit app

✅ Final Result

✔ Early warning system
✔ High-risk students flagged
✔ Clear reasons shown
✔ Advisor actions suggested
✔ Easy to explain to staff
