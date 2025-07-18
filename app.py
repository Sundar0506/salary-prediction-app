import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import os

# ----------------------
# Load Model
# ----------------------
model = joblib.load(open("salary_prediction_model.pkl", "rb"))

# ----------------------
# Language Support
# ----------------------
translations = {
    "en": {
        "title": "💼 Salary Prediction App",
        "upload_csv": "📂 Upload a CSV file",
        "uploaded_preview": "Uploaded Data Preview",
        "eda": "📊 EDA Visualizations",
        "bulk_prediction": "📈 Bulk Salary Prediction",
        "predict_all": "Predict for All Records",
        "download_csv": "⬇ Download Predictions as CSV",
        "manual_prediction": "🧮 Predict for Single Entry",
        "predict_salary": "Predict Salary",
        "predicted_salary": "💰 Predicted Salary",
        "download_pdf": "📄 Download PDF Report with Charts",
        "sidebar": ["EDA", "Prediction", "Report"]
    },
    "hi": {
        "title": "💼 वेतन अनुमान ऐप",
        "upload_csv": "📂 CSV फ़ाइल अपलोड करें",
        "uploaded_preview": "अपलोड किया गया डेटा पूर्वावलोकन",
        "eda": "📊 डेटा विज़ुअलाइज़ेशन",
        "bulk_prediction": "📈 बल्क वेतन अनुमान",
        "predict_all": "सभी रिकॉर्ड्स के लिए अनुमान लगाएं",
        "download_csv": "⬇ अनुमानित परिणाम डाउनलोड करें (CSV)",
        "manual_prediction": "🧮 एकल प्रविष्टि के लिए अनुमान लगाएं",
        "predict_salary": "वेतन का अनुमान लगाएं",
        "predicted_salary": "💰 अनुमानित वेतन",
        "download_pdf": "📄 PDF रिपोर्ट डाउनलोड करें (चार्ट सहित)",
        "sidebar": ["EDA", "पूर्वानुमान", "रिपोर्ट"]
    }
}

# ----------------------
# Sidebar Language Selector
# ----------------------
st.sidebar.title("🌐 Language / भाषा चुनें")
language = st.sidebar.radio("Select Language:", ["English", "हिंदी"])
lang_key = "en" if language == "English" else "hi"
t = translations[lang_key]

# ----------------------
# Sidebar Navigation
# ----------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", t["sidebar"])

# ----------------------
# Page Layout & Styling
# ----------------------
st.set_page_config(page_title="Salary Prediction App", page_icon="💼", layout="wide")
st.markdown("""
    <style>
        .main { background: #ffffff; padding: 20px; border-radius: 8px; }
        h1 { color: #2E86C1; }
    </style>
""", unsafe_allow_html=True)

st.title(t["title"])

# ----------------------
# Shared Variables
# ----------------------
df = None
fig1 = fig2 = fig3 = None

# ----------------------
# EDA Page
# ----------------------
if page == t["sidebar"][0]:
    st.header(t["eda"])
    uploaded_file = st.file_uploader(t["upload_csv"], type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader(t["uploaded_preview"])
        st.dataframe(df.head())

        df_clean = df.dropna()

        # Salary Distribution
        fig1, ax1 = plt.subplots()
        sns.histplot(df_clean["Salary"], bins=30, kde=True, ax=ax1)
        ax1.set_title("Salary Distribution")
        st.pyplot(fig1)

        # Gender-wise Salary
        fig2, ax2 = plt.subplots()
        sns.boxplot(x="Gender", y="Salary", data=df_clean, ax=ax2)
        ax2.set_title("Gender-wise Salary Distribution")
        st.pyplot(fig2)

        # Correlation Heatmap
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.heatmap(df_clean.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax3)
        ax3.set_title("Correlation Heatmap")
        st.pyplot(fig3)

# ----------------------
# Prediction Page
# ----------------------
elif page == t["sidebar"][1]:
    st.header(t["bulk_prediction"])
    uploaded_file = st.file_uploader(t["upload_csv"], type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader(t["uploaded_preview"])
        st.dataframe(df.head())
        df_clean = df.dropna()

        if st.button(t["predict_all"]):
            predictions = model.predict(df_clean.drop(columns=["Salary"], errors='ignore'))
            df_clean["Predicted Salary"] = predictions
            st.success("✅ Predictions completed!")
            st.dataframe(df_clean.head())

            # Download predictions
            csv_file = df_clean.to_csv(index=False).encode("utf-8")
            st.download_button(t["download_csv"], csv_file, "predicted_salaries.csv", "text/csv")

    # Manual Prediction
    st.subheader(t["manual_prediction"])
    age = st.number_input("Age", min_value=18, max_value=60, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
    job_title = st.text_input("Job Title", "Software Engineer")
    experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)

    input_df = pd.DataFrame([[age, gender, education, job_title, experience]],
                            columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"])

    if st.button(t["predict_salary"]):
        predicted_salary = model.predict(input_df)[0]
        st.success(f"{t['predicted_salary']}: ${predicted_salary:,.2f}")

# ----------------------
# Report Page
# ----------------------
elif page == t["sidebar"][2]:
    st.header(t["download_pdf"])
    uploaded_file = st.file_uploader(t["upload_csv"], type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        # Generate EDA charts before creating PDF
        df_clean = df.dropna()
        fig1, ax1 = plt.subplots()
        sns.histplot(df_clean["Salary"], bins=30, kde=True, ax=ax1)
        ax1.set_title("Salary Distribution")
        temp_img1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig1.savefig(temp_img1.name)

        fig2, ax2 = plt.subplots()
        sns.boxplot(x="Gender", y="Salary", data=df_clean, ax=ax2)
        ax2.set_title("Gender-wise Salary Distribution")
        temp_img2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig2.savefig(temp_img2.name)

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.heatmap(df_clean.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax3)
        ax3.set_title("Correlation Heatmap")
        temp_img3 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig3.savefig(temp_img3.name)

        if st.button(t["download_pdf"]):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            c = canvas.Canvas(temp_file.name, pagesize=A4)
            width, height = A4

            # Title
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "Salary Prediction Report")
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 70, "Generated by Streamlit App")

            # Add Table
            y_position = height - 100
            table_data = [df.columns.tolist()] + df.head(10).values.tolist()
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            table.wrapOn(c, 50, y_position - 50)
            table.drawOn(c, 50, y_position - 200)

            # Add Charts
            c.drawImage(temp_img1.name, 50, y_position - 500, width=200, height=120)
            c.drawImage(temp_img2.name, 300, y_position - 500, width=200, height=120)
            c.showPage()
            c.drawImage(temp_img3.name, 100, height - 400, width=300, height=200)

            c.save()

            with open(temp_file.name, "rb") as pdf_file:
                st.download_button("⬇ Download PDF", pdf_file.read(), "Salary_Report.pdf", "application/pdf")
