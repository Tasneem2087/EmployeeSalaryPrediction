import streamlit as st
import pandas as pd
import joblib

# Load the model, scaler, and encoders
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar input features
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", list(encoders['workclass'].classes_))
fnlwgt = st.sidebar.number_input("Fnlwgt", min_value=10000, max_value=1000000, value=200000)
education_num = st.sidebar.slider("Education Num", 5, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", list(encoders['marital-status'].classes_))
occupation = st.sidebar.selectbox("Occupation", list(encoders['occupation'].classes_))
relationship = st.sidebar.selectbox("Relationship", list(encoders['relationship'].classes_))
race = st.sidebar.selectbox("Race", list(encoders['race'].classes_))
sex = st.sidebar.selectbox("Sex", list(encoders['sex'].classes_))
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=100000, value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)
native_country = st.sidebar.selectbox("Native Country", list(encoders['native-country'].classes_))

# Input DataFrame
input_data = {
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [education_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'sex': [sex],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
}
input_df = pd.DataFrame(input_data)

# Encode categorical columns
for col in encoders:
    try:
        input_df[col] = encoders[col].transform(input_df[col])
    except ValueError:
        st.error(f"⚠️ Prediction failed: y contains previously unseen label in '{col}'.")
        st.stop()

# Scale inputs
scaled_input = scaler.transform(input_df)

# Show input
st.write("### 🔍 Input Data")
st.write(input_df)

# Predict
if st.button("🎯 Predict Salary Class"):
    prediction = model.predict(scaled_input)
    result = ">50K" if prediction[0] == 1 else "≤50K"
    st.success(f"✅ Prediction: Employee earns {result}")

# Batch Prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    try:
        for col in encoders:
            if col in batch_df.columns:
                batch_df[col] = encoders[col].transform(batch_df[col])
        batch_df = batch_df.drop(columns=['education'], errors='ignore')  # drop extra if present
        batch_scaled = scaler.transform(batch_df.drop(columns=['income'], errors='ignore'))
        batch_preds = model.predict(batch_scaled)
        batch_df['Prediction'] = ['>50K' if p == 1 else '≤50K' for p in batch_preds]
        st.write(batch_df.head())
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Predictions CSV", csv, file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"⚠️ Error in batch prediction: {str(e)}")
