import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    with open("Student_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -------------------------------
# Load Data (Optional)
# -------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Employee_clean_Data.csv")
        return df
    except:
        return None

data = load_data()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📘 Student Performance Prediction App")
st.write("Enter the input values below to get prediction")

# -------------------------------
# Auto-detect numeric columns
# -------------------------------
if data is not None:
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    st.sidebar.header("Dataset Info")
    st.sidebar.write(f"Loaded Columns: {list(data.columns)}")
else:
    numeric_cols = []

# -------------------------------
# User Input
# -------------------------------
st.subheader("Enter Feature Values")

user_inputs = {}
for col in numeric_cols:
    user_inputs[col] = st.number_input(f"{col}", value=0)

# Convert to DataFrame
input_df = pd.DataFrame([user_inputs])

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 **Prediction: {prediction}**")
    except Exception as e:
        st.error(f"❌ Error while predicting: {e}")
        st.info("Make sure feature names match your model training features.")
