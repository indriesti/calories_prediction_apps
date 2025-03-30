import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
with open("best_model_calories.pkl", "rb") as file:
    model = pickle.load(file)

with open("label_encoder.pkl", "rb") as f:
    le_gender = pickle.load(f)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit App Configuration
st.set_page_config(page_title="Calories Prediction", page_icon="🔥", layout="wide")

# Sidebar Navigation
st.sidebar.image("calories_icon.jpg", width=250)
st.sidebar.title("⚡ App Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Predict"])

# Home Page
if page == "Home":
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>🔥 Calories Burn Prediction App 💪 </h1>
            <h3> Hey There! Ready to Track Your Calories? 👋</h3>
            <p>This app helps to estimate the number of calories burned based on your physical attributes and exercise details.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display image in the center
    st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://thumbs.dreamstime.com/b/illustration-olympic-sports-player-223664607.jpg" width="600">
    </div>
    """,
    unsafe_allow_html=True
)


    # More engaging description
    st.markdown(
        """
        ## 🌟 Why Use This App?
        ✅ Uses a **trained Machine Learning model** for accurate calorie predictions.  
        ✅ Provides an **interactive UI** for easy input and insights.  
        ✅ Displays **model performance metrics** and **data visualizations**.  
        ✅ Helps you **analyze and optimize your workouts** effectively.  

        ### 📌 How to Use:
        1️⃣ **Enter your personal details** 🏋️ (Height, Weight, etc.)  
        2️⃣ **Input your workout details** ⏱️ (Duration, Heart Rate, etc.)  
        3️⃣ **Click ‘Predict’** 🔥 and get instant results!  
        4️⃣ **Analyze the insights & improve your fitness plan** 📊  

        ---
        💡 **Ready to optimize your training? Start predicting now!**
        """,
        unsafe_allow_html=True
    )

elif page == "Predict":
    st.title("🎯 Predict Your Calories Burned")

    # Create columns for better UI layout
    col1, col2 = st.columns(2)

    with col1:
        height = float(st.number_input("📏 Height (cm)", min_value=100, max_value=250, value=170, step=1))
        weight = float(st.number_input("⚖ Weight (kg)", min_value=20, max_value=200, value=70, step=1))
        duration = float(st.number_input("⏳ Exercise Duration (minutes)", min_value=1, max_value=240, value=30, step=1))
        age = float(st.number_input("🎂 Age", min_value=10, max_value=100, value=25))

    with col2:
        heart_rate = float(st.number_input("❤️ Heart Rate (bpm)", min_value=40, max_value=200, value=90, step=1))
        body_temp = float(st.number_input("🌡 Body Temperature (°C)", min_value=30.0, max_value=47.0, value=37.5, step=0.1))
        gender = st.selectbox("⚧ Gender", ["Male", "Female"])

    # Pindahkan gender_numeric ke dalam blok
    gender_numeric = 1.0 if gender == "Male" else 0.0

    # Prepare input data (ensure the order matches the training data)
    input_data = np.array([[gender_numeric, age, height, weight, duration, heart_rate, body_temp]], dtype=np.float64)

    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    # Prediction Button
    if st.button("🚀 Predict Calories Burned"):
        try:
            prediction = model.predict(input_data_scaled)[0]
            prediction = max(prediction, 0)
            st.success(f"🔥 Estimated Calories Burned: **{prediction:.2f} kcal**")
            st.toast("Prediction Complete! 🎯")
        except Exception as e:
            st.error(f"⚠️ Error in prediction: {e}")

# Footer
st.sidebar.info("📌 Built by Indriani Estiningtyas 🚀")