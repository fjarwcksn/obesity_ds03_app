import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================
# 🌐 International Medical Theme
# ==============================
st.set_page_config(page_title="Obesity Risk Predictor", page_icon="🩺", layout="wide")
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #f0f4f8 0%, #e0f7fa 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        padding: 2rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
    }
    h1, h2, h3 {
        color: #00796b;
    }
    .stButton>button {
        background-color: #00796b;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.image("https://cdn-icons-png.flaticon.com/512/5997/5997770.png", width=90)
st.title("🩺 Obesity Risk Prediction App")
st.markdown("""
This application uses a **machine learning model** to predict the level of obesity based on your lifestyle and health metrics. 

🔍 Just fill in the form and get an instant result. This tool is for **educational and early awareness purposes only**.
""")

# ==============================
# 📥 Input Form
# ==============================
st.markdown("### 📋 Daily Health Assessment Form")
with st.form(key='obesity_form'):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider('Age', 10, 100, 25)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        height = st.number_input('Height (m)', 1.0, 2.5, step=0.01, value=1.70)
        weight = st.number_input('Weight (kg)', 30.0, 200.0, step=0.1, value=70.0)
        family_history = st.selectbox('Family history of obesity?', ['yes', 'no'])
        FAVC = st.selectbox('High-calorie food frequently consumed?', ['yes', 'no'])
        FCVC = st.slider('Frequency of vegetable consumption (1-3)', 1.0, 3.0, 2.0)

    with col2:
        NCP = st.slider('Number of main meals per day', 1.0, 4.0, 3.0)
        CAEC = st.selectbox('Snacking habits?', ['no', 'Sometimes', 'Frequently', 'Always'])
        SMOKE = st.selectbox('Do you smoke?', ['yes', 'no'])
        CH2O = st.slider('Daily water intake (liters)', 1.0, 3.0, 2.0)
        SCC = st.selectbox('Have you monitored your calories?', ['yes', 'no'])
        FAF = st.slider('Physical activity per week (hrs)', 0.0, 5.0, 2.0)
        TUE = st.slider('Daily screen time (hrs)', 0.0, 5.0, 2.0)
        CALC = st.selectbox('Alcohol consumption?', ['no', 'Sometimes', 'Frequently', 'Always'])
        MTRANS = st.selectbox('Primary mode of transport', ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

    submit_button = st.form_submit_button(label='Predict Now')

# ==============================
# 🔮 Load Model & Predict
# ==============================
if submit_button:
    try:
        # DataFrame Input
        user_input = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Height': [height],
            'Weight': [weight],
            'family_history_with_overweight': [family_history],
            'FAVC': [FAVC],
            'FCVC': [FCVC],
            'NCP': [NCP],
            'CAEC': [CAEC],
            'SMOKE': [SMOKE],
            'CH2O': [CH2O],
            'SCC': [SCC],
            'FAF': [FAF],
            'TUE': [TUE],
            'CALC': [CALC],
            'MTRANS': [MTRANS]
        })

        # Load model, scaler, dan expected columns
        model = pickle.load(open('rf_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        expected_columns = pickle.load(open('columns.pkl', 'rb'))

        # One-hot encoding dan align kolom
        user_input_encoded = pd.get_dummies(user_input)
        input_data = pd.DataFrame(0, index=[0], columns=expected_columns)
        for col in user_input_encoded.columns:
            if col in input_data.columns:
                input_data.at[0, col] = user_input_encoded.at[0, col]

        # Scaling dan prediksi
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        label_map = ['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I', 'Obesity_Type_II',
                     'Obesity_Type_III', 'Overweight_Level_I', 'Overweight_Level_II']

        kategori = label_map[prediction]
        if kategori == 'Normal_Weight':
            st.success(f"✅ Prediction: **{kategori}**\n\nYour body weight is in the healthy range. Keep up the good lifestyle!")
        else:
            st.warning(f"⚠️ Prediction: **{kategori}**\n\nThis indicates a non-ideal body weight. Please consult your physician or adopt healthier habits.")

    except Exception as e:
        st.error("An error occurred while processing your prediction. Please ensure all model files are available.")
        st.exception(e)
