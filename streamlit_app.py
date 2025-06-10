import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================
# 🎨 App UI Configuration
# ==============================
st.set_page_config(page_title="Prediksi Obesitas", page_icon="🩺", layout="centered")
st.markdown("""
<style>
    .main {
        background-color: #f4f6f9;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🩺 Aplikasi Prediksi Tingkat Obesitas")
st.write("""
Aplikasi ini menggunakan model Machine Learning untuk memprediksi tingkat obesitas berdasarkan data kebiasaan harian dan informasi fisik pengguna.
Silakan isi formulir berikut untuk mendapatkan hasil prediksi.
""")

# ==============================
# 📥 Input Form
# ==============================
with st.form(key='obesity_form'):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider('Usia', 10, 100, 25)
        gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
        height = st.number_input('Tinggi Badan (m)', 1.0, 2.5, step=0.01, value=1.70)
        weight = st.number_input('Berat Badan (kg)', 30.0, 200.0, step=0.1, value=70.0)
        family_history = st.selectbox('Riwayat Keluarga Obesitas', ['yes', 'no'])
        FAVC = st.selectbox('Konsumsi Makanan Tinggi Kalori?', ['yes', 'no'])
        FCVC = st.slider('Frekuensi Konsumsi Sayur (1-3)', 1.0, 3.0, 2.0)

    with col2:
        NCP = st.slider('Jumlah Makan per Hari', 1.0, 4.0, 3.0)
        CAEC = st.selectbox('Cemilan di Luar Jam Makan?', ['no', 'Sometimes', 'Frequently', 'Always'])
        SMOKE = st.selectbox('Apakah Merokok?', ['yes', 'no'])
        CH2O = st.slider('Konsumsi Air Harian (liter)', 1.0, 3.0, 2.0)
        SCC = st.selectbox('Pernah Mengurangi Makan?', ['yes', 'no'])
        FAF = st.slider('Aktivitas Fisik Mingguan (jam)', 0.0, 5.0, 2.0)
        TUE = st.slider('Waktu Layar Harian (jam)', 0.0, 5.0, 2.0)
        CALC = st.selectbox('Konsumsi Alkohol?', ['no', 'Sometimes', 'Frequently', 'Always'])
        MTRANS = st.selectbox('Alat Transportasi Utama', ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

    submit_button = st.form_submit_button(label='🔍 Prediksi Sekarang')

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

        st.success(f"✅ Hasil Prediksi: **{label_map[prediction]}**")

    except Exception as e:
        st.error("Terjadi kesalahan saat memproses prediksi. Pastikan file `rf_model.pkl`, `scaler.pkl`, dan `columns.pkl` tersedia.")
        st.exception(e)
