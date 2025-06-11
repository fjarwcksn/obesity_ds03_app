import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================
# 🎨 Konfigurasi Tampilan UI
# ==============================
st.set_page_config(page_title="Prediksi Risiko Obesitas", page_icon="🩺", layout="wide")
st.markdown("""
<style>
    body {
        background: linear-gradient(120deg, #e0f7fa, #f0f4f8);
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
    }
    h1, h2, h3 {
        color: #009688;
    }
    .stButton>button {
        background-color: #009688;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.image("https://cdn-icons-png.flaticon.com/512/4320/4320337.png", width=80)
st.title("🩺 Aplikasi Prediksi Risiko Obesitas")
st.markdown("""
Selamat datang di aplikasi prediksi obesitas! 🧠 
Gunakan aplikasi ini untuk mengetahui estimasi tingkat obesitas berdasarkan kebiasaan harian dan data fisik kamu.

⚠️ Hasil ini bersifat indikatif dan tidak menggantikan diagnosis dokter.
""")

# ==============================
# 📥 Formulir Input Pengguna
# ==============================
st.markdown("### 📋 Formulir Pemeriksaan Kesehatan Harian")
with st.form(key='obesity_form'):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider('Usia', 10, 100, 25)
        gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
        height = st.number_input('Tinggi Badan (meter)', 1.0, 2.5, step=0.01, value=1.70)
        weight = st.number_input('Berat Badan (kg)', 30.0, 200.0, step=0.1, value=70.0)
        family_history = st.selectbox('Ada riwayat obesitas di keluarga?', ['ya', 'tidak'])
        FAVC = st.selectbox('Sering makan makanan tinggi kalori?', ['ya', 'tidak'])
        FCVC = st.slider('Frekuensi makan sayur (1=jarang, 3=sering)', 1.0, 3.0, 2.0)

    with col2:
        NCP = st.slider('Jumlah makan utama per hari', 1.0, 4.0, 3.0)
        CAEC = st.selectbox('Kebiasaan ngemil?', ['tidak', 'Kadang-kadang', 'Sering', 'Selalu'])
        SMOKE = st.selectbox('Apakah kamu merokok?', ['ya', 'tidak'])
        CH2O = st.slider('Konsumsi air harian (liter)', 1.0, 3.0, 2.0)
        SCC = st.selectbox('Apakah pernah mengurangi makanan?', ['ya', 'tidak'])
        FAF = st.slider('Aktivitas fisik mingguan (jam)', 0.0, 5.0, 2.0)
        TUE = st.slider('Waktu layar per hari (jam)', 0.0, 5.0, 2.0)
        CALC = st.selectbox('Konsumsi alkohol?', ['tidak', 'Kadang-kadang', 'Sering', 'Selalu'])
        MTRANS = st.selectbox('Transportasi utama', ['Transportasi Umum', 'Jalan Kaki', 'Mobil', 'Motor', 'Sepeda'])

    submit_button = st.form_submit_button(label='🔍 Prediksi Sekarang')

# ==============================
# 🔮 Prediksi dengan Model
# ==============================
if submit_button:
    try:
        # DataFrame dari input
        user_input = pd.DataFrame({
            'Age': [age],
            'Gender': ["Male" if gender == "Laki-laki" else "Female"],
            'Height': [height],
            'Weight': [weight],
            'family_history_with_overweight': ["yes" if family_history == "ya" else "no"],
            'FAVC': ["yes" if FAVC == "ya" else "no"],
            'FCVC': [FCVC],
            'NCP': [NCP],
            'CAEC': [CAEC],
            'SMOKE': ["yes" if SMOKE == "ya" else "no"],
            'CH2O': [CH2O],
            'SCC': ["yes" if SCC == "ya" else "no"],
            'FAF': [FAF],
            'TUE': [TUE],
            'CALC': [CALC],
            'MTRANS': [MTRANS.replace(" ", "_").replace("Transportasi_Umum", "Public_Transportation")]
        })

        # Load model dan scaler
        model = pickle.load(open('rf_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        expected_columns = pickle.load(open('columns.pkl', 'rb'))

        # Encoding dan penyesuaian kolom
        user_input_encoded = pd.get_dummies(user_input)
        input_data = pd.DataFrame(0, index=[0], columns=expected_columns)
        for col in user_input_encoded.columns:
            if col in input_data.columns:
                input_data.at[0, col] = user_input_encoded.at[0, col]

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        label_map = ['Berat Badan Kurang', 'Normal', 'Obesitas Tipe I', 'Obesitas Tipe II',
                     'Obesitas Tipe III', 'Kelebihan Berat Badan I', 'Kelebihan Berat Badan II']

        hasil = label_map[prediction]
        if hasil == 'Normal':
            st.success(f"✅ Hasil Prediksi: **{hasil}**\n\nBerat badan Anda termasuk normal. Pertahankan pola hidup sehat dan aktif!")
        else:
            st.warning(f"⚠️ Hasil Prediksi: **{hasil}**\n\nBerat badan Anda tidak dalam kategori normal. Disarankan untuk menjaga pola makan dan aktivitas.")

    except Exception as e:
        st.error("Terjadi kesalahan saat memproses prediksi. Pastikan file `rf_model.pkl`, `scaler.pkl`, dan `columns.pkl` tersedia.")
        st.exception(e)
