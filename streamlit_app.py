import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================
# 🌿 Tampilan Modern dan Fresh
# ==============================
st.set_page_config(page_title="Prediksi Risiko Obesitas", page_icon="🩺", layout="wide")
st.markdown("""
<style>
    body {
        background: linear-gradient(120deg, #e0f7fa, #f8fdff);
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding: 2rem 2rem 1rem;
    }
    h1 {
        color: #00695c;
        font-weight: 700;
        margin-bottom: 0.2em;
    }
    .subheader-text {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5em;
    }
    .stButton>button {
        background-color: #009688;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .stSelectbox>div>div {
        background-color: #f1f8f9;
    }
    .stSlider>div {
        background-color: #f1f8f9;
    }
</style>
""", unsafe_allow_html=True)

st.title("Prediksi Risiko Obesitas")
st.markdown("""
<div class="subheader-text">
Gunakan aplikasi ini untuk memperkirakan tingkat risiko obesitas berdasarkan gaya hidup dan informasi fisik Anda.
Aplikasi ini bersifat edukatif dan tidak menggantikan konsultasi medis profesional.
</div>
""", unsafe_allow_html=True)

# ==============================
# 📥 Formulir Input
# ==============================
st.markdown("### Form Pemeriksaan Harian")
with st.form(key='obesity_form'):
    col1, col2, col3 = st.columns([1.2, 1.2, 1])

    with col1:
        age = st.slider('Usia', 10, 100, 25)
        gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
        height = st.number_input('Tinggi Badan (m)', 1.0, 2.5, step=0.01, value=1.70)
        weight = st.number_input('Berat Badan (kg)', 30.0, 200.0, step=0.1, value=70.0)
        family_history = st.selectbox('Riwayat Obesitas Keluarga', ['ya', 'tidak'])

    with col2:
        FAVC = st.selectbox('Sering Makanan Tinggi Kalori?', ['ya', 'tidak'])
        FCVC = st.slider('Frekuensi Makan Sayur (1-3)', 1.0, 3.0, 2.0)
        NCP = st.slider('Jumlah Makan per Hari', 1.0, 4.0, 3.0)
        CAEC = st.selectbox('Ngemil?', ['tidak', 'Kadang-kadang', 'Sering', 'Selalu'])
        CH2O = st.slider('Konsumsi Air Harian (L)', 1.0, 3.0, 2.0)

    with col3:
        SCC = st.selectbox('Pernah Mengurangi Makan?', ['ya', 'tidak'])
        SMOKE = st.selectbox('Merokok?', ['ya', 'tidak'])
        FAF = st.slider('Aktivitas Fisik / Minggu (jam)', 0.0, 5.0, 2.0)
        TUE = st.slider('Waktu Layar / Hari (jam)', 0.0, 5.0, 2.0)
        CALC = st.selectbox('Konsumsi Alkohol?', ['tidak', 'Kadang-kadang', 'Sering', 'Selalu'])
        MTRANS = st.selectbox('Transportasi Utama', ['Transportasi Umum', 'Jalan Kaki', 'Mobil', 'Motor', 'Sepeda'])

    submit_button = st.form_submit_button(label='Prediksi Sekarang')

# ==============================
# 🔮 Proses Prediksi
# ==============================
if submit_button:
    try:
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
            'CH2O': [CH2O],
            'SCC': ["yes" if SCC == "ya" else "no"],
            'SMOKE': ["yes" if SMOKE == "ya" else "no"],
            'FAF': [FAF],
            'TUE': [TUE],
            'CALC': [CALC],
            'MTRANS': [MTRANS.replace(" ", "_").replace("Transportasi_Umum", "Public_Transportation")]
        })

        model = pickle.load(open('rf_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        expected_columns = pickle.load(open('columns.pkl', 'rb'))

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
            st.success(f"Hasil Prediksi: {hasil}\n\n✅ Berat badan Anda termasuk normal. Pertahankan gaya hidup sehat!")
        else:
            st.warning(f"Hasil Prediksi: {hasil}\n\n⚠️ Berat badan Anda tidak dalam kategori normal. Perlu perhatian lebih terhadap gaya hidup Anda.")

    except Exception as e:
        st.error("Terjadi kesalahan saat memproses prediksi. Pastikan file model tersedia.")
        st.exception(e)
