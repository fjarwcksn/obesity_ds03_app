# 🩺 Obesity Risk Prediction App

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A modern and interactive **Streamlit web application** to predict an individual's **obesity level** based on health and lifestyle inputs such as age, diet, physical activity, water intake, and more.

This project was developed as part of the **Capstone Project – Bengkel Koding Data Science Program**.

---

## 🚀 Live App

🔗 [Click here to open the app on Streamlit Cloud](https://obesityds03.streamlit.app/)  
📱 Responsive UI for both desktop and mobile users.

---

## 💡 Project Highlights

- ✅ Built with Python and Streamlit
- 🧠 Machine Learning model using **Random Forest**
- ⚙️ Features real-time prediction with clean UI
- 🧪 Tested with multiple user cases
- 📊 Input includes both numeric and categorical health indicators
- 📦 Fully deployable to Streamlit Cloud or local machine

---

## 🧠 Models Evaluated

During development, we tested three classification algorithms:

| Model                  | Accuracy  | Notes                      |
|------------------------|-----------|----------------------------|
| Logistic Regression    | ~71%      | Simple & fast baseline     |
| Random Forest ✅       | **~94%**  | Best performance, deployed |
| Support Vector Machine | ~78%      | Moderate performance       |

🟢 **Random Forest** was selected and deployed as the final model.

---

## 🔍 How It Works

Users are asked to fill out a health screening form which includes:
- Demographics (age, gender, height, weight)
- Lifestyle habits (diet, snacking, smoking, alcohol)
- Physical activity (exercise frequency, water intake)
- Screen time and transport type

🧠 The model classifies obesity into one of the following categories:

- `Insufficient_Weight`
- `Normal_Weight`
- `Overweight_Level_I`
- `Overweight_Level_II`
- `Obesity_Type_I`
- `Obesity_Type_II`
- `Obesity_Type_III`

---

## 📁 Project Structure

| File               | Description                          |
|--------------------|--------------------------------------|
| `streamlit_app.py` | Streamlit frontend application       |
| `rf_model.pkl`     | Trained Random Forest model          |
| `scaler.pkl`       | Preprocessing scaler (StandardScaler)|
| `columns.pkl`      | Feature columns after encoding       |
| `Capstone-DS03.ipynb` | Notebook for EDA, training & evaluation |
| `requirements.txt` | Python dependencies                  |

---

## 💾 Dataset

- Dataset: **Obesity Level Prediction Dataset**
- Source: UCI Machine Learning Repository  
- Link: [View Dataset on Google Drive](https://drive.google.com/file/d/16mZS56ed1SQyimDxRGGKvihvIMri2exM/view)

---

## ⚙️ Run Locally

To run the app on your own machine:

```bash
# Clone the repository
git clone https://github.com/your-username/obesity_ds03_app.git
cd obesity_ds03_app

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run streamlit_app.py

```

---

## 📝 Dataset Info
Dataset used: **[Obesity Level Prediction Dataset](https://drive.google.com/file/d/16mZS56ed1SQyimDxRGGKvihvIMri2exM/view)**  
Source: UCI Machine Learning Repository

---

## 👤 Author
**Fajar Adji W**  
Student - Bengkel Koding Data Science

---

## 📃 License
MIT License (free to use, share, and modify)
