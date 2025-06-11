# 📊 Obesity Level Prediction App

A Streamlit web app for predicting obesity level based on user input features such as age, weight, dietary habits, physical activity, and more.

## 🚀 Live App
[Click here to open the app](https://obesityds03.streamlit.app/) 

---

## 🧠 Models Used
This project includes three classification models:
- Logistic Regression
- Random Forest (best performance)
- Support Vector Machine (SVM)

Random Forest was selected as the final model and deployed.

---

## ⚙️ How It Works
Users fill in a form with health and lifestyle inputs. The model predicts the obesity class:
- Insufficient_Weight
- Normal_Weight
- Overweight_Level_I
- Overweight_Level_II
- Obesity_Type_I
- Obesity_Type_II
- Obesity_Type_III

---

## 📁 Files in this Repository
| File               | Description                          |
|--------------------|--------------------------------------|
| `streamlit_app.py` | Streamlit UI logic                   |
| `rf_model.pkl`     | Trained Random Forest model          |
| `scaler.pkl`       | StandardScaler for preprocessing     |
| `requirements.txt` | List of Python dependencies          |
| `Capstone-DS03.ipynb` | Model development & training notebook |

---

## 📦 Installation & Run Locally
```bash
# Clone the repository
$ git clone https://github.com/your-username/obesity_ds03_app.git
$ cd obesity_ds03_app

# Install dependencies
$ pip install -r requirements.txt

# Run Streamlit app
$ streamlit run streamlit_app.py
```

---

## 📝 Dataset Info
Dataset used: **[Obesity Level Prediction Dataset](https://www.kaggle.com/datasets/sanjanabasu/obesity-prediction)**  
Source: UCI Machine Learning Repository

---

## 👤 Author
**Fajar Adji W**  
Student - Bengkel Koding Data Science

---

## 📃 License
MIT License (free to use, share, and modify)
