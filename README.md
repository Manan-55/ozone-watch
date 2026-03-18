# 🌫️ OzoneWatch — Air Quality Risk Prediction System

> A live end-to-end Machine Learning system that predicts dangerous ozone level days using real-time atmospheric data from Houston, TX.

---

## 🚀 Live Demo
👉 [Launch OzoneWatch App](#) ← add your Streamlit link here

---

## 🎯 Problem Statement
Houston, Texas records 60+ ozone alert days per year. Ground-level ozone causes serious respiratory issues and requires early warnings for public safety.

**Goal:** Can we predict a dangerous ozone day before it happens using atmospheric weather readings?

---

## 📊 Dataset
- **Source:** UCI ML Repository via OpenML
- **Size:** 2534 days of atmospheric data from Houston, TX (1998-2004)
- **Features:** 72 atmospheric measurements — temperature, wind speed, humidity, solar radiation, pressure
- **Target:** Binary — Normal Day (0) or Ozone Day (1)
- **Challenge:** Severe class imbalance — 94% Normal, 6% Ozone days

---

## 🔬 What I Did

### 1. Exploratory Data Analysis
- Discovered 94/6 class imbalance — making accuracy a misleading metric
- Found multicollinearity among temperature features (V40-V44 correlation: 0.96-0.99)
- Identified that ozone days cluster at higher temperatures (25-35°C) and lower wind speeds (0-3 km/h)

### 2. Data Preprocessing
- Handled ARFF format conversion and missing value treatment
- Applied **SMOTE** to balance training data from 127 → 1900 ozone samples

### 3. Model Building & Comparison
| Model | Precision | Recall | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.27 | **0.91** | 0.42 |
| Random Forest | 0.43 | 0.36 | 0.39 |
| XGBoost | 0.36 | 0.42 | 0.39 |

**Selected Logistic Regression** — highest recall (0.91) means only 3 out of 33 ozone days missed. For a public health system, missing a dangerous day is far costlier than a false alarm.

### 4. Model Explainability
- Implemented **SHAP values** to explain individual predictions
- Identified solar radiation (V62) as the strongest predictor of ozone days

### 5. Threshold Tuning
- Optimized decision threshold from 0.5 → 0.91
- Improved F1-score from 0.42 → 0.49 while maintaining acceptable recall

### 6. Deployment
- Built live web app using **Streamlit**
- Integrated **Open-Meteo API** to fetch real-time Houston weather automatically
- App predicts today's ozone risk using actual current atmospheric conditions

---

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)

- **Language:** Python 3.12
- **ML:** Scikit-learn, XGBoost, Imbalanced-learn
- **Explainability:** SHAP
- **Deployment:** Streamlit
- **API:** Open-Meteo (real-time weather)
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## 📁 Project Structure
```
ozone-watch/
│
├── data/
│   └── ozone-level-8hr.arff    # Raw dataset
│
├── ozone_watch.ipynb            # Full ML pipeline notebook
├── app.py                       # Streamlit web application
├── model.pkl                    # Saved trained model
├── features.pkl                 # Feature columns
└── README.md
```

---

## ⚡ Run Locally
```bash
git clone https://github.com/Manan-55/ozone-watch.git
cd ozone-watch
pip install -r requirements.txt
streamlit run app.py
```

---

## 💡 Key Insights
1. **Accuracy is misleading** — 94% accuracy achievable by predicting Normal always. Used F1 and Recall instead.
2. **SMOTE over random oversampling** — creates synthetic realistic samples preserving data distribution
3. **Recall > Precision** for public health — missing an ozone day is more dangerous than a false alarm
4. **Solar radiation is the strongest predictor** — aligns with known science that sunlight triggers ozone formation

---

## 👤 Author
**Manan Prajapati**
- 📧 mananprajapati82@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/manan-prajapati-bb4757319)
- 🐙 [GitHub](https://github.com/Manan-55)
