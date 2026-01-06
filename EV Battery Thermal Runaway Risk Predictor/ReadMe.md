# 🔋 EV Battery Thermal Runaway Risk Predictor

A machine learning–powered web application that predicts the **risk and probability of thermal runaway in EV batteries** based on real-time battery parameters.  
Built with a focus on **safety analytics**, **explainability**, and **practical deployment** using Streamlit.

---

## Short Description

This project uses **XGBoost-based classification and regression models** to assess whether an electric vehicle (EV) battery is at **high or low risk of thermal runaway**, and estimates the **probability of occurrence** using key electrical, thermal, and environmental features.

The application is designed as a **demo/educational project** showcasing applied machine learning, model deployment, and risk prediction in EV systems.

---

## Tech Stack / Tools Used

- **Programming Language:** Python 🐍  
- **Web Framework:** Streamlit  
- **Machine Learning:** XGBoost  
- **Data Processing:** NumPy, Pandas  
- **Model Serialization:** Pickle  
- **Environment:** Local / Virtual Environment  

---

## Key Features

-  Predicts **thermal runaway risk (High / Low)**
-  Estimates **probability of thermal runaway**
-  Uses **separate classification and regression models**
-  Interactive UI with input validation and safety ranges
-  Lightweight, fast, and beginner-friendly deployment
-  Clean, professional Streamlit interface

---

## Dataset / Inputs

The model expects numerical battery parameters such as:

- Pack Voltage (V)
- Cell Voltage (V)
- Charge Current (A)
- State of Charge (%)
- Maximum Temperature (°C)
- Average Temperature (°C)
- Internal Resistance (mΩ)
- Pressure (kPa)

> Note: The dataset used for training is for **demonstration purposes only** and does not represent real-world certified EV battery data.

---

## How It Works (High-Level)

1. User inputs battery parameters through the Streamlit UI
2. Inputs are validated against predefined safe operating ranges
3. Features are passed to:
   - **XGBoost Classifier** → predicts risk category
   - **XGBoost Regressor** → estimates probability
4. Results are displayed with clear visual indicators
5. User receives both **classification + probability output**

---

## Model Performance & Evaluation

### Evaluation Metrics Used

- **Classifier**
  - Accuracy
  - Precision
  - Recall
  - F1-score
- **Regressor**
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score

### Model Performance Summary

- Classifier performs well on balanced demo data
- Regression model provides smooth probability estimates
- Performance may vary significantly with real-world data

---

## Limitations

- Trained on **synthetic / limited dataset**
- Not validated on real EV battery systems
- Probability output is **indicative, not safety-certified**
- Does not replace physical battery management systems (BMS)

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/ev-battery-thermal-runaway.git
cd ev-battery-thermal-runaway

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Results / Output

- 🔴 High Risk or 🟢 Low Risk classification
- 📊 Estimated probability of thermal runaway
- Clear UI feedback for safer decision interpretation

---

⭐ If you found this project helpful, consider giving it a star!
