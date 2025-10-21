import streamlit as st
import numpy as np
import pickle

#Models
clf = pickle.load(open('btms_classifier.pkl', 'rb'))
reg = pickle.load(open('btms_regressor.pkl', 'rb'))

#Defining features and their realistic valid ranges (example ranges, adjust as needed)
feature_limits = {
    'PackVoltageV': (200, 450),            #200V to 450V
    'CellVoltageV': (3.0, 4.5),            #3V to 4.5V per cell
    'ChargeCurrentA': (0, 300),             #Charging current 0 to 300 A
    'AvgTempC': (0, 70),                    #Average temp 0 to 70 °C
    'MaxTempC': (0, 90),                    #Max temp 0 to 90 °C
    'InternalResistancemOhm': (0, 100),    #Internal resistance 0 to 100 milliohms
    'PressurekPa': (90, 120),               #Pressure between 90 to 120 kPa
    'SOC': (0, 100)                         #State of charge 0 to 100%
}

st.set_page_config(
    page_title="Battery Thermal Runaway Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 24px;
        margin-top: 20px;
    }
    .stTextInput>div>input {
        font-size: 16px;
        padding: 8px;
        border: 2px solid #4CAF50;
        border-radius: 6px;
        transition: border-color 0.3s;
    }
    .stTextInput>div>input:focus {
        border-color: #388E3C;
        outline: none;
    }
    .title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: bold;
        color: #2E7D32;
    }
    .result-high {
        background-color: #FFCDD2;
        color: #C62828;
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    .result-low {
        background-color: #C8E6C9;
        color: #2E7D32;
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    .subheader-style {
        color: #388E3C;
        font-weight: 600;
        font-size: 22px;
    }
    .warning {
        color: #D84315;
        font-weight: 600;
        font-size: 16px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="title">EV Battery Thermal Runaway Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

st.header("Input Battery Parameters")

inputs = {}
errors = []
empty_inputs = False

for feature, (low, high) in feature_limits.items():
    while True:
        val = st.text_input(f"Enter {feature} (Range: {low} – {high}):", value="0.0")
        try:
            fval = float(val)
            if fval == 0.0:
                empty_inputs = True
            elif not (low <= fval <= high):
                st.warning(f"{feature} should be between {low} and {high}.")
                continue
            inputs[feature] = fval
            break
        except ValueError:
            st.warning(f"Please enter a numeric value for {feature}.")

#if empty_inputs:
   #st.markdown('<div class="warning">Please provide valid values (other than 0) for input parameters to proceed with prediction.</div>', unsafe_allow_html=True)

if st.button("Predict Risk & Probability"):
    if empty_inputs:
        st.error("Prediction cannot be performed because some inputs are zero. Please input valid values within specified ranges.")
    else:
        input_array = np.array([inputs[feat] for feat in feature_limits.keys()]).reshape(1, -1)
        risk_class = clf.predict(input_array)[0]
        risk_prob = reg.predict(input_array)[0]
        risk_status = "High Risk of Thermal Runaway" if risk_class == 1 else "Low Risk"

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="subheader-style">Classification Result</div>', unsafe_allow_html=True)
            if risk_class == 1:
                st.markdown(f'<div class="result-high">{risk_status}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-low">{risk_status}</div>', unsafe_allow_html=True)
            st.write(f"Probability of Risk (from regressor): **{risk_prob:.2f}**")
        with col2:
            st.markdown('<div class="subheader-style">Regression Output</div>', unsafe_allow_html=True)
            st.write(f"Estimated Probability of Thermal Runaway:")
            st.markdown(f"<h2 style='color:#2E7D32'>{risk_prob:.2f}</h2>", unsafe_allow_html=True)
