import streamlit as st
import numpy as np
import pickle
from st_keyup import st_keyup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Page Configuration
st.set_page_config(page_title="Data Science Predictor", layout="centered")

#Custom CSS for Elegance
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput > div > div > input {
        background-color: #1a1c24;
        color: #ffffff;
        border-radius: 10px;
        border: 1px solid #3d4150;
        padding: 15px;
    }
    .prediction-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 20px;
    }
    .user-text {
        color: #ffffff;
        font-size: 24px;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    .ghost-text {
        color: #6c757d;
        font-size: 24px;
        font-style: italic;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = load_model('next_word_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

#Header Section
st.title("✨ Data Science Intel")
st.markdown("<p style='color: #888;'>Start typing to see the model's train of thought in real-time.</p>", unsafe_allow_html=True)

#Interactive Input using st_keyup for real-time response
user_input = st_keyup("", placeholder="Type 'Data science' or 'Healthcare'...", key="elegant_input")

if user_input:
    #Prediction logic
    generated_text = user_input
    num_to_predict = 8  
    
    for _ in range(num_to_predict):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=80, padding='pre')
        
        #Predict
        predicted_probs = model.predict(token_list, verbose=0)
        pos = np.argmax(predicted_probs)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == pos:
                output_word = word
                break
        
        if not output_word:
            break
        
        generated_text += " " + output_word

    #The Elegant Display
    prediction_only = generated_text[len(user_input):]
    
    st.markdown(f"""
        <div class="prediction-container">
            <span class="user-text">{user_input}</span><span class="ghost-text">{prediction_only}</span>
        </div>
        """, unsafe_allow_html=True)
else:
    #Empty State
    st.markdown("""
        <div style="text-align: center; padding: 50px; color: #444;">
            <p>Waitng for your input...</p>
        </div>
        """, unsafe_allow_html=True)