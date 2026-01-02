# Data Science Intel – Next Word Prediction App

An elegant **Next Word Prediction** web application built using **Deep Learning (LSTM)** and **Streamlit**.  
This project demonstrates my understanding of **NLP pipelines, model loading, tokenization, and real-time inference** in a clean, interactive UI.

> **Important Note**  
> **Try changing or improving the data — this is an example demonstrating knowledge of mine.**  
> The goal of this project is learning, experimentation, and showcasing concepts rather than production-level accuracy.

---

## Project Overview

This application predicts the **next sequence of words** based on user input using a trained neural network model.  
As the user types, the model generates text in real-time, visually separating **user input** and **model predictions** for clarity.

---

## Key Concepts Demonstrated

- Natural Language Processing (NLP)
- Tokenization & Padding
- Sequence Modeling with LSTM
- Model serialization (`.h5`, `.pkl`)
- Real-time predictions
- Streamlit UI/UX customization
- Caching ML assets for performance

---

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Streamlit**
- **Pickle**

---

## How It Works

1. User types text into the input box
2. Text is tokenized and padded
3. The model predicts the most probable next word
4. The prediction is appended to the input
5. Steps repeat to generate a short sequence

The predicted words appear in a **ghost-text style**, making the model’s “thought process” easy to visualize.

---

## Running the App Locally
streamlit run app.py

---

## Experimentation & Learning

- This project is intentionally designed to be modifiable:
- Replace the training dataset
- Retrain the model with better text corpora
- Adjust sequence length
- Improve prediction logic
- Enhance UI styling


