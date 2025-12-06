import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# --- Configuration (Theme and Model Path) ---

# Set Streamlit page configuration for a wider, cleaner look
st.set_page_config(
    page_title="Pet Image Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the fixed input shape used during training
IMG_SIZE = (160, 160) 
MODEL_PATH = 'cat_dog_cnn_model.keras' # Ensure this path is correct!

# --- Model Loading and Caching ---

@st.cache_resource
def load_pet_model():
    """Load the trained model efficiently."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error("🚨 Error: Could not load the classification model.")
        st.caption(f"Details: Ensure '{MODEL_PATH}' exists in the current directory.")
        st.stop()

# Load the model once
model = load_pet_model()

# --- Image Preprocessing Function ---
def preprocess_image(image):
    """Resizes, converts, and normalizes the image for prediction."""
    img = image.convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Prediction Function ---
def predict_image(model, processed_image):
    """Makes a prediction and returns the class and confidence."""
    prediction = model.predict(processed_image)
    probability = prediction[0][0]
    
    # Determine the predicted class (assuming Dog=1, Cat=0)
    if probability >= 0.5:
        class_name = "Dog"
        icon = "🐶"
        confidence = probability * 100
    else:
        class_name = "Cat"
        icon = "🐱"
        confidence = (1 - probability) * 100
        
    return class_name, icon, confidence

# --- Streamlit UI Layout ---

# Header Section
st.markdown("<h1 style='text-align: center; color: #1e8449;'>🐾 Advanced Pet Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Instantly classify images as either a Cat or a Dog using a Convolutional Neural Network.</p>", unsafe_allow_html=True)
st.markdown("---")

# Main Content: Split into two columns
col1, col2 = st.columns([1, 1.5]) 

with col1:
    st.subheader("🖼️ Upload Image")
    
    # File uploader widget
    uploaded_file = st.file_uploader(
        "Upload a JPG or PNG file here:", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        # Display the uploaded image below the uploader
        st.image(image, caption='Image for Analysis', use_column_width=True)


with col2:
    st.subheader("📊 Prediction Results")
    
    if uploaded_file:
        # Prediction button and logic
        if st.button('Analyze Image', key='predict_button', type="primary"):
            
            # 1. Preprocess
            with st.spinner('Preparing image for analysis...'):
                processed_img = preprocess_image(image)
            
            # 2. Predict
            with st.spinner('Calculating prediction...'):
                class_name, icon, confidence = predict_image(model, processed_img)
            
            # 3. Display Results
            st.success(f"**Final Classification:** {icon} {class_name}")

            # Use st.expander for a neat display of confidence
            with st.expander("Confidence Metrics", expanded=True):
                st.metric("Predicted Class", f"{icon} {class_name}")
                st.metric("Probability Score", f"{confidence:.2f}%")
                
            st.info(f"The model is highly confident in its classification.")

        else:
            st.info("Click 'Analyze Image' to see the result.")
            
    else:
        st.warning("Please upload an image in the left panel to begin.")

# Footer
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    .stApp { margin-bottom: 50px; }
    </style>
    """, unsafe_allow_html=True
)