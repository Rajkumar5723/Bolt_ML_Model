import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Check if GPU is available
if len(tf.config.list_physical_devices('GPU')) > 0:
    st.success("🚀 GPU is available! Running on CUDA.")
else:
    st.warning("⚠️ No GPU found. Running on CPU.")

# Load the trained model
MODEL_PATH = "bolt_tightness_model_resnet.h5"  # Ensure model is in the same directory
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Check the path and try again.")
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))  # ResNet input size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_image(img):
    img_array = preprocess_image(img)
    
    try:
        # Use GPU if available
        with tf.device('/GPU:0'):
            prediction = model.predict(img_array)[0][0]
    except Exception as e:
        st.warning("⚠️ Running on CPU (GPU not detected)")
        prediction = model.predict(img_array)[0][0]
    
    # Interpret results
    label = "Loose" if prediction > 0.5 else "Tight"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# Streamlit UI
st.title("🔩 Bolt Tightness Detection")
st.write("Upload an image to check if the bolt is **Tight or Loose**.")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    st.write("🕵️ Analyzing the image...")
    label, confidence = predict_image(img)
    
    # Show prediction result
    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence:.2f}")

st.write("💡 This model uses **ResNet50V2** for classification.")