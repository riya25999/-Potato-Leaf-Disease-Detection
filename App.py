import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potato_model.h5")

model = load_model()

# ---------------- TITLE ----------------
st.title("🥔 Potato Leaf Disease Detection")

# ---------------- CLASS NAMES ----------------
class_names = ["Early Blight", "Late Blight", "Healthy"]

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Potato Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # 🔥 IMPORTANT: Correct size (64x64 from training)
    img = image.resize((64, 64))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    if st.button("🔍 Predict Disease"):

        prediction = model.predict(img)

        result = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.success(f"🌿 Disease: {result}")
        st.info(f"📊 Confidence: {confidence * 100:.2f}%")