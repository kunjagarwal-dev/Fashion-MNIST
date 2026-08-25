import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------- Page Config ----------
st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👕",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    .title-text {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #06b6d4, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .subtitle-text {
        text-align: center;
        color: #9ca3af;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #06b6d4, #a855f7);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-top: 15px;
    }
    .prediction-label {
        font-size: 36px;
        font-weight: 900;
        color: white;
        margin: 0;
    }
    .confidence-text {
        color: #f3f4f6;
        font-size: 16px;
        margin-top: 5px;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #06b6d4, #a855f7);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ---------- Load TFLite Model ----------
@st.cache_resource
def get_interpreter():
    interpreter = tf.lite.Interpreter(model_path="models/fashion_mobilenet_quant.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = get_interpreter()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = input_details[0]['shape'][1]  # e.g. 96

# ---------- Header ----------
st.markdown('<p class="title-text">👕 Fashion Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Upload a clothing image — powered by a quantized TFLite model</p>', unsafe_allow_html=True)

# ---------- Layout ----------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### 📤 Upload image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)
    predict_btn = st.button("🔮 Predict")

with col2:
    st.markdown("#### 📊 Result")
    result_placeholder = st.empty()
    chart_placeholder = st.empty()

    if not predict_btn:
        result_placeholder.info("Upload an image and click **Predict** to see the result here.")

# ---------- Prediction Logic ----------
if predict_btn:
    if uploaded_file is None:
        result_placeholder.warning("Please upload an image first.")
    else:
        img = Image.open(uploaded_file).convert("L")           # grayscale
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img).astype("float32")
        img_array = np.stack([img_array] * 3, axis=-1)          # grayscale -> 3-channel
        img_array = (img_array / 127.5) - 1.0                   # MobileNetV2 preprocessing
        img_array = np.expand_dims(img_array, axis=0).astype(input_details[0]['dtype'])

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_class = CLASS_NAMES[int(np.argmax(predictions))]
        confidence = float(np.max(predictions)) * 100

        result_placeholder.markdown(f"""
            <div class="prediction-box">
                <p class="prediction-label">{predicted_class}</p>
                <p class="confidence-text">Confidence: {confidence:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        chart_placeholder.bar_chart(
            {"Confidence": predictions},
            height=250,
        )

# ---------- Footer ----------
st.markdown("---")
st.caption("Powered by a quantized TFLite MobileNetV2 model (2.55 MB, ~90.85% test accuracy) · TensorFlow + Streamlit")