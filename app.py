import os
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import time
import base64

HF_API_TOKEN = "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"
IMAGE_MODEL = "Salesforce/blip-image-captioning-large" # Image captioning model for landmark detection
TEXT_MODEL = "facebook/bart-large-cnn"  # Text generation model for detailed description
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-{tgt}"  # Translation model

st.set_page_config(
    page_title="Landmark Description App",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🌍 Landmark Description App")
st.markdown("Discover detailed descriptions of landmarks around the world. Upload an image, and let the AI describe it for you!")


def analyze_image(image):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    API_URL = f"https://api-inference.huggingface.co/models/{IMAGE_MODEL}"
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            data=buffered.getvalue(),
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and 'generated_text' in result[0]:
                return result[0]['generated_text'].strip().capitalize()
        return "Landmark Not Recognized"
    except Exception as e:
        return "Error: Unable to process the image."

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    translate_option = st.checkbox("Enable Translation", True)
    target_lang = st.selectbox("Language", ["en", "es", "fr", "de", "zh", "ja"])


# Function to generate description
def generate_description(landmark):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{TEXT_MODEL}",
            headers=headers,
            json={
                "inputs": f"Provide detailed historical, architectural, and visitor information about {landmark}.",
                "parameters": {
                    "max_length": 500,
                    "num_beams": 4,
                    "no_repeat_ngram_size": 2
                }
            },
            timeout=100
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and 'summary_text' in result[0]:
                return result[0]['summary_text']
        
        return f"Unable to generate description for {landmark}. Please try again later."
    except Exception as e:
        return f"Unable to generate description for {landmark}. Please check the landmark name or try again later."

# Function to translate text
def translate_text(text, target_lang):
    if target_lang == "en":
        return text
        
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{TRANSLATION_MODEL.format(tgt=target_lang)}",
            headers=headers,
            json={"inputs": text},
            params={"wait_for_model": True},
            timeout=40
        )
        return response.json()[0]['translation_text'] if response.status_code == 200 else text
    except Exception as e:
        return f"Translation failed: {e}"

# File Upload Section
uploaded_file = st.file_uploader("Upload an image of a landmark", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing image..."):
        landmark_name = analyze_image(image)
    
    st.subheader(f"🌍 Identified Landmark: {landmark_name}")
    
    if landmark_name != "Landmark Not Recognized":
        with st.spinner("📝 Generating description..."):
            description = generate_description(landmark_name)
        
        st.write(description)

        if translate_option and target_lang != "en":
            with st.spinner(f"🌐 Translating to {target_lang.upper()}..."):
                translated_desc = translate_text(description, target_lang)
            st.subheader(f"📖 Translated Description ({target_lang.upper()})")
            st.write(translated_desc)
    else:
        st.warning("Sorry, the landmark couldn't be identified. Please try a different image.")

# Custom CSS for better appearance
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #f7f7f7;
    }
    .sidebar .sidebar-header {
        font-size: 1.2em;
        color: #4CAF50;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)
