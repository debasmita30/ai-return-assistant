import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import altair as alt
import os
import requests
from io import BytesIO

# --- Constants ---
DEFAULT_IMAGE_URL = "https://assets.myntassets.com/w_412,q_30,dpr_3,fl_progressive,f_webp/assets/images/29261846/2024/4/30/7d624718-2668-4e42-a1d3-b7beb0dad5d41714465033384Dresses1.jpg"

# --- NLTK Data Setup ---
def ensure_nltk_data():
    """Checks for NLTK data packages and downloads them if missing."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        with st.spinner("One-time setup: Downloading necessary NLTK data..."):
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
        st.success("NLTK data is ready.")

ensure_nltk_data()

# --- Load models/data ---
@st.cache_resource
def load_models():
    """Load and cache the machine learning models and sentiment analyzer."""
    text_model, vectorizer, image_model = None, None, None
    try:
        text_model = joblib.load('models/text_classifier_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        image_model = keras.models.load_model('models/image_classifier_model.keras')
        st.success("AI models loaded successfully.")
    except (FileNotFoundError, IOError):
        st.warning("Could not load AI models. Running in demonstration mode.")
        st.info("To enable predictions, add model files to a 'models' directory.")
    analyzer = SentimentIntensityAnalyzer()
    return text_model, vectorizer, image_model, analyzer

@st.cache_data
def load_data():
    """Load and cache the product catalog and reviews datasets."""
    catalog_df, reviews_df = pd.DataFrame(), pd.DataFrame()
    try:
        catalog_df = pd.read_csv('data/processed/catalog.csv')
        reviews_df = pd.read_csv('data/raw/Womens Clothing E-Commerce Reviews.csv')
        reviews_df['Clothing ID'] = reviews_df['Clothing ID'].astype(str)
    except FileNotFoundError:
        st.error("Data files not found. App requires 'catalog.csv' and 'Womens Clothing E-Commerce Reviews.csv'.")
    return catalog_df, reviews_df

# --- Helper Functions ---
def create_fallback_image():
    """Generates a fallback PIL image if a URL fails."""
    img = Image.new('RGB', (160, 160), color=(39, 51, 70))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    d.text((10, 10), "Image not available", fill=(255, 255, 255), font=font)
    return img

lemmatizer = WordNetLemmatizer()
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess_text(text):
    """Clean and preprocess text for NLP model."""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower().strip()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

def predict_image(image_model, image_input):
    """Predicts on a file uploader object or URL with error handling."""
    if image_model is None: return "Defective"
    
    try:
        if isinstance(image_input, str) and image_input.startswith('http'):
            response = requests.get(image_input, timeout=5)
            response.raise_for_status()  # Raise an exception for bad status codes
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_input)
    except (requests.exceptions.RequestException, UnidentifiedImageError, IOError):
        st.warning("Could not load image from URL. Using a fallback for analysis.")
        img = create_fallback_image()
    
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((160, 160))
    img_array = np.array(img)
    
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)

    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    img_array = np.expand_dims(img_array, axis=0)
    pred = image_model.predict(img_array)
    return "Normal" if pred[0][0] < 0.5 else "Defective"

def calculate_risk_score(severity, image_prediction, complaint_mismatch):
    """Calculate a risk score for the return request."""
    score = severity * 10
    if image_prediction == "Defective": score += 40
    if complaint_mismatch: score += 20
    return min(score, 100)

# --- Load Resources ---
text_model, vectorizer, image_model, analyzer = load_models()
catalog_df, reviews_df = load_data()

# --- UI Layout ---
st.set_page_config(layout="wide", page_title="AI Return Assistant")
st.title("🤖 AI-Powered Return Assistant")

col1, col2 = st.columns(2)

with col1:
    st.header("Return Request Details")
    product_id_input = st.text_input("Enter Product ID", value="1078", help="E.g., 1078, 862, 999")
    complaint = st.radio("Select Customer Complaint", ["Wrong Colour", "Size Issue", "Defective", "Not as Described", "Other"])
    if complaint == "Other":
        complaint = st.text_input("Please describe the issue:")
    severity = st.slider("How severe is the issue?", 1, 10, 5)
    
    use_default_image = st.checkbox("Use default/demo image", value=True)
    uploaded_image = None
    if not use_default_image:
        uploaded_image = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])

    approve_checkbox = st.checkbox("Manually Approve Return?")
    analyze_button = st.button("Analyze Return")

with col2:
    st.header("Analysis & Recommendation")

    # Display a placeholder before analysis is run
    if not analyze_button:
        st.subheader("Analysis Preview")
        st.image(DEFAULT_IMAGE_URL, caption="Awaiting Analysis", width=200)
        st.info("Fill out the details on the left and click 'Analyze Return' to see the results.")

    if analyze_button:
        if product_id_input and complaint and (uploaded_image or use_default_image):
            with st.spinner("Analyzing..."):
                # --- Select image to process ---
                image_to_process = DEFAULT_IMAGE_URL if use_default_image else uploaded_image
                caption = "Default Demo Image" if use_default_image else "Uploaded Customer Image"

                # --- AI Analysis Section ---
                st.subheader("Analysis Results")
                
                if vectorizer and text_model:
                    processed_complaint = preprocess_text(complaint)
                    text_vector = vectorizer.transform([processed_complaint])
                    predicted_class = text_model.predict(text_vector)[0]
                else:
                    predicted_class = "Tops"
                
                image_prediction = predict_image(image_model, image_to_process)

                product_name = "Product not found"
                expected_class = "N/A"
                product_id_str = str(product_id_input).strip()
                
                if not catalog_df.empty:
                    product_details = catalog_df[catalog_df['product_id'].astype(str) == product_id_str]
                    if not product_details.empty:
                        product_name = product_details.iloc[0]['product_name']
                        expected_class = product_details.iloc[0]['article_type']

                st.image(image_to_process, caption=caption, width=200)
                st.write(f"**Product Name:** {product_name}")
                st.write(f"**Predicted Complaint Category:** {predicted_class}")
                st.write(f"**Image Assessment:** {image_prediction}")

                complaint_mismatch = (predicted_class != expected_class) and (expected_class != "N/A")
                risk_score = calculate_risk_score(severity, image_prediction, complaint_mismatch)

                st.subheader("Return Risk Score")
                st.progress(risk_score / 100)
                if risk_score >= 70:
                    st.error(f"High Risk Return ⚠️ ({risk_score}%)")
                elif risk_score >= 40:
                    st.warning(f"Moderate Risk Return ⚠️ ({risk_score}%)")
                else:
                    st.success(f"Low Risk Return ✅ ({risk_score}%)")

                st.subheader("Final Recommendation")
                if approve_checkbox:
                    st.success("✅ Return Manually Approved by User")
                else:
                    if risk_score >= 70: st.error("❌ Reject Return or Investigate Further")
                    elif risk_score >= 40: st.warning("⚠️ Manual Review Required")
                    else: st.success("✅ Approve Return")

                # --- Global Review Analysis Section ---
                st.markdown("---")
                st.subheader("Global Review Analysis (All Products)")

                if not reviews_df.empty:
                    sentiment_counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
                    negative_reviews_text = []

                    for _, row in reviews_df.iterrows():
                        if pd.notna(row['Review Text']):
                            score = analyzer.polarity_scores(row['Review Text'])['compound']
                            if score <= -0.05:
                                sentiment_counts["Negative"] += 1
                                negative_reviews_text.append(row['Review Text'])
                            elif score >= 0.05:
                                sentiment_counts["Positive"] += 1
                            else:
                                sentiment_counts["Neutral"] += 1
                    
                    if negative_reviews_text:
                        with st.expander("Show Recent Negative Reviews (Global)"):
                            for r in negative_reviews_text[:5]: st.info(f'"{r}"')
                    else:
                        st.write("No distinct negative reviews found.")

                    st.markdown("##### Global Sentiment Distribution")
                    df_sent = pd.DataFrame(list(sentiment_counts.items()), columns=["Sentiment", "Count"])
                    
                    color_scale = alt.Scale(
                        domain=["Positive", "Neutral", "Negative"],
                        range=["#2ca0ac", "#ffcc00", "#d62728"]
                    )
                    
                    chart = alt.Chart(df_sent).mark_bar().encode(
                        x=alt.X("Sentiment", sort=["Negative", "Neutral", "Positive"]),
                        y=alt.Y("Count", axis=alt.Axis(tickMinStep=1)),
                        color=alt.Color("Sentiment", scale=color_scale, legend=None),
                        tooltip=['Sentiment', 'Count']
                    ).properties(width=400, height=300)
                    
                    st.altair_chart(chart, use_container_width=True)
        elif analyze_button:
            st.error("Please provide all inputs: Product ID, Complaint, and an Image (or use the default).")

