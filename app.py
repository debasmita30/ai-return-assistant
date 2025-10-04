import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import altair as alt

# --- NLTK Data Check ---
# Use the modern and correct exception (LookupError) for handling missing NLTK data.
# This makes the app more robust on new deployments.
def download_nltk_data():
    """Checks for and downloads required NLTK data, showing progress in Streamlit."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        st.info("One-time download of NLTK data (punkt, wordnet) in progress...")
        nltk.download('punkt')
        nltk.download('wordnet')
        st.success("NLTK data is ready.")

download_nltk_data()

# --- Load Models and Data ---
@st.cache_resource
def load_models():
    """
    Load and cache ML models. Falls back to a demo mode if files are not found,
    preventing the app from crashing.
    """
    analyzer = SentimentIntensityAnalyzer()
    try:
        text_model = joblib.load('models/text_classifier_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        image_model = keras.models.load_model('models/image_classifier_model.keras')
        st.success("AI models loaded successfully.")
    except FileNotFoundError:
        st.warning("Model files not found. The app will run in demo mode with placeholder predictions.")
        text_model, vectorizer, image_model = None, None, None
    return text_model, vectorizer, image_model, analyzer

@st.cache_data
def load_data():
    """
    Load and cache datasets. Handles FileNotFoundError to prevent crashes if
    data is missing.
    """
    try:
        catalog_df = pd.read_csv('data/processed/catalog.csv')
        reviews_df = pd.read_csv('data/raw/Womens Clothing E-Commerce Reviews.csv')
        # Standardize 'Clothing ID' to string for reliable matching.
        reviews_df['Clothing ID'] = reviews_df['Clothing ID'].astype(str)
        return catalog_df, reviews_df
    except FileNotFoundError:
        st.error("Crucial data files are missing. Please ensure data is in the correct directory.")
        return pd.DataFrame(), pd.DataFrame()

# --- Helper Functions ---
lemmatizer = WordNetLemmatizer()
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess_text(text):
    """Clean and tokenize text for NLP analysis."""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower().strip()
    tokens = nltk.word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(lemmatized)

def predict_image(image_model, uploaded_image):
    """Resize, process, and predict the class of an uploaded image."""
    if image_model is None:
        return "Defective"  # Demo mode placeholder
    img = Image.open(uploaded_image).resize((160, 160))
    img_array = np.array(img)
    if img_array.shape[2] == 4:  # Remove alpha channel for consistency
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    prediction = image_model.predict(img_array)
    return "Normal" if prediction[0][0] < 0.5 else "Defective"

def calculate_risk_score(severity, image_prediction, complaint_mismatch):
    """Calculate a return risk score based on multiple inputs."""
    score = severity * 10
    if image_prediction == "Defective": score += 40
    if complaint_mismatch: score += 20
    return min(score, 100)

# --- Main App Logic ---
text_model, vectorizer, image_model, analyzer = load_models()
catalog_df, reviews_df = load_data()

st.set_page_config(layout="wide", page_title="AI Return Assistant")
st.title("🤖 AI-Powered Return Assistant")

# --- UI Layout ---
col1, col2 = st.columns(2)

with col1:
    st.header("Return Request Details")
    product_id_input = st.text_input("Enter Product ID", help="e.g., 1078, 862, 999")
    complaint = st.radio("Select Customer Complaint", ["Wrong Colour", "Size Issue", "Defective", "Not as Described", "Other"])
    if complaint == "Other":
        complaint = st.text_input("Please describe the issue in detail:")
    severity = st.slider("How severe is the issue?", 1, 10, 5)
    uploaded_image = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])
    approve_checkbox = st.checkbox("Manually Approve This Return?")
    analyze_button = st.button("Analyze Return", type="primary")

with col2:
    st.header("Analysis & Recommendation")
    if analyze_button:
        if product_id_input and complaint and uploaded_image:
            with st.spinner("Analyzing return request..."):
                # --- 1. AI Analysis ---
                st.subheader("AI Analysis Results")
                
                predicted_class = "Tops" # Default for demo mode
                if text_model and vectorizer:
                    processed_complaint = preprocess_text(complaint)
                    text_vector = vectorizer.transform([processed_complaint])
                    predicted_class = text_model.predict(text_vector)[0]
                
                image_prediction = predict_image(image_model, uploaded_image)
                
                product_id_str = str(product_id_input).strip()
                product_details = catalog_df[catalog_df['product_id'].astype(str) == product_id_str]
                
                product_name = product_details.iloc[0]['product_name'] if not product_details.empty else "Product Not Found"
                expected_class = product_details.iloc[0]['article_type'] if not product_details.empty else "N/A"

                st.image(uploaded_image, caption="Uploaded Customer Image", width=200)
                st.write(f"**Product Name:** {product_name}")
                st.write(f"**Predicted Complaint Category:** `{predicted_class}`")
                st.write(f"**Image Assessment:** `{image_prediction}`")

                # --- 2. Risk Scoring ---
                complaint_mismatch = (predicted_class != expected_class) and (expected_class != "N/A")
                risk_score = calculate_risk_score(severity, image_prediction, complaint_mismatch)

                st.subheader("Return Risk Score")
                st.progress(risk_score / 100)
                if risk_score >= 70: st.error(f"High Risk Return ⚠️ ({risk_score}%)")
                elif risk_score >= 40: st.warning(f"Moderate Risk Return ({risk_score}%)")
                else: st.success(f"Low Risk Return ✅ ({risk_score}%)")

                # --- 3. Final Recommendation ---
                st.subheader("Final Recommendation")
                if approve_checkbox:
                    st.success("✅ Return Manually Approved by User")
                else:
                    if risk_score >= 70: st.error("❌ Reject Return or Investigate Further")
                    elif risk_score >= 40: st.warning("⚠️ Manual Review Recommended")
                    else: st.success("✅ Approve Return")
                
                st.markdown("---")

                # --- 4. Historical Review Analysis (Global) ---
                st.subheader("Global Review Analysis (All Products)")
                if not reviews_df.empty:
                    sentiment_counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
                    negative_reviews = []
                    
                    for _, row in reviews_df.iterrows():
                        if pd.notna(row['Review Text']):
                            score = analyzer.polarity_scores(row['Review Text'])['compound']
                            if score <= -0.05: sentiment_counts["Negative"] += 1
                            elif score >= 0.05: sentiment_counts["Positive"] += 1
                            else: sentiment_counts["Neutral"] += 1
                            
                            if score <= -0.05 or row['Rating'] <= 2:
                                negative_reviews.append(row['Review Text'])
                    
                    with st.expander("Show Recent Negative Reviews (Global)"):
                        if negative_reviews:
                            for r in negative_reviews[-5:]: st.info(f'"{r}"')
                        else:
                            st.write("No distinct negative reviews found across all products.")
                    
                    # --- Sentiment Chart ---
                    df_sent = pd.DataFrame(list(sentiment_counts.items()), columns=["Sentiment", "Count"])
                    color_scale = alt.Scale(domain=["Positive", "Neutral", "Negative"], range=["#2ca02c", "#ffcc00", "#d62728"])
                    chart = alt.Chart(df_sent).mark_bar().encode(
                        x=alt.X("Sentiment", sort=["Negative", "Neutral", "Positive"]),
                        y=alt.Y("Count", axis=alt.Axis(tickMinStep=1)),
                        color=alt.Color("Sentiment", scale=color_scale, legend=None),
                        tooltip=['Sentiment', 'Count']
                    ).properties(title="Global Review Sentiment Distribution")
                    st.altair_chart(chart, use_container_width=True)
        else:
            st.error("Please provide all inputs: Product ID, Complaint, and Image.")

