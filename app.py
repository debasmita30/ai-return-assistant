import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # Import VADER

# --- Caching Models and Data ---
@st.cache_resource
def load_models():
    text_model = joblib.load('models/text_classifier_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    image_model = keras.models.load_model('models/image_classifier_model.keras')
    # ADDED: Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    return text_model, vectorizer, image_model, analyzer

@st.cache_data
def load_data():
    catalog_df = pd.read_csv('data/processed/catalog.csv')
    reviews_df = pd.read_csv('data/raw/Womens Clothing E-Commerce Reviews.csv')
    return catalog_df, reviews_df

# --- Helper Functions ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text), re.I|re.A).lower().strip()
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return " ".join(lemmatized_tokens)

def predict_image(image_model, uploaded_image):
    img = Image.open(uploaded_image).resize((160, 160))
    img_array = np.array(img)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    prediction = image_model.predict(img_array)
    return "Normal" if prediction[0][0] < 0.5 else "Defective"

# --- Main App ---
# UPDATED: Load the sentiment analyzer as well
text_model, vectorizer, image_model, analyzer = load_models()
catalog_df, reviews_df = load_data() 

st.set_page_config(layout="wide")
st.title("🤖 AI-Powered Return Assistant")

# ... (The rest of your UI code remains the same until the review display section)
col1, col2 = st.columns(2)

with col1:
    st.header("Return Request Details")
    product_id_input = st.text_input("Enter Product ID (e.g., 1078, 862)")
    complaint_text = st.text_area("Enter Customer Complaint")
    uploaded_image = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])
    analyze_button = st.button("Analyze Return")

with col2:
    st.header("Analysis & Recommendation")
    if analyze_button:
        if product_id_input and complaint_text and uploaded_image:
            with st.spinner('Analyzing...'):
                # (All the analysis code remains the same here)
                st.subheader("Inputs Received")
                st.image(uploaded_image, caption="Uploaded Customer Image", width=250)
                
                cleaned_text = preprocess_text(complaint_text)
                text_vector = vectorizer.transform([cleaned_text])
                predicted_class = text_model.predict(text_vector)[0]
                image_prediction = predict_image(image_model, uploaded_image)

                try:
                    product_id = int(product_id_input)
                    product_details = catalog_df[catalog_df['product_id'] == product_id].iloc[0]
                    product_name = product_details['product_name']
                    expected_class = product_details['article_type']
                except (ValueError, IndexError):
                    product_name = "Product not found"
                    expected_class = "N/A"

                st.subheader("AI Analysis")
                st.metric("Predicted Complaint Category", predicted_class)
                st.metric("Image Assessment", image_prediction)
                st.metric("Product Name", product_name)
                
                st.subheader("Final Recommendation")
                if image_prediction == "Defective":
                    st.success("✅ Approve Return: Visual evidence of defect detected.")
                elif predicted_class != expected_class and expected_class != "N/A":
                     st.warning("⚠️ Review Required: Complaint seems mismatched with the product type.")
                else:
                    st.error("❌ Reject or Investigate: No clear visual defect and complaint is consistent with product.")

                # --- UPDATED SECTION TO DISPLAY REVIEWS ---
                st.subheader("Recent Negative Reviews for this Product")
                product_reviews = reviews_df[reviews_df['Clothing ID'] == product_id].copy()
                truly_negative_reviews = []
                for index, row in product_reviews.iterrows():
                    # Check for low rating AND negative text sentiment
                    if row['Rating'] <= 2:
                        if pd.notna(row['Review Text']):
                            sentiment_score = analyzer.polarity_scores(row['Review Text'])
                            # The 'compound' score is a good overall measure. < 0 is negative.
                            if sentiment_score['compound'] < 0:
                                truly_negative_reviews.append(row['Review Text'])
                
                if truly_negative_reviews:
                    # Display the top 3 truly negative reviews
                    for review_text in truly_negative_reviews[:3]:
                        st.info(f'"{review_text}"')
                else:
                    st.write("No truly negative reviews found in the dataset for this product ID.")
        else:
            st.error("Please provide all three inputs: Product ID, Complaint, and Image.")