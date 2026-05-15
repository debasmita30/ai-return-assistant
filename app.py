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
import requests
from io import BytesIO

# --- Constants ---
DEFAULT_IMAGE_URL = "https://assets.myntassets.com/w_412,q_30,dpr_3,fl_progressive,f_webp/assets/images/29261846/2024/4/30/7d624718-2668-4e42-a1d3-b7beb0dad5d41714465033384Dresses1.jpg"

# --- NLTK Setup ---
lemmatizer = WordNetLemmatizer()
stop_words = set(ENGLISH_STOP_WORDS)

def ensure_nltk_data():
    """Try to download NLTK data; app still works if download fails."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            st.warning("Could not download NLTK 'punkt'. Using basic tokenizer.")
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        try:
            nltk.download('wordnet', quiet=True)
        except:
            st.warning("Could not download NLTK 'wordnet'. Lemmatization may be limited.")

ensure_nltk_data()

def preprocess_text(text):
    """Clean and preprocess text safely; fallback if NLTK unavailable."""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower().strip()
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        tokens = text.split()
    try:
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    except:
        pass
    return " ".join(tokens)

# --- Load models and sentiment analyzer ---
@st.cache_resource
def load_models():
    text_model, vectorizer, image_model = None, None, None
    try:
        text_model = joblib.load('models/text_classifier_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        image_model = keras.models.load_model('models/image_classifier_model.keras')
        st.success("AI models loaded successfully.")
    except (FileNotFoundError, IOError):
        st.warning("Could not load AI models. Running in demo mode.")
    analyzer = SentimentIntensityAnalyzer()
    return text_model, vectorizer, image_model, analyzer

# --- Load data ---
@st.cache_data
def load_data():
    catalog_df, reviews_df = pd.DataFrame(), pd.DataFrame()
    try:
        catalog_df = pd.read_csv('data/processed/catalog.csv')
        reviews_df = pd.read_csv('data/raw/Womens Clothing E-Commerce Reviews.csv')
        reviews_df['Clothing ID'] = reviews_df['Clothing ID'].astype(str)
    except FileNotFoundError:
        st.error("Required data files not found.")
    return catalog_df, reviews_df

# --- Helper Functions ---
def create_fallback_image():
    img = Image.new('RGB', (160, 160), color=(39, 51, 70))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    d.text((10, 10), "Image not available", fill=(255, 255, 255), font=font)
    return img

def predict_image(image_model, image_input):
    """Predict whether image shows a defective or normal product."""
    if image_model is None:
        return "Defective"
    try:
        if isinstance(image_input, str) and image_input.startswith('http'):
            response = requests.get(image_input, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_input)
    except (requests.exceptions.RequestException, UnidentifiedImageError, IOError):
        st.warning("Could not load image, using fallback.")
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

def predict_image_from_pil(image_model, pil_img):
    """Predict from an already-loaded PIL image."""
    if image_model is None:
        # Demo mode: simulate random result based on image brightness
        arr = np.array(pil_img.convert('RGB'))
        brightness = arr.mean()
        return "Normal" if brightness > 100 else "Defective"
    try:
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        pil_img = pil_img.resize((160, 160))
        img_array = np.array(pil_img)
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        img_array = np.expand_dims(img_array, axis=0)
        pred = image_model.predict(img_array)
        return "Normal" if pred[0][0] < 0.5 else "Defective"
    except Exception as e:
        st.warning(f"Image prediction error: {e}")
        return "Unknown"

def calculate_risk_score(severity, image_prediction, complaint_mismatch):
    score = severity * 10
    if image_prediction == "Defective":
        score += 40
    if complaint_mismatch:
        score += 20
    return min(score, 100)

def get_risk_label(risk_score):
    if risk_score >= 70:
        return "high", f"High Risk Return ⚠️ ({risk_score}%)"
    elif risk_score >= 40:
        return "medium", f"Moderate Risk Return ⚠️ ({risk_score}%)"
    else:
        return "low", f"Low Risk Return ✅ ({risk_score}%)"

def display_image_grid(images_with_labels, image_model):
    """Display a grid of uploaded dress images with individual predictions."""
    st.subheader("📸 Uploaded Dress Images — Individual Analysis")
    cols_per_row = 3
    results = []

    for i in range(0, len(images_with_labels), cols_per_row):
        batch = images_with_labels[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, (pil_img, label) in enumerate(batch):
            with cols[j]:
                st.image(pil_img, caption=label, use_container_width=True)
                prediction = predict_image_from_pil(image_model, pil_img)
                if prediction == "Defective":
                    st.error(f"🔴 {prediction}")
                elif prediction == "Normal":
                    st.success(f"🟢 {prediction}")
                else:
                    st.warning(f"🟡 {prediction}")
                results.append((label, prediction))

    return results

# --- Load resources ---
text_model, vectorizer, image_model, analyzer = load_models()
catalog_df, reviews_df = load_data()

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Return Assistant", page_icon="🤖")

# --- Custom CSS for enhanced styling ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 50%, #0f1117 100%);
    }
    .upload-zone {
        border: 2px dashed #4a9eff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background: rgba(74, 158, 255, 0.05);
        margin: 10px 0;
    }
    .batch-summary {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4a9eff;
    }
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Powered Return Assistant")

# --- Tabs for Single Return vs Batch Upload ---
tab1, tab2 = st.tabs(["📦 Single Return Analysis", "📂 Batch Dress Image Upload"])

# ============================================================
# TAB 1 — Original Single Return Analysis (unchanged logic)
# ============================================================
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.header("Return Request Details")
        product_id_input = st.text_input("Enter Product ID", value="1078", key="single_pid")
        complaint = st.radio(
            "Select Customer Complaint",
            ["Wrong Colour", "Size Issue", "Defective", "Not as Described", "Other"],
            key="single_complaint"
        )
        if complaint == "Other":
            complaint = st.text_input("Describe the issue:", key="single_other")
        severity = st.slider("How severe is the issue?", 1, 10, 5, key="single_severity")

        st.markdown("#### 🖼️ Product Image")
        image_source = st.radio(
            "Image Source",
            ["Use default/demo image", "Upload image file", "Enter image URL"],
            key="single_img_source"
        )

        uploaded_image = None
        custom_url = None

        if image_source == "Upload image file":
            uploaded_image = st.file_uploader(
                "Upload Product Image",
                type=["jpg", "png", "jpeg", "webp"],
                key="single_upload",
                help="Upload a dress or product image for defect analysis"
            )
            if uploaded_image:
                preview = Image.open(uploaded_image)
                st.image(preview, caption="Uploaded Image Preview", width=160)
                uploaded_image.seek(0)  # reset for later use

        elif image_source == "Enter image URL":
            custom_url = st.text_input(
                "Paste image URL",
                placeholder="https://example.com/product.jpg",
                key="single_url"
            )
            if custom_url:
                try:
                    r = requests.get(custom_url, timeout=5)
                    preview = Image.open(BytesIO(r.content))
                    st.image(preview, caption="URL Image Preview", width=160)
                except Exception:
                    st.warning("Could not load preview from URL.")

        approve_checkbox = st.checkbox("Manually Approve Return?", key="single_approve")
        analyze_button = st.button("🔍 Analyze Return", type="primary", key="single_analyze")

    with col2:
        st.header("Analysis & Recommendation")

        if not analyze_button:
            st.subheader("Analysis Preview")
            st.image(DEFAULT_IMAGE_URL, caption="Awaiting Analysis", width=200)
            st.info("Fill out details and click 'Analyze Return'.")

        if analyze_button:
            # Determine image to process
            use_default = (image_source == "Use default/demo image")
            has_upload = (image_source == "Upload image file" and uploaded_image is not None)
            has_url = (image_source == "Enter image URL" and custom_url)

            if product_id_input and complaint and (use_default or has_upload or has_url):
                with st.spinner("Analyzing return request..."):
                    if use_default:
                        image_to_process = DEFAULT_IMAGE_URL
                        caption = "Default Demo Image"
                    elif has_upload:
                        image_to_process = uploaded_image
                        caption = f"Uploaded: {uploaded_image.name}"
                    else:
                        image_to_process = custom_url
                        caption = "Customer-Provided URL Image"

                    # --- Text Analysis ---
                    if vectorizer and text_model:
                        processed_complaint = preprocess_text(complaint)
                        text_vector = vectorizer.transform([processed_complaint])
                        predicted_class = text_model.predict(text_vector)[0]
                    else:
                        predicted_class = "Tops"

                    # --- Image Analysis ---
                    image_prediction = predict_image(image_model, image_to_process)

                    # --- Product Info ---
                    product_name, expected_class = "Product not found", "N/A"
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
                    risk_level, risk_label = get_risk_label(risk_score)

                    st.subheader("Return Risk Score")
                    st.progress(risk_score / 100)
                    if risk_level == "high":
                        st.error(risk_label)
                    elif risk_level == "medium":
                        st.warning(risk_label)
                    else:
                        st.success(risk_label)

                    st.subheader("Final Recommendation")
                    if approve_checkbox:
                        st.success("✅ Return Manually Approved by User")
                    else:
                        if risk_level == "high":
                            st.error("❌ Reject Return or Investigate Further")
                        elif risk_level == "medium":
                            st.warning("⚠️ Manual Review Required")
                        else:
                            st.success("✅ Approve Return")

                    # --- Global Review Analysis Section ---
                    st.markdown("---")
                    st.subheader("Global Review Analysis (All Products)")

                    if not reviews_df.empty:
                        sentiment_counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
                        negative_reviews_text = []

                        for _, row in reviews_df.iterrows():
                            review_text = str(row.get('Review Text', ''))
                            if review_text.strip():
                                processed_review = preprocess_text(review_text)
                                score = analyzer.polarity_scores(processed_review)['compound']
                                if score <= -0.05:
                                    sentiment_counts["Negative"] += 1
                                    negative_reviews_text.append(review_text)
                                elif score >= 0.05:
                                    sentiment_counts["Positive"] += 1
                                else:
                                    sentiment_counts["Neutral"] += 1

                        if negative_reviews_text:
                            with st.expander("Show Recent Negative Reviews (Global)"):
                                for r in negative_reviews_text[:5]:
                                    st.info(f'"{r}"')
                        else:
                            st.write("No distinct negative reviews found.")

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
                    else:
                        st.info("No reviews data available for global analysis.")
            else:
                st.error("Please provide all inputs: Product ID, Complaint, and an Image.")

# ============================================================
# TAB 2 — Batch Dress Image Upload & Analysis
# ============================================================
with tab2:
    st.header("📂 Batch Dress Image Upload & Analysis")
    st.markdown(
        "Upload **multiple dress images** at once to get defect predictions for each. "
        "This is ideal for quality control checks or bulk return processing."
    )

    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    batch_files = st.file_uploader(
        "📤 Drop dress images here (JPG, PNG, JPEG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_upload",
        help="You can select multiple files at once using Ctrl+Click or Cmd+Click"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if batch_files:
        st.markdown(f"**{len(batch_files)} image(s) uploaded.** Ready for analysis.")

        # Optional: add complaint context for batch
        with st.expander("⚙️ Optional: Set Complaint Context for Batch"):
            batch_complaint = st.selectbox(
                "Complaint type for all images",
                ["Defective", "Wrong Colour", "Size Issue", "Not as Described", "Other"],
                key="batch_complaint"
            )
            batch_severity = st.slider("Severity (applies to all)", 1, 10, 5, key="batch_severity")

        run_batch = st.button("🔍 Analyze All Uploaded Images", type="primary", key="batch_analyze")

        if run_batch:
            with st.spinner(f"Analyzing {len(batch_files)} image(s)..."):
                images_with_labels = []
                for f in batch_files:
                    try:
                        pil_img = Image.open(f)
                        images_with_labels.append((pil_img, f.name))
                        f.seek(0)
                    except Exception as e:
                        st.warning(f"Could not open {f.name}: {e}")

                if images_with_labels:
                    results = display_image_grid(images_with_labels, image_model)

                    # --- Summary Statistics ---
                    st.markdown("---")
                    st.subheader("📊 Batch Analysis Summary")

                    normal_count = sum(1 for _, r in results if r == "Normal")
                    defective_count = sum(1 for _, r in results if r == "Defective")
                    unknown_count = sum(1 for _, r in results if r == "Unknown")
                    total = len(results)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Images", total)
                    m2.metric("✅ Normal", normal_count)
                    m3.metric("🔴 Defective", defective_count)
                    m4.metric("🟡 Unknown", unknown_count)

                    if total > 0:
                        defect_rate = round((defective_count / total) * 100, 1)
                        st.markdown(
                            f'<div class="batch-summary">'
                            f'<b>Defect Rate: {defect_rate}%</b> — '
                            f'{defective_count} out of {total} images flagged as defective.'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        # Risk assessment for batch
                        batch_image_pred = "Defective" if defective_count > 0 else "Normal"
                        batch_risk = calculate_risk_score(batch_severity, batch_image_pred, False)
                        _, batch_risk_label = get_risk_label(batch_risk)

                        st.progress(batch_risk / 100)
                        if batch_risk >= 70:
                            st.error(f"Batch Risk: {batch_risk_label}")
                        elif batch_risk >= 40:
                            st.warning(f"Batch Risk: {batch_risk_label}")
                        else:
                            st.success(f"Batch Risk: {batch_risk_label}")

                    # --- Altair bar chart for batch results ---
                    df_batch = pd.DataFrame({
                        "Assessment": ["Normal", "Defective", "Unknown"],
                        "Count": [normal_count, defective_count, unknown_count]
                    })
                    batch_color = alt.Scale(
                        domain=["Normal", "Defective", "Unknown"],
                        range=["#2ca0ac", "#d62728", "#ffcc00"]
                    )
                    batch_chart = alt.Chart(df_batch).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                        x=alt.X("Assessment"),
                        y=alt.Y("Count", axis=alt.Axis(tickMinStep=1)),
                        color=alt.Color("Assessment", scale=batch_color, legend=None),
                        tooltip=["Assessment", "Count"]
                    ).properties(width=400, height=280, title="Image Assessment Distribution")
                    st.altair_chart(batch_chart, use_container_width=True)

                    # --- Detailed results table ---
                    with st.expander("📋 View Detailed Results Table"):
                        df_results = pd.DataFrame(results, columns=["Filename", "Assessment"])
                        df_results.index += 1
                        st.dataframe(df_results, use_container_width=True)
    else:
        st.info("👆 Upload one or more dress images above to begin batch analysis.")
        st.markdown("""
        **How it works:**
        1. Click the upload area or drag & drop dress images
        2. Optionally set a complaint type and severity
        3. Click **Analyze All Uploaded Images**
        4. View per-image predictions and a batch summary
        """)
