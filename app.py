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
import time

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_IMAGE_URL = "https://assets.myntassets.com/w_412,q_30,dpr_3,fl_progressive,f_webp/assets/images/29261846/2024/4/30/7d624718-2668-4e42-a1d3-b7beb0dad5d41714465033384Dresses1.jpg"

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="ReturnIQ — AI Return Intelligence",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# ─── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0b0e !important;
    color: #e8e6e1 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: #0f1014 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Typography ── */
h1, h2, h3, h4 { font-family: 'DM Sans', sans-serif !important; letter-spacing: -0.02em; }

/* ── Main content area ── */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ── Brand header ── */
.brand-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.brand-logo {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
}
.brand-title { font-size: 1.5rem; font-weight: 600; color: #fff; letter-spacing: -0.03em; }
.brand-subtitle { font-size: 0.78rem; color: #6b7280; letter-spacing: 0.05em; text-transform: uppercase; font-weight: 400; }
.brand-badge {
    margin-left: auto;
    padding: 4px 12px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    font-size: 0.72rem;
    color: #a5b4fc;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #13141a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, linear-gradient(90deg, #6366f1, #8b5cf6));
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 600;
    color: #fff;
    line-height: 1;
    font-family: 'DM Mono', monospace;
    letter-spacing: -0.02em;
}
.metric-label {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
    font-weight: 500;
}
.metric-delta {
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    margin-top: 4px;
}
.delta-up { color: #34d399; }
.delta-down { color: #f87171; }
.delta-neutral { color: #a1a1aa; }

/* ── Risk score display ── */
.risk-container {
    background: #13141a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.25rem 0;
}
.risk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}
.risk-title {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
}
.risk-score-display {
    font-size: 3.5rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    letter-spacing: -0.04em;
    line-height: 1;
}
.risk-low { color: #34d399; }
.risk-med { color: #fbbf24; }
.risk-high { color: #f87171; }

.risk-bar-track {
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
    overflow: hidden;
    margin: 0.75rem 0;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}

/* ── Verdict badge ── */
.verdict {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 18px;
    border-radius: 10px;
    font-size: 0.85rem;
    font-weight: 500;
    width: 100%;
    justify-content: center;
    margin-top: 0.5rem;
}
.verdict-approve {
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.25);
    color: #34d399;
}
.verdict-review {
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.25);
    color: #fbbf24;
}
.verdict-reject {
    background: rgba(248,113,113,0.1);
    border: 1px solid rgba(248,113,113,0.25);
    color: #f87171;
}
.verdict-manual {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    color: #a5b4fc;
}

/* ── AI tag badges ── */
.ai-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
}
.ai-tag-normal {
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.2);
    color: #34d399;
}
.ai-tag-defective {
    background: rgba(248,113,113,0.1);
    border: 1px solid rgba(248,113,113,0.2);
    color: #f87171;
}
.ai-tag-neutral {
    background: rgba(161,161,170,0.1);
    border: 1px solid rgba(161,161,170,0.2);
    color: #a1a1aa;
}

/* ── Info rows ── */
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.85rem;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #6b7280; font-weight: 400; }
.info-val { color: #e8e6e1; font-weight: 500; font-family: 'DM Mono', monospace; font-size: 0.82rem; }

/* ── Sentiment insight card ── */
.insight-card {
    background: #13141a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 12px;
}
.insight-header {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

/* ── Review quotes ── */
.review-quote {
    background: #0f1014;
    border-left: 2px solid rgba(248,113,113,0.4);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 8px;
    font-size: 0.83rem;
    color: #9ca3af;
    font-style: italic;
    line-height: 1.6;
}

/* ── Severity indicator dots ── */
.sev-dots { display: flex; gap: 4px; margin-top: 4px; }
.sev-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: rgba(255,255,255,0.15);
}
.sev-dot-active { background: #6366f1; }
.sev-dot-high { background: #f87171; }

/* ── Streamlit widget overrides ── */
.stSlider > div > div > div > div { background: #6366f1 !important; }
.stSlider [data-baseweb="slider"] > div:nth-child(3) > div { background: #6366f1 !important; }

div[data-baseweb="select"] > div {
    background: #13141a !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: #e8e6e1 !important;
}

div[data-baseweb="input"] > div {
    background: #13141a !important;
    border-color: rgba(255,255,255,0.1) !important;
}
div[data-baseweb="input"] input {
    color: #e8e6e1 !important;
    background: transparent !important;
}

/* Radio */
div[data-testid="stRadio"] label {
    background: #13141a !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
    color: #9ca3af !important;
    font-size: 0.85rem !important;
    cursor: pointer;
    transition: all 0.2s;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(99,102,241,0.15) !important;
    border-color: rgba(99,102,241,0.35) !important;
    color: #a5b4fc !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.5rem !important;
    letter-spacing: 0.01em !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Checkbox */
div[data-testid="stCheckbox"] label span { color: #9ca3af !important; font-size: 0.85rem !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #13141a !important;
    border: 1px dashed rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* Sidebar labels */
[data-testid="stSidebar"] .stMarkdown p {
    color: #6b7280 !important;
    font-size: 0.8rem !important;
}

/* Altair chart background */
.vega-embed { background: transparent !important; }
.vega-embed canvas { background: transparent !important; }

/* Progress bar override */
div[data-testid="stProgress"] > div > div { background: #6366f1 !important; }

/* Text input labels */
.stTextInput label, .stSelectbox label, .stSlider label,
.stRadio label[data-baseweb="radio"], .stFileUploader label {
    color: #6b7280 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }

/* Columns gap */
[data-testid="column"] { padding: 0 0.75rem !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child { padding-right: 0 !important; }

/* Expander */
div[data-testid="stExpander"] {
    background: #13141a !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
div[data-testid="stExpander"] summary {
    color: #9ca3af !important;
    font-size: 0.83rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── NLTK Setup ───────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(ENGLISH_STOP_WORDS)

def ensure_nltk_data():
    for resource, path in [('punkt', 'tokenizers/punkt'), ('wordnet', 'corpora/wordnet')]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except:
                pass

ensure_nltk_data()

def preprocess_text(text):
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

# ─── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    text_model, vectorizer, image_model = None, None, None
    try:
        text_model = joblib.load('models/text_classifier_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        image_model = keras.models.load_model('models/image_classifier_model.keras')
    except (FileNotFoundError, IOError):
        pass
    analyzer = SentimentIntensityAnalyzer()
    return text_model, vectorizer, image_model, analyzer

@st.cache_data
def load_data():
    catalog_df, reviews_df = pd.DataFrame(), pd.DataFrame()
    try:
        catalog_df = pd.read_csv('data/processed/catalog.csv')
        reviews_df = pd.read_csv('data/raw/Womens Clothing E-Commerce Reviews.csv')
        reviews_df['Clothing ID'] = reviews_df['Clothing ID'].astype(str)
    except FileNotFoundError:
        pass
    return catalog_df, reviews_df

# ─── Helpers ──────────────────────────────────────────────────────────────────
def create_fallback_image():
    img = Image.new('RGB', (160, 160), color=(19, 20, 26))
    d = ImageDraw.Draw(img)
    d.text((10, 70), "No Image", fill=(100, 100, 120))
    return img

def predict_image(image_model, image_input):
    if image_model is None:
        return "Normal"
    try:
        if isinstance(image_input, str) and image_input.startswith('http'):
            response = requests.get(image_input, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_input)
    except Exception:
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
    score = severity * 10
    if image_prediction == "Defective": score += 40
    if complaint_mismatch: score += 20
    return min(score, 100)

def risk_color_class(score):
    if score >= 70: return "risk-high"
    if score >= 40: return "risk-med"
    return "risk-low"

def risk_bar_color(score):
    if score >= 70: return "#f87171"
    if score >= 40: return "#fbbf24"
    return "#34d399"

def severity_dots(val, total=10):
    dots = ""
    for i in range(1, total + 1):
        cls = "sev-dot-active"
        if i > 7: cls = "sev-dot-high"
        active = "sev-dot-active" if i <= val else "sev-dot"
        if i <= val and i > 7:
            active = "sev-dot-high"
        dots += f'<div class="sev-dot {active if i <= val else "sev-dot"}"></div>'
    return dots

# ─── Load resources ───────────────────────────────────────────────────────────
text_model, vectorizer, image_model, analyzer = load_models()
catalog_df, reviews_df = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0 0.5rem 1.5rem;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;">
            <div style="width:32px;height:32px;background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;">🔬</div>
            <div>
                <div style="font-size:0.9rem;font-weight:600;color:#fff;letter-spacing:-0.02em;">ReturnIQ</div>
                <div style="font-size:0.68rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.06em;">v2.0 · AI Engine</div>
            </div>
        </div>
        <div style="font-size:0.72rem;color:#4b5563;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:0.6rem;">Model Status</div>
    </div>
    """, unsafe_allow_html=True)

    # Model status indicators
    models_status = [
        ("Text Classifier", text_model is not None),
        ("TF-IDF Vectorizer", vectorizer is not None),
        ("Image CNN", image_model is not None),
        ("Sentiment VADER", True),
    ]
    for name, ok in models_status:
        dot_color = "#34d399" if ok else "#f87171"
        label = "Online" if ok else "Demo"
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
            <span style="font-size:0.8rem;color:#9ca3af;">{name}</span>
            <span style="display:flex;align-items:center;gap:5px;font-size:0.72rem;color:{dot_color};font-family:'DM Mono',monospace;">
                <span style="width:6px;height:6px;border-radius:50%;background:{dot_color};display:inline-block;"></span>{label}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem;color:#4b5563;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:0.75rem;padding:0 0.5rem;">Data Sources</div>
    """, unsafe_allow_html=True)

    data_status = [
        ("Product Catalog", not catalog_df.empty, len(catalog_df) if not catalog_df.empty else 0),
        ("Reviews Dataset", not reviews_df.empty, len(reviews_df) if not reviews_df.empty else 0),
    ]
    for name, ok, count in data_status:
        dot = "#34d399" if ok else "#ef4444"
        val = f"{count:,} rows" if ok else "Missing"
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;padding:7px 0.5rem;border-bottom:1px solid rgba(255,255,255,0.04);">
            <span style="font-size:0.8rem;color:#9ca3af;">{name}</span>
            <span style="font-size:0.72rem;color:{dot};font-family:'DM Mono',monospace;">{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:0.75rem;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.15);border-radius:10px;font-size:0.78rem;color:#6b7280;line-height:1.6;">
        <span style="color:#a5b4fc;font-weight:500;">Risk scoring</span> combines complaint severity, image defect detection, and category mismatch signals to generate a composite return fraud score.
    </div>
    """, unsafe_allow_html=True)

# ─── Brand Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
    <div class="brand-logo">🔬</div>
    <div>
        <div class="brand-title">ReturnIQ</div>
        <div class="brand-subtitle">AI Return Intelligence Platform</div>
    </div>
    <div class="brand-badge">● Live System</div>
</div>
""", unsafe_allow_html=True)

# ─── Layout ───────────────────────────────────────────────────────────────────
input_col, result_col = st.columns([1, 1.1], gap="large")

# ══════════════════════════════════════════════════════════════════
# LEFT: Input Panel
# ══════════════════════════════════════════════════════════════════
with input_col:
    st.markdown('<div class="section-label">Return Details</div>', unsafe_allow_html=True)

    product_id_input = st.text_input(
        "Product ID",
        value="1078",
        placeholder="e.g. 1078, 2043, ...",
        help="Enter the product SKU or catalog ID"
    )

    complaint = st.radio(
        "Customer Complaint",
        ["Wrong Colour", "Size Issue", "Defective", "Not as Described", "Other"],
        horizontal=False
    )
    if complaint == "Other":
        complaint = st.text_input("Describe the issue:", placeholder="Describe in detail...")

    severity = st.slider("Issue Severity", 1, 10, 5, help="1 = minor inconvenience, 10 = major defect")

    # Visual severity dots
    dots_html = ""
    for i in range(1, 11):
        if i <= severity:
            c = "#f87171" if i > 7 else "#fbbf24" if i > 4 else "#34d399"
            dots_html += f'<div style="width:9px;height:9px;border-radius:50%;background:{c};"></div>'
        else:
            dots_html += '<div style="width:9px;height:9px;border-radius:50%;background:rgba(255,255,255,0.1);"></div>'

    st.markdown(f"""
    <div style="display:flex;gap:5px;margin-top:-0.5rem;margin-bottom:0.75rem;align-items:center;">
        {dots_html}
        <span style="margin-left:8px;font-size:0.75rem;color:#6b7280;font-family:'DM Mono',monospace;">
            {"Critical" if severity > 7 else "Moderate" if severity > 4 else "Low"} · {severity}/10
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1rem;">Product Image</div>', unsafe_allow_html=True)

    use_default_image = st.checkbox("Use default demo image", value=True)
    uploaded_image = None
    if not use_default_image:
        uploaded_image = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])

    approve_checkbox = st.checkbox("⚡ Manual Override — Force Approve")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    analyze_button = st.button("Run AI Analysis →", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# RIGHT: Results Panel
# ══════════════════════════════════════════════════════════════════
with result_col:
    if not analyze_button:
        # Idle state
        st.markdown("""
        <div style="background:#13141a;border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:2.5rem;text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:1rem;opacity:0.4;">🔬</div>
            <div style="font-size:1rem;font-weight:500;color:#4b5563;letter-spacing:-0.01em;">Awaiting Analysis</div>
            <div style="font-size:0.8rem;color:#374151;margin-top:0.5rem;">Configure return details and click <strong style="color:#6366f1;">Run AI Analysis</strong></div>
        </div>
        """, unsafe_allow_html=True)

        # Show a preview image
        img_src = DEFAULT_IMAGE_URL if use_default_image else None
        if img_src:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1,2,1])
            with c2:
                st.image(img_src, use_container_width=True, caption="Preview")

    # ─── ANALYSIS RESULTS ────────────────────────────────────────
    if analyze_button:
        if product_id_input and complaint and (uploaded_image or use_default_image):
            with st.spinner("Running multi-modal AI analysis..."):
                time.sleep(0.3)  # brief pause for UX

                image_to_process = DEFAULT_IMAGE_URL if use_default_image else uploaded_image
                caption = "Demo Image" if use_default_image else "Customer Upload"

                # Text analysis
                if vectorizer and text_model:
                    processed_complaint = preprocess_text(complaint)
                    text_vector = vectorizer.transform([processed_complaint])
                    predicted_class = text_model.predict(text_vector)[0]
                else:
                    predicted_class = "Tops"

                # Image analysis
                image_prediction = predict_image(image_model, image_to_process)

                # Product lookup
                product_name, expected_class = "Not found in catalog", "N/A"
                product_id_str = str(product_id_input).strip()
                if not catalog_df.empty:
                    product_details = catalog_df[catalog_df['product_id'].astype(str) == product_id_str]
                    if not product_details.empty:
                        product_name = product_details.iloc[0]['product_name']
                        expected_class = product_details.iloc[0]['article_type']

                complaint_mismatch = (predicted_class != expected_class) and (expected_class != "N/A")
                risk_score = calculate_risk_score(severity, image_prediction, complaint_mismatch)

            # ── Section: Product Image ──
            st.markdown('<div class="section-label">Submitted Image</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image(image_to_process, caption=caption, use_container_width=True)

            # ── Section: AI Signal Summary ──
            st.markdown('<div class="section-label" style="margin-top:1.25rem;">AI Signal Summary</div>', unsafe_allow_html=True)

            img_tag_cls = "ai-tag-defective" if image_prediction == "Defective" else "ai-tag-normal"
            img_tag_icon = "⚠" if image_prediction == "Defective" else "✓"
            mismatch_cls = "ai-tag-defective" if complaint_mismatch else "ai-tag-normal"
            mismatch_text = "Mismatch" if complaint_mismatch else "Match"
            mismatch_icon = "⚠" if complaint_mismatch else "✓"

            st.markdown(f"""
            <div class="insight-card">
                <div style="display:flex;flex-direction:column;gap:2px;">
                    <div class="info-row">
                        <span class="info-key">Product ID</span>
                        <span class="info-val">{product_id_str}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-key">Catalog Name</span>
                        <span class="info-val" style="font-family:'DM Sans',sans-serif;font-size:0.83rem;">{product_name[:32]}{'...' if len(product_name) > 32 else ''}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-key">Expected Category</span>
                        <span class="info-val">{expected_class}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-key">AI Predicted Category</span>
                        <span class="info-val">{predicted_class}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-key">Category Signal</span>
                        <span class="ai-tag {mismatch_cls}">{mismatch_icon} {mismatch_text}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-key">Image Assessment</span>
                        <span class="ai-tag {img_tag_cls}">{img_tag_icon} {image_prediction}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-key">Severity Input</span>
                        <span class="info-val">{severity} / 10</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Risk Score ──
            rc = risk_color_class(risk_score)
            bar_col = risk_bar_color(risk_score)
            risk_label = "HIGH RISK" if risk_score >= 70 else "MODERATE" if risk_score >= 40 else "LOW RISK"

            st.markdown(f"""
            <div class="risk-container">
                <div class="risk-header">
                    <span class="risk-title">Composite Risk Score</span>
                    <span style="font-size:0.72rem;color:#6b7280;font-family:'DM Mono',monospace;">{risk_label}</span>
                </div>
                <div class="risk-score-display {rc}">{risk_score}<span style="font-size:1.2rem;opacity:0.5;">%</span></div>
                <div class="risk-bar-track">
                    <div class="risk-bar-fill" style="width:{risk_score}%;background:{bar_col};"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#4b5563;font-family:'DM Mono',monospace;">
                    <span>0 · Safe</span><span>40 · Review</span><span>70 · Reject</span><span>100 · Fraud</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Verdict ──
            if approve_checkbox:
                st.markdown('<div class="verdict verdict-manual">⚡ Manually Approved — Override Active</div>', unsafe_allow_html=True)
            elif risk_score >= 70:
                st.markdown('<div class="verdict verdict-reject">✗ Reject Return — Investigate Further</div>', unsafe_allow_html=True)
            elif risk_score >= 40:
                st.markdown('<div class="verdict verdict-review">⚑ Manual Review Required</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict verdict-approve">✓ Approve Return — Low Risk Detected</div>', unsafe_allow_html=True)

            # ── Score Breakdown mini-metrics ──
            st.markdown('<div class="section-label" style="margin-top:1.25rem;">Score Breakdown</div>', unsafe_allow_html=True)

            base_score = severity * 10
            img_bonus = 40 if image_prediction == "Defective" else 0
            mm_bonus = 20 if complaint_mismatch else 0

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card" style="--accent:linear-gradient(90deg,#6366f1,#8b5cf6);">
                    <div class="metric-value" style="font-size:1.4rem;">{base_score}</div>
                    <div class="metric-label">Severity Base</div>
                    <div class="metric-delta delta-neutral">from severity × 10</div>
                </div>
                <div class="metric-card" style="--accent:{'linear-gradient(90deg,#f87171,#ef4444)' if img_bonus else 'linear-gradient(90deg,#34d399,#10b981)'};">
                    <div class="metric-value" style="font-size:1.4rem;">+{img_bonus}</div>
                    <div class="metric-label">Image Signal</div>
                    <div class="metric-delta {'delta-down' if img_bonus else 'delta-up'}">{'defect detected' if img_bonus else 'image normal'}</div>
                </div>
                <div class="metric-card" style="--accent:{'linear-gradient(90deg,#fbbf24,#f59e0b)' if mm_bonus else 'linear-gradient(90deg,#34d399,#10b981)'};">
                    <div class="metric-value" style="font-size:1.4rem;">+{mm_bonus}</div>
                    <div class="metric-label">Category Signal</div>
                    <div class="metric-delta {'delta-down' if mm_bonus else 'delta-up'}">{'mismatch penalty' if mm_bonus else 'category match'}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ─── GLOBAL REVIEW ANALYTICS ──────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-label">Global Review Intelligence</div>', unsafe_allow_html=True)

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

                total_reviews = sum(sentiment_counts.values())
                neg_pct = round(sentiment_counts["Negative"] / total_reviews * 100, 1) if total_reviews > 0 else 0
                pos_pct = round(sentiment_counts["Positive"] / total_reviews * 100, 1) if total_reviews > 0 else 0

                # Sentiment mini-metrics
                st.markdown(f"""
                <div class="metric-grid">
                    <div class="metric-card" style="--accent:linear-gradient(90deg,#34d399,#10b981);">
                        <div class="metric-value" style="font-size:1.4rem;">{sentiment_counts['Positive']:,}</div>
                        <div class="metric-label">Positive</div>
                        <div class="metric-delta delta-up">↑ {pos_pct}% of total</div>
                    </div>
                    <div class="metric-card" style="--accent:linear-gradient(90deg,#a1a1aa,#71717a);">
                        <div class="metric-value" style="font-size:1.4rem;">{sentiment_counts['Neutral']:,}</div>
                        <div class="metric-label">Neutral</div>
                        <div class="metric-delta delta-neutral">— {round(sentiment_counts['Neutral']/total_reviews*100,1) if total_reviews else 0}% of total</div>
                    </div>
                    <div class="metric-card" style="--accent:linear-gradient(90deg,#f87171,#ef4444);">
                        <div class="metric-value" style="font-size:1.4rem;">{sentiment_counts['Negative']:,}</div>
                        <div class="metric-label">Negative</div>
                        <div class="metric-delta delta-down">↓ {neg_pct}% of total</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Altair chart — dark theme
                df_sent = pd.DataFrame(list(sentiment_counts.items()), columns=["Sentiment", "Count"])
                color_scale = alt.Scale(
                    domain=["Positive", "Neutral", "Negative"],
                    range=["#34d399", "#6b7280", "#f87171"]
                )
                chart = alt.Chart(df_sent).mark_bar(
                    cornerRadiusTopLeft=5,
                    cornerRadiusTopRight=5
                ).encode(
                    x=alt.X("Sentiment:N",
                            sort=["Positive", "Neutral", "Negative"],
                            axis=alt.Axis(
                                labelColor="#6b7280",
                                tickColor="transparent",
                                domainColor="rgba(255,255,255,0.08)",
                                labelFont="DM Sans"
                            )),
                    y=alt.Y("Count:Q",
                            axis=alt.Axis(
                                tickMinStep=1,
                                labelColor="#6b7280",
                                gridColor="rgba(255,255,255,0.05)",
                                domainColor="transparent",
                                labelFont="DM Sans"
                            )),
                    color=alt.Color("Sentiment:N", scale=color_scale, legend=None),
                    tooltip=[
                        alt.Tooltip('Sentiment:N'),
                        alt.Tooltip('Count:Q'),
                    ]
                ).properties(
                    width="container",
                    height=200,
                    background="transparent",
                    title=alt.TitleParams(
                        text="Sentiment Distribution — All Reviews",
                        color="#4b5563",
                        fontSize=11,
                        font="DM Sans",
                        offset=10
                    )
                ).configure_view(
                    strokeWidth=0,
                    fill="transparent"
                )
                st.altair_chart(chart, use_container_width=True)

                # Negative review quotes
                if negative_reviews_text:
                    with st.expander(f"🔴 Recent Negative Reviews ({len(negative_reviews_text)} total)"):
                        for r in negative_reviews_text[:5]:
                            short = r[:180] + ("..." if len(r) > 180 else "")
                            st.markdown(f'<div class="review-quote">"{short}"</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="insight-card" style="text-align:center;color:#4b5563;font-size:0.85rem;padding:2rem;">
                    No reviews data loaded — place CSV at <code style="color:#6366f1;">data/raw/</code>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.2);border-radius:10px;padding:1rem 1.25rem;font-size:0.85rem;color:#f87171;">
                ⚠ Please provide all inputs: Product ID, Complaint, and an image (or use demo).
            </div>
            """, unsafe_allow_html=True)
