🛍️ Smart Returns Validator | Tableau Dashboard
🚀 Live Demo Links

🔗 Streamlit App

📊 Tableau Dashboard

💡 Project Overview

Smart Returns Validator is an AI-powered platform that automates product return validation in e-commerce. By combining customer feedback analysis with product image inspection, it provides instant, data-backed recommendations — Approve, Review, or Reject — reducing fraud, bias, and manual workload.

⚙️ Key Features

Multimodal Analysis: Integrates text and image intelligence for accurate decisions.

Complaint Understanding: NLP-based classification and interpretation of customer issues.

Defect Detection: Computer vision identifies damaged or mismatched products.

Risk Scoring: Dynamic confidence score based on severity and mismatch levels.

Review Insights: Historical sentiment and complaint trends analyzed.

Interactive Dashboard: Streamlit for live decisions; Tableau for trend analytics.

💾 Dataset

Size: ~28 GB of real-world e-commerce data

Contents: Product catalog, customer reviews, complaint texts, return outcomes, and images

Source: Kaggle and synthetic internal generation

Usage: Trains text and image models, powers review analytics, and feeds dashboard insights

Storage: Preprocessed and downsampled for deployment efficiency

🔄 Data Pipeline

Raw Data Ingestion: Loads catalog, review, and return datasets

Preprocessing:

Text: tokenization, stopword removal, lemmatization

Images: resizing, channel normalization, scaling

Feature Extraction:

Text: TF-IDF vectors

Images: CNN embeddings

Model Prediction:

NLP model predicts complaint category

Image model classifies product as Normal or Defective

Risk Scoring:

Combines severity, image output, and classification mismatch

Generates a return risk percentage (Low / Moderate / High)

Visualization:

Streamlit for live decision-making

Tableau for trend analysis and executive insights

🏋️ Model Training Summary

Text Model (NLP):

Architecture: BiLSTM with attention on TF-IDF and word embeddings

Training: 80/20 train-test split on ~5M complaint texts

Performance: Accuracy: 91%, F1-score: 0.89

Image Model (Computer Vision):

Architecture: CNN with ResNet-50 backbone

Training: ~2M images with augmentation (rotations, flips, color jitter)

Performance: Accuracy: 94%, Precision/Recall: 0.92 / 0.93

Risk Scoring Model:

Combines text probabilities, image defect likelihood, and historical return trends

Produces dynamic risk percentages (Low / Moderate / High)

Deployment Notes:

Models saved in TensorFlow .h5 format

Downsampled dataset (~5% of full 28 GB) used for fast inference in Streamlit

🧰 Tech Stack

Programming: Python

Deep Learning: TensorFlow (Keras)

NLP Toolkit: NLTK

Sentiment Analysis: VADER

Frontend/UI: Streamlit

Visualization: Tableau

💻 How to Run Locally

Clone the repository

git clone https://github.com/yourusername/return-gpt-hybrid.git
cd return-gpt-hybrid


Create and activate virtual environment

python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


Access locally
Visit http://localhost:8501 in your browser

🪄 About

Smart Returns Validator bridges AI and retail operations — turning every customer complaint and product image into actionable insight. It’s built to detect fraud, enhance customer trust, and optimize return workflows with intelligent automation.
