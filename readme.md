🤖 AI-Powered Return Assistant
Live Demo: [https://ai-return-assistant-mgjhvkazynfsvdlqe4gcpa.streamlit.app/]

Project Overview
This project is a full-stack data science application built to solve a key challenge in e-commerce: efficiently and accurately validating customer return requests. Manual validation is slow, costly, and prone to inconsistency. This AI-powered assistant automates the process by using a multimodal approach, analyzing both the customer's written complaint and the visual evidence they provide.

The result is an interactive tool that provides an instant, evidence-based recommendation (Approve, Review, or Reject), enabling operations teams to handle returns more effectively and identify potential fraud.

Key Features
Multimodal Analysis: The system doesn't just look at text or images alone; it integrates both to make a more intelligent decision.

NLP-Powered Complaint Classification: A machine learning model trained with Scikit-learn and NLTK reads and categorizes the customer's complaint text.

Computer Vision for Defect Detection: A deep learning model built with TensorFlow/Keras analyzes the product image to identify visual evidence of defects.

Intelligent Recommendation Engine: A logic-based system provides a final recommendation based on the combined outputs of the AI models.

Interactive Dashboard: The entire system is deployed as a user-friendly web application using Streamlit, allowing non-technical users to get instant AI-powered insights.

Tech Stack
Data Analysis & Manipulation: Pandas, NumPy

Machine Learning (NLP): Scikit-learn, NLTK

Deep Learning (Vision): TensorFlow, Keras, Pillow

Data Visualization: Matplotlib, Seaborn

Web Application & Deployment: Streamlit, Streamlit Community Cloud

Core Language: Python
