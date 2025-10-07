Live App Demo | Tableau Dashboard

Project Overview: This is a full-stack data science application built to solve a key challenge in e-commerce: efficiently and accurately validating customer return requests. Manual validation is slow, costly, and prone to inconsistency. This AI-powered assistant automates the process by using a multimodal approach, analyzing both the customer's written complaint and the visual evidence they provide.

The result is an interactive tool that provides an instant, evidence-based recommendation (Approve, Review, or Reject), enabling operations teams to handle returns more effectively and identify potential fraud.

Key Features:

Multimodal Analysis: The system doesn't just look at text or images alone; it integrates both to make a more intelligent decision.

NLP Complaint Classification: It uses a trained model to understand and categorize the customer's complaint from the text.

Computer Vision for Defect Detection: A deep learning model inspects the product image to visually confirm if it's defective.

Risk Scoring Algorithm: A weighted scoring system combines the complaint severity, model predictions, and product data to generate an intuitive risk score.

Historical Data Insights: The app pulls and analyzes past reviews for the product, providing valuable context about known issues or customer sentiment.

Interactive UI: Built with Streamlit for a responsive and user-friendly experience.

Tech Stack:

Language: Python

Data Science & ML: Pandas, NumPy, Scikit-learn, TensorFlow (Keras), NLTK (for NLP), VADER (for Sentiment Analysis)

Web Framework: Streamlit

Data Visualization: Altair, Tableau Public

Deployment: Streamlit Community Cloud, Git/GitHub
