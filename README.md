Hotel Revenue Protection: Booking Cancellation Prediction System

A machine learning solution designed to predict hotel booking cancellations and provide real-time revenue protection policy recommendations.

Project Overview

Unpredicted hotel booking cancellations result in significant revenue loss and vacant rooms. This project utilizes historical hotel booking demand data to build a high-accuracy classification model that identifies high-risk bookings in advance, allowing hotels to enforce appropriate deposit policies (e.g., non-refundable rates or advance deposits).

Key Features & Highlights

High Predictive Performance: Achieved an outstanding ROC-AUC score of 0.9067 using an optimized Random Forest pipeline.

Data Leakage Prevention: Built robust ColumnTransformer pipelines combining StandardScaler and OneHotEncoder to ensure strict separation of preprocessing and modeling.

Advanced Feature Engineering: Engineered critical domain-specific predictors including total stay nights, total guests, special request flags, and previous cancellation indicators.

Interactive Web Application: Deployed a fully functional, real-time prediction tool built with Streamlit for hotel managers to assess risk instantly.

Dataset

Source: Hotel Booking Demand Dataset (~119,390 records).

Target Variable: is_canceled (Binary: 0 for confirmed, 1 for canceled).

Project Structure

├── hotel_cancellation_model.joblib # Saved production pipeline
├── app.py                          # Streamlit web application
├── requirements.txt                # Required Python packages
└── README.md                       # Project documentation