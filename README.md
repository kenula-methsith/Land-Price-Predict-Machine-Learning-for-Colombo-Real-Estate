# Land-Price-Predict-Machine-Learning-for-Colombo-Real-Estate

Land Price Predict: Machine Learning for Colombo Real Estate

📌 Project Overview

This project builds a machine learning web application to predict land prices in Colombo, Sri Lanka.
It uses ensemble learning (LightGBM + XGBoost + Gradient Boosting + Stacking) to achieve high accuracy (R² > 0.85) in estimating land values based on property and location features.

The project is deployed with a Flask web app, allowing users to input land details (size, type, address) and get real-time price predictions with insights into derived location-based features.

Features

Data-Driven Predictions: Model trained on historical land price data from Colombo.
Location Intelligence: Distance-based features (Fort, banks, supermarkets, expressways, railways).
Ensemble Model: Combines LightGBM, XGBoost, and Gradient Boosting with a Ridge meta-model.
Web Application: User-friendly Flask interface for price prediction.
Explainability: Returns derived features alongside predictions.

├── app.py                             
├── colombo_land_price_model.pkl   
├── model_features.pkl             
├── address_mapping.csv            
├── Land prices of Colombo district-Sri Lanka.csv  
├── templates
  └── index.html                
