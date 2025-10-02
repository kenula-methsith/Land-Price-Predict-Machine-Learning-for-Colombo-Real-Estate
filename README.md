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


<img width="1919" height="978" alt="Screenshot 2025-10-02 160635" src="https://github.com/user-attachments/assets/4ff092e9-e561-464a-9fb0-627a628bfa58" />
<img width="1911" height="967" alt="Screenshot 2025-10-02 160845" src="https://github.com/user-attachments/assets/d1de2a07-e597-40da-90b7-b7edcd323f02" />
<img width="1919" height="925" alt="Screenshot 2025-10-02 160943" src="https://github.com/user-attachments/assets/7b0b9502-be0b-4db7-b68b-f74d263ef224" />
