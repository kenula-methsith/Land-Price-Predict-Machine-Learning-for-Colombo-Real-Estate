import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model and features
with open('colombo_land_price_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('model_features.pkl', 'rb') as file:
    model_features = pickle.load(file)

# Load address mapping for distance-based features
address_data = pd.read_csv('address_mapping.csv')

@app.route('/')
def index():
    # Get unique land types and addresses for form dropdowns
    df = pd.read_csv('Land prices of Colombo district-Sri Lanka.csv')
    land_types = df['Land_type'].unique().tolist()
    addresses = df['Address'].unique().tolist()
    return render_template('index.html', land_types=land_types, addresses=addresses)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'Land_type': request.form['land_type'],
            'Land_size(Perches)': float(request.form['land_size']),
            'Address': request.form['address']
        }

        # Get distance-based features from address mapping
        address_row = address_data[address_data['Address'] == data['Address']]
        if address_row.empty:
            return jsonify({'prediction': None, 'error': 'Address not found in dataset'})

        distance_features = {
            'Distance from fort': address_row['Distance from fort'].iloc[0],
            'min_dist_nearest_bank': address_row['min_dist_nearest_bank'].iloc[0],
            'min_dist_nearest_Supermarket': address_row['min_dist_nearest_Supermarket'].iloc[0],
            'min_dist_nearest_express': address_row['min_dist_nearest_express'].iloc[0],
            'min_dist_nearest_railway': address_row['min_dist_nearest_railway'].iloc[0]
        }
        data.update(distance_features)

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure all model features are present
        for feature in model_features:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value for missing numerical features

        # Select only model features in correct order
        input_df = input_df[model_features]

        # Make prediction (model predicts log-transformed price, so reverse it)
        log_pred = model.predict(input_df)[0]
        prediction = np.expm1(log_pred)  # Reverse log1p transformation

        return jsonify({
    'prediction': f'{prediction:,.0f}',
    'error': None,
    'derived_features': {key: round(value, 2) for key, value in distance_features.items()}
})
    except Exception as e:
        return jsonify({'prediction': None, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)