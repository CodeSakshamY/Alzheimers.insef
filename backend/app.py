import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS


# Load model and label encoder
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Features your model expects
FEATURE_COLUMNS = [
    'DXMPTR1', 'DXMPTR2', 'DXMPTR3', 'DXMPTR4', 'DXMPTR5', 'DXMPTR6',
    'Abeta40', 'Abeta42', 'pTau181', 'pTau217', 'GFAP', 'NfL',
    'MMSE', 'DHA', 'formic_acid', 'lactoferrin', 'NTP', 'Abeta_ratio'
]

def engineer_features_single(features):
    """Optional: replicate your feature engineering here"""
    if 'Abeta42' in features and 'Abeta40' in features:
        features['Abeta42_40_ratio'] = features['Abeta42'] / (features['Abeta40'] + 1e-10)
    if 'pTau217' in features and 'Abeta42' in features:
        features['pTau217_Abeta42_ratio'] = features['pTau217'] / (features['Abeta42'] + 1e-10)
    if 'pTau181' in features and 'Abeta42' in features:
        features['pTau181_Abeta42_ratio'] = features['pTau181'] / (features['Abeta42'] + 1e-10)
    if 'NfL' in features and 'GFAP' in features:
        features['NfL_GFAP_ratio'] = features['NfL'] / (features['GFAP'] + 1e-10)
    dxmptr_keys = [k for k in features.keys() if k.startswith('DXMPTR')]
    if len(dxmptr_keys) >= 2:
        dxmptr_values = [features[k] for k in dxmptr_keys]
        features['DXMPTR_mean'] = np.mean(dxmptr_values)
        features['DXMPTR_std'] = np.std(dxmptr_values)
    return features

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = engineer_features_single(data)
    
    # Ensure the feature order
    X_input = np.array([[features.get(f, 0) for f in FEATURE_COLUMNS + ['Abeta42_40_ratio','pTau217_Abeta42_ratio','pTau181_Abeta42_ratio','NfL_GFAP_ratio','DXMPTR_mean','DXMPTR_std']]])
    
    proba = model.predict_proba(X_input)[0]
    pred_class = np.argmax(proba)
    pred_label = label_encoder.inverse_transform([pred_class])[0]
    max_prob = float(proba[pred_class])
    
    # Risk level
    if max_prob < 0.4:
        risk = "Low"
    elif max_prob <= 0.7:
        risk = "Medium"
    else:
        risk = "High"
    
    all_probabilities = {label_encoder.inverse_transform([i])[0]: float(p) for i,p in enumerate(proba)}
    
    return jsonify({
        "diagnosis": pred_label,
        "probability": max_prob,
        "risk_level": risk,
        "all_probabilities": all_probabilities,
        "predictions_until_retrain": 100  # dummy
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
