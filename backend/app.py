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
    'Abeta40', 'Abeta42', 'pTau181', 'pTau217','npTau217', 'GFAP', 'NfL',
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
    data = request.json

    # Raw inputs
    Abeta40 = data.get("Abeta40", 0.0)
    Abeta42 = data.get("Abeta42", 0.0)
    pTau181 = data.get("pTau181", 0.0)
    pTau217 = data.get("pTau217", 0.0)
    GFAP = data.get("GFAP", 0.0)
    NfL = data.get("NfL", 0.0)
    MMSE = data.get("MMSE", 0.0)
    DHA = data.get("DHA", 0.0)
    formic_acid = data.get("formic_acid", 0.0)
    lactoferrin = data.get("lactoferrin", 0.0)
    NTP = data.get("NTP", 0.0)
    npTau217 = data.get("npTau217", 0.0)
    Abeta_ratio = data.get("Abeta_ratio", 0.0)

    DXMPTR = [
        data.get("DXMPTR1", 0.0),
        data.get("DXMPTR2", 0.0),
        data.get("DXMPTR3", 0.0),
        data.get("DXMPTR4", 0.0),
        data.get("DXMPTR5", 0.0),
        data.get("DXMPTR6", 0.0),
    ]

    # ===== FEATURE ENGINEERING (MATCH TRAINING) =====
    Abeta42_40_ratio = Abeta42 / (Abeta40 + 1e-10)
    pTau217_Abeta42_ratio = pTau217 / (Abeta42 + 1e-10)
    pTau181_Abeta42_ratio = pTau181 / (Abeta42 + 1e-10)
    NfL_GFAP_ratio = NfL / (GFAP + 1e-10)

    biomarker_burden = pTau217 + GFAP + NfL
    biomarker_burden -= Abeta42 * 0.01
    DXMDES = 1.0


    # ===== FINAL FEATURE VECTOR (26 FEATURES) =====
    X_input = np.array([[
        *DXMPTR,DXMDES,
        Abeta40, Abeta42, pTau181, pTau217,npTau217, GFAP, NfL,
        MMSE, DHA, formic_acid, lactoferrin, NTP, Abeta_ratio,
        Abeta42_40_ratio,
        pTau217_Abeta42_ratio,
        pTau181_Abeta42_ratio,
        NfL_GFAP_ratio,
        biomarker_burden
    ]])

    proba = model.predict_proba(X_input)[0]
    pred_idx = int(np.argmax(proba))
    diagnosis = label_encoder.inverse_transform([pred_idx])[0]

    return jsonify({
        "diagnosis": diagnosis,
        "probability": float(proba[pred_idx]),
        "all_probabilities": {
            label_encoder.classes_[i]: float(proba[i])
            for i in range(len(proba))
        },
        "risk_level": (
            "high" if proba[pred_idx] > 0.7 else
            "medium" if proba[pred_idx] > 0.4 else
            "low"
        ),
        "predictions_until_retrain": 1000
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


print(model.n_features_in_)
