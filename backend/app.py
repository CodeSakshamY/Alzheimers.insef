import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ------------------------
# Load model and label encoder
# ------------------------
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# ------------------------
# Flask app
# ------------------------
app = Flask(__name__)
CORS(app)

# ------------------------
# Prediction route
# ------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # ------------------------
    # 1. Fixed + raw base features
    # ------------------------
    # Fixed diagnostic measure not input by frontend
    data["DXMDES"] = 1

    # If Abeta_ratio missing, compute it
    if "Abeta_ratio" not in data and "Abeta42" in data and "Abeta40" in data:
        data["Abeta_ratio"] = data["Abeta42"] / (data["Abeta40"] + 1e-10)

    # ------------------------
    # 2. Engineered features (exactly as in ML)
    # ------------------------
    engineered_features = []

    # Aβ42/Aβ40 ratio (gold standard biomarker)
    if "Abeta42" in data and "Abeta40" in data:
        data["Abeta42_40_ratio"] = data["Abeta42"] / (data["Abeta40"] + 1e-10)
        engineered_features.append("Abeta42_40_ratio")

    # Tau/Aβ42 ratios
    if "pTau217" in data and "Abeta42" in data:
        data["pTau217_Abeta42_ratio"] = data["pTau217"] / (data["Abeta42"] + 1e-10)
        engineered_features.append("pTau217_Abeta42_ratio")

    if "pTau181" in data and "Abeta42" in data:
        data["pTau181_Abeta42_ratio"] = data["pTau181"] / (data["Abeta42"] + 1e-10)
        engineered_features.append("pTau181_Abeta42_ratio")

    # NfL/GFAP ratio
    if "NfL" in data and "GFAP" in data:
        data["NfL_GFAP_ratio"] = data["NfL"] / (data["GFAP"] + 1e-10)
        engineered_features.append("NfL_GFAP_ratio")

    # Composite biomarker burden
    if all(k in data for k in ["pTau217", "GFAP", "NfL"]):
        data["biomarker_burden"] = data["pTau217"] + data["GFAP"] + data["NfL"]
        if "Abeta42" in data:
            data["biomarker_burden"] -= data["Abeta42"] * 0.01
        engineered_features.append("biomarker_burden")

    # DXMPTR aggregations
    dxmptr_keys = [k for k in data.keys() if k.startswith("DXMPTR")]
    if len(dxmptr_keys) >= 2:
        dxmptr_values = [data[k] for k in dxmptr_keys]
        data["DXMPTR_mean"] = np.mean(dxmptr_values)
        data["DXMPTR_std"] = np.std(dxmptr_values)
        engineered_features.extend(["DXMPTR_mean", "DXMPTR_std"])

    # ------------------------
    # 3. Ensure all model features exist (fill missing with 0)
    # ------------------------
    ALL_FEATURES = [
        "DXMDES",
        "DXMPTR1","DXMPTR2","DXMPTR3","DXMPTR4","DXMPTR5","DXMPTR6",
        "Abeta40","Abeta42","Abeta_ratio",
        "GFAP","NfL",
        "npTau217","pTau181","pTau217",
        "DHA","FORMICACID","LACTOFERRIN","MMSE","NTP",
        "Abeta42_40_ratio",
        "pTau217_Abeta42_ratio",
        "pTau181_Abeta42_ratio",
        "NfL_GFAP_ratio",
        "biomarker_burden",
        "DXMPTR_mean",
        "DXMPTR_std"
    ]

    # Fill any missing features with 0 (or 1 for DXMDES already handled)
    for f in ALL_FEATURES:
        if f not in data:
            data[f] = 0

    # ------------------------
    # 4. Prepare input and predict
    # ------------------------
    X = [[data[f] for f in ALL_FEATURES]]

    proba = model.predict_proba(X)[0]
    idx = int(proba.argmax())
    label = label_encoder.inverse_transform([idx])[0]

    # ------------------------
    # 5. Return JSON response
    # ------------------------
    return jsonify({
        "prediction": label,
        "confidence": float(proba[idx])
    })

# ------------------------
# Run server
# ------------------------
if __name__ == "__main__":
    print(f"Model expects {model.n_features_in_} features")
    app.run(host="0.0.0.0", port=5000)
