from http.server import BaseHTTPRequestHandler
import json
import numpy as np
from api.predict_logic import load_and_train_model, engineer_features_single, FEATURE_COLUMNS

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            model, label_encoder, feature_names = load_and_train_model()
            features_dict = engineer_features_single(data, FEATURE_COLUMNS)
            X_input = np.array([[features_dict.get(feat, 0) for feat in feature_names]])

            probabilities = model.predict_proba(X_input)[0]
            predicted_class = int(np.argmax(probabilities))
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            max_probability = float(probabilities[predicted_class])

            if max_probability < 0.40:
                risk_level = "Low"
            elif max_probability <= 0.70:
                risk_level = "Medium"
            else:
                risk_level = "High"

            prob_dict = {label_encoder.inverse_transform([i])[0]: float(prob)
                         for i, prob in enumerate(probabilities)}

            response = {
                'diagnosis': predicted_label,
                'probability': max_probability,
                'risk_level': risk_level,
                'all_probabilities': prob_dict,
                'predictions_until_retrain': 50  # optional counter
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
