import json
import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

MODEL_CACHE = {'model': None, 'label_encoder': None, 'feature_names': None}
RANDOM_SEED = 42
FEATURE_COLUMNS = [
    'DXMPTR1','DXMPTR2','DXMPTR3','DXMPTR4','DXMPTR5','DXMPTR6',
    'Abeta40','Abeta42','pTau181','pTau217','GFAP','NfL',
    'MMSE','DHA','formic_acid','lactoferrin','NTP','Abeta_ratio'
]
TARGET_COLUMN = 'Diagnosis'

def load_and_train_model():
    if MODEL_CACHE['model']:
        return MODEL_CACHE['model'], MODEL_CACHE['label_encoder'], MODEL_CACHE['feature_names']

    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset.xlsx')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("dataset.xlsx not found")

    df = pd.read_excel(dataset_path)
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    df_filtered = df[available_features + [TARGET_COLUMN]].dropna()

    # Feature engineering
    if 'Abeta42' in available_features and 'Abeta40' in available_features:
        df_filtered['Abeta42_40_ratio'] = df_filtered['Abeta42'] / (df_filtered['Abeta40'] + 1e-10)
    if 'pTau217' in available_features and 'Abeta42' in available_features:
        df_filtered['pTau217_Abeta42_ratio'] = df_filtered['pTau217'] / (df_filtered['Abeta42'] + 1e-10)
    if 'pTau181' in available_features and 'Abeta42' in available_features:
        df_filtered['pTau181_Abeta42_ratio'] = df_filtered['pTau181'] / (df_filtered['Abeta42'] + 1e-10)
    if 'NfL' in available_features and 'GFAP' in available_features:
        df_filtered['NfL_GFAP_ratio'] = df_filtered['NfL'] / (df_filtered['GFAP'] + 1e-10)
    dxmptr_cols = [col for col in available_features if col.startswith('DXMPTR')]
    if len(dxmptr_cols) >= 2:
        df_filtered['DXMPTR_mean'] = df_filtered[dxmptr_cols].mean(axis=1)
        df_filtered['DXMPTR_std'] = df_filtered[dxmptr_cols].std(axis=1)

    all_features = available_features + [c for c in df_filtered.columns if c not in available_features + [TARGET_COLUMN]]

    X = df_filtered[all_features].values
    y = df_filtered[TARGET_COLUMN].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_SEED, stratify=y_encoded
    )

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = LGBMClassifier(
        objective='multiclass',
        num_class=len(label_encoder.classes_),
        metric='multi_logloss',
        boosting_type='gbdt',
        verbosity=-1,
        random_state=RANDOM_SEED,
        class_weight=class_weights_dict,
        is_unbalance=True,
        num_leaves=50,
        max_depth=10,
        learning_rate=0.05,
        n_estimators=500,
        min_child_samples=30,
        lambda_l1=5.0,
        lambda_l2=5.0,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5
    )

    model.fit(X_train_res, y_train_res)

    MODEL_CACHE['model'] = model
    MODEL_CACHE['label_encoder'] = label_encoder
    MODEL_CACHE['feature_names'] = all_features

    return model, label_encoder, all_features

def engineer_features_single(input_data):
    features = input_data.copy()
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
        dx_values = [features[k] for k in dxmptr_keys]
        features['DXMPTR_mean'] = np.mean(dx_values)
        features['DXMPTR_std'] = np.std(dx_values)
    return features

# ====================
# Vercel serverless handler
# ====================
def handler(req):
    try:
        data = json.loads(req.body.decode("utf-8"))
        model, label_encoder, feature_names = load_and_train_model()
        features = engineer_features_single(data)
        X_input = np.array([[features.get(f, 0) for f in feature_names]])

        probs = model.predict_proba(X_input)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        max_prob = float(probs[pred_idx])

        risk = "Low" if max_prob < 0.4 else "Medium" if max_prob <= 0.7 else "High"
        all_probs = {label_encoder.inverse_transform([i])[0]: float(p) for i, p in enumerate(probs)}

        response = {
            "diagnosis": pred_label,
            "probability": max_prob,
            "risk_level": risk,
            "all_probabilities": all_probs,
            "predictions_until_retrain": 50
        }

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps(response)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)})
        }
