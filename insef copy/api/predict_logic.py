# api/predict_logic.py
import warnings
warnings.filterwarnings('ignore')

import os
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Optional: pickle if you want to save model
# import pickle

# Global cache
_model_cache = {
    'model': None,
    'label_encoder': None,
    'feature_names': None
}

RANDOM_SEED = 42
FEATURE_COLUMNS = [
    'DXMPTR1', 'DXMPTR2', 'DXMPTR3', 'DXMPTR4', 'DXMPTR5', 'DXMPTR6',
    'Abeta40', 'Abeta42', 'pTau181', 'pTau217', 'GFAP', 'NfL',
    'MMSE', 'DHA', 'formic_acid', 'lactoferrin', 'NTP', 'Abeta_ratio'
]
TARGET_COLUMN = 'Diagnosis'

def load_and_train_model():
    if _model_cache['model'] is not None:
        return _model_cache['model'], _model_cache['label_encoder'], _model_cache['feature_names']

    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset.xlsx')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("dataset.xlsx not found")

    df = pd.read_excel(dataset_path)
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    df_filtered = df[available_features + [TARGET_COLUMN]].copy()
    df_filtered = df_filtered.dropna()

    engineered_features = []

    if 'Abeta42' in available_features and 'Abeta40' in available_features:
        df_filtered['Abeta42_40_ratio'] = df_filtered['Abeta42'] / (df_filtered['Abeta40'] + 1e-10)
        engineered_features.append('Abeta42_40_ratio')

    if 'pTau217' in available_features and 'Abeta42' in available_features:
        df_filtered['pTau217_Abeta42_ratio'] = df_filtered['pTau217'] / (df_filtered['Abeta42'] + 1e-10)
        engineered_features.append('pTau217_Abeta42_ratio')

    if 'pTau181' in available_features and 'Abeta42' in available_features:
        df_filtered['pTau181_Abeta42_ratio'] = df_filtered['pTau181'] / (df_filtered['Abeta42'] + 1e-10)
        engineered_features.append('pTau181_Abeta42_ratio')

    if 'NfL' in available_features and 'GFAP' in available_features:
        df_filtered['NfL_GFAP_ratio'] = df_filtered['NfL'] / (df_filtered['GFAP'] + 1e-10)
        engineered_features.append('NfL_GFAP_ratio')

    if 'pTau217' in available_features and 'GFAP' in available_features and 'NfL' in available_features:
        df_filtered['biomarker_burden'] = df_filtered['pTau217'] + df_filtered['GFAP'] + df_filtered['NfL']
        if 'Abeta42' in available_features:
            df_filtered['biomarker_burden'] -= df_filtered['Abeta42'] * 0.01
        engineered_features.append('biomarker_burden')

    dxmptr_cols = [col for col in available_features if col.startswith('DXMPTR')]
    if len(dxmptr_cols) >= 2:
        df_filtered['DXMPTR_mean'] = df_filtered[dxmptr_cols].mean(axis=1)
        df_filtered['DXMPTR_std'] = df_filtered[dxmptr_cols].std(axis=1)
        engineered_features.extend(['DXMPTR_mean', 'DXMPTR_std'])

    all_features = available_features + engineered_features

    X = df_filtered[all_features].values
    y = df_filtered[TARGET_COLUMN].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=RANDOM_SEED, stratify=y_encoded
    )

    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    params = {
        'objective': 'multiclass',
        'num_class': len(label_encoder.classes_),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': RANDOM_SEED,
        'class_weight': class_weights_dict,
        'is_unbalance': True,
        'num_leaves': 50,
        'max_depth': 10,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_samples': 30,
        'lambda_l1': 5.0,
        'lambda_l2': 5.0,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
    }

    model = LGBMClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)

    _model_cache['model'] = model
    _model_cache['label_encoder'] = label_encoder
    _model_cache['feature_names'] = all_features

    return model, label_encoder, all_features


def engineer_features_single(input_data, available_features):
    features = input_data.copy()

    if 'Abeta42' in features and 'Abeta40' in features:
        features['Abeta42_40_ratio'] = features['Abeta42'] / (features['Abeta40'] + 1e-10)

    if 'pTau217' in features and 'Abeta42' in features:
        features['pTau217_Abeta42_ratio'] = features['pTau217'] / (features['Abeta42'] + 1e-10)

    if 'pTau181' in features and 'Abeta42' in features:
        features['pTau181_Abeta42_ratio'] = features['pTau181'] / (features['Abeta42'] + 1e-10)

    if 'NfL' in features and 'GFAP' in features:
        features['NfL_GFAP_ratio'] = features['NfL'] / (features['GFAP'] + 1e-10)

    if 'pTau217' in features and 'GFAP' in features and 'NfL' in features:
        features['biomarker_burden'] = features['pTau217'] + features['GFAP'] + features['NfL']
        if 'Abeta42' in features:
            features['biomarker_burden'] -= features['Abeta42'] * 0.01

    dxmptr_keys = [k for k in features.keys() if k.startswith('DXMPTR')]
    if len(dxmptr_keys) >= 2:
        dxmptr_values = [features[k] for k in dxmptr_keys]
        features['DXMPTR_mean'] = np.mean(dxmptr_values)
        features['DXMPTR_std'] = np.std(dxmptr_values)

    return features


def predict_single(input_data):
    model, label_encoder, feature_names = load_and_train_model()
    features_dict = engineer_features_single(input_data, FEATURE_COLUMNS)
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

    return {
        'diagnosis': predicted_label,
        'probability': max_probability,
        'risk_level': risk_level,
        'all_probabilities': prob_dict
    }

