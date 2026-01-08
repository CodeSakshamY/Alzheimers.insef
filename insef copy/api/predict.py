import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

app = FastAPI()

RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "DXMPTR1", "DXMPTR2", "DXMPTR3", "DXMPTR4", "DXMPTR5", "DXMPTR6",
    "Abeta40", "Abeta42", "pTau181", "pTau217", "GFAP", "NfL",
    "MMSE", "DHA", "formic_acid", "lactoferrin", "NTP", "Abeta_ratio"
]

TARGET_COLUMN = "Diagnosis"

_model_cache = {
    "model": None,
    "encoder": None,
    "features": None
}

class PatientData(BaseModel):
    Abeta40: float
    Abeta42: float
    pTau181: float
    pTau217: float
    GFAP: float
    NfL: float
    Abeta_ratio: float
    MMSE: float
    DHA: float
    formic_acid: float
    lactoferrin: float
    NTP: float
    DXMPTR1: float
    DXMPTR2: float
    DXMPTR3: float
    DXMPTR4: float
    DXMPTR5: float
    DXMPTR6: float


def load_and_train_model():
    if _model_cache["model"] is not None:
        return _model_cache.values()

    df = pd.read_excel("dataset.xlsx").dropna()

    X_base = df[FEATURE_COLUMNS].copy()

    # Feature engineering
    X_base["Abeta42_40_ratio"] = X_base["Abeta42"] / (X_base["Abeta40"] + 1e-10)
    X_base["pTau217_Abeta42_ratio"] = X_base["pTau217"] / (X_base["Abeta42"] + 1e-10)
    X_base["DXMPTR_mean"] = X_base[[c for c in FEATURE_COLUMNS if c.startswith("DXMPTR")]].mean(axis=1)
    X_base["DXMPTR_std"] = X_base[[c for c in FEATURE_COLUMNS if c.startswith("DXMPTR")]].std(axis=1)

    y = LabelEncoder().fit_transform(df[TARGET_COLUMN])

    X_train, _, y_train, _ = train_test_split(
        X_base, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    smote = SMOTE(random_state=RANDOM_SEED)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.05,
        random_state=RANDOM_SEED
    )

    model.fit(X_train, y_train)

    encoder = LabelEncoder()
    encoder.fit(df[TARGET_COLUMN])

    _model_cache["model"] = model
    _model_cache["encoder"] = encoder
    _model_cache["features"] = X_base.columns.tolist()

    return model, encoder, X_base.columns.tolist()


@app.post("/predict")
def predict(data: PatientData):
    model, encoder, features = load_and_train_model()

    input_dict = data.dict()

    input_dict["Abeta42_40_ratio"] = input_dict["Abeta42"] / (input_dict["Abeta40"] + 1e-10)
    input_dict["pTau217_Abeta42_ratio"] = input_dict["pTau217"] / (input_dict["Abeta42"] + 1e-10)
    input_dict["DXMPTR_mean"] = np.mean([input_dict[k] for k in input_dict if k.startswith("DXMPTR")])
    input_dict["DXMPTR_std"] = np.std([input_dict[k] for k in input_dict if k.startswith("DXMPTR")])

    X = np.array([[input_dict.get(f, 0) for f in features]])

    probs = model.predict_proba(X)[0]
    idx = int(np.argmax(probs))

    return {
        "diagnosis": encoder.inverse_transform([idx])[0],
        "probability": float(probs[idx]),
        "risk_level": "Low" if probs[idx] < 0.4 else "Medium" if probs[idx] < 0.7 else "High",
        "all_probabilities": {
            encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(probs)
        }
    }
