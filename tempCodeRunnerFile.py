import cv2
import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os

from HarmoMed import HarmoMed_lir

# ===============================
# CONFIG
# ===============================

YELLOW_HSV_LOWER = np.array([18, 80, 80])
YELLOW_HSV_UPPER = np.array([35, 255, 255])

STANDARD_SENSOR = {
    "MQ_135": 200,
    "MQ_136": 50,
    "MQ_137": 25,
    "MQ_138": 30
}

MODEL_PATH = "risk_model.pkl"
SCALER_PATH = "scaler.pkl"

FEATURE_NAMES = [
    "yellow_ratio", "h_mean", "s_mean", "v_mean",
    "MQ135_z", "MQ136_z", "MQ137_z", "MQ138_z",
    "questionnaire"
]

# ===============================
# DATA STRUCTURE
# ===============================

@dataclass
class AnalysisResult:
    image_features: List[float]
    sensor_features: List[float]
    questionnaire_score: float
    risk_probability: float
    risk_level: str

# ===============================
# IMAGE FEATURE EXTRACTION
# ===============================

def extract_image_features(image_path: str) -> List[float]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)

    yellow_ratio = np.count_nonzero(mask) / mask.size
    h_mean = np.mean(h[mask > 0]) if np.any(mask) else 0
    s_mean = np.mean(s[mask > 0]) if np.any(mask) else 0
    v_mean = np.mean(v[mask > 0]) if np.any(mask) else 0

    return [
        round(yellow_ratio * 100, 3),
        round(h_mean, 3),
        round(s_mean, 3),
        round(v_mean, 3)
    ]

# ===============================
# SENSOR FEATURE EXTRACTION
# ===============================

def extract_sensor_features(csv_path: str) -> List[float]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    features = []

    for sensor, ref in STANDARD_SENSOR.items():
        if sensor not in df.columns:
            raise ValueError(f"Missing sensor column: {sensor}")

        mean_val = df[sensor].mean()
        z = (mean_val - ref) / ref
        features.append(round(z, 3))

    return features

# ===============================
# QUESTIONNAIRE
# ===============================

def questionnaire_score(answers: List[int]) -> float:
    if len(answers) != 10:
        raise ValueError("Questionnaire must have 10 answers")
    if any(a < 0 or a > 4 for a in answers):
        raise ValueError("Answers must be between 0–4")

    return round(sum(answers) / 40, 3)

# ===============================
# RISK LABEL
# ===============================

def risk_label(prob: float) -> str:
    if prob < 0.30:
        return "Low risk"
    elif prob < 0.60:
        return "Moderate risk"
    elif prob < 0.80:
        return "High risk"
    else:
        return "Very high risk"

# ===============================
# SYNTHETIC MODEL TRAINING
# ===============================

def train_synthetic_model(n_samples: int = 800):
    np.random.seed(42)

    X, y = [], []

    for _ in range(n_samples):
        yellow = np.clip(np.random.normal(15, 10), 0, 60)
        h = np.random.uniform(15, 35)
        s = np.random.uniform(80, 255)
        v = np.random.uniform(80, 255)

        mq135 = np.random.normal(200, 60)
        mq136 = np.random.normal(50, 20)
        mq137 = np.random.normal(25, 10)
        mq138 = np.random.normal(30, 15)

        q_score = np.random.uniform(0, 1)

        features = [
            yellow, h, s, v,
            (mq135 - 200) / 200,
            (mq136 - 50) / 50,
            (mq137 - 25) / 25,
            (mq138 - 30) / 30,
            q_score
        ]

        risk_index = yellow * 0.4 + abs(mq136 - 50) * 0.3 + q_score * 40
        p = 1 / (1 + np.exp(-(risk_index - 35) / 8))
        label = np.random.binomial(1, p)

        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

# ===============================
# MODEL LOADING
# ===============================

def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        train_synthetic_model()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ===============================
# FEATURE BUILDER
# ===============================

def build_feature_vector(image_path, sensor_csv, answers):
    img_feat = extract_image_features(image_path)
    sensor_feat = extract_sensor_features(sensor_csv)
    q_score = questionnaire_score(answers)

    return np.array(img_feat + sensor_feat + [q_score], dtype=np.float32)

# ===============================
# HARMOMED SAFE WRAPPER
# ===============================

def run_harmomed_safe(input_images, ref_image, out_path):
    result = HarmoMed_lir(input_images, ref_image, out_path)

    if not isinstance(result, str):
        raise TypeError("HarmoMed_lir must return image path")

    if not os.path.exists(result):
        raise FileNotFoundError(f"HarmoMed output not found: {result}")

    return result

# ===============================
# MAIN AI SCREENING
# ===============================

def run_ai_screening(image_path, sensor_csv, questionnaire_answers) -> AnalysisResult:
    x = build_feature_vector(image_path, sensor_csv, questionnaire_answers).reshape(1, -1)

    model, scaler = load_model_and_scaler()
    x_scaled = scaler.transform(x)

    prob = float(model.predict_proba(x_scaled)[0, 1])

    return AnalysisResult(
        image_features=x[0][:4].tolist(),
        sensor_features=x[0][4:8].tolist(),
        questionnaire_score=float(x[0][8]),
        risk_probability=round(prob, 3),
        risk_level=risk_label(prob)
    )

# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":

    processed_image = run_harmomed_safe(
        ["output.jpg"],
        "wtest2.jpg",
        "images/1.jpg"
    )

    result = run_ai_screening(
        processed_image,
        "sensor.csv",
        [3, 2, 4, 1, 3, 2, 4, 3, 2, 1]
    )

    print("Image features:", result.image_features)
    print("Sensor features:", result.sensor_features)
    print("Questionnaire score:", result.questionnaire_score)
    print("Risk probability:", result.risk_probability)
    print("Risk level:", result.risk_level)
