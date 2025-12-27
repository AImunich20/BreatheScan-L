"""
AI Multimodal Risk Screening System
----------------------------------
Image + Sensor (E-nose) + Questionnaire
Designed for medical AI prototype (BreatheScan-L ready)

Author: Mint 🤍
Version: 2.0.0
"""

# =========================
# Imports
# =========================
import cv2
import os
import joblib
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from HarmoMed import HarmoMed_lir

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_VERSION = "2.0.0"
MODEL_PATH = "risk_model.pkl"
SCALER_PATH = "scaler.pkl"

EXPECTED_FEATURE_DIM = 9
EPS = 1e-6

YELLOW_HSV_LOWER = np.array([18, 80, 80])
YELLOW_HSV_UPPER = np.array([35, 255, 255])

STANDARD_SENSOR = {
    "MQ_135": 200,
    "MQ_136": 50,
    "MQ_137": 25,
    "MQ_138": 30
}

FEATURE_NAMES = [
    "yellow_ratio",
    "h_mean",
    "s_mean",
    "v_mean",
    "MQ135_z",
    "MQ136_z",
    "MQ137_z",
    "MQ138_z",
    "questionnaire"
]

@dataclass
class AnalysisResult:
    image_features: List[float]
    sensor_features: List[float]
    questionnaire_score: float
    risk_probability: float
    risk_level: str
    model_version: str
    risk_breakdown: Dict[str, float]

def extract_image_features(image_path: str) -> List[float]:
    """
    Extract jaundice-related features from image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)

    yellow_ratio = np.count_nonzero(mask) / (mask.size + EPS)

    if np.any(mask):
        h_mean = np.mean(h[mask > 0])
        s_mean = np.mean(s[mask > 0])
        v_mean = np.mean(v[mask > 0])
    else:
        h_mean = s_mean = v_mean = 0.0

    return [
        round(yellow_ratio * 100, 3),
        round(h_mean, 3),
        round(s_mean, 3),
        round(v_mean, 3)
    ]

def extract_sensor_features(csv_path: str) -> List[float]:
    """
    Z-normalize sensor readings relative to healthy baseline
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    features = []

    for sensor, ref in STANDARD_SENSOR.items():
        if sensor not in df.columns:
            raise ValueError(f"Missing sensor: {sensor}")

        mean_val = df[sensor].mean()
        z_score = (mean_val - ref) / (ref + EPS)
        features.append(round(z_score, 3))

    return features

def questionnaire_score(answers: List[int]) -> float:
    """
    10 questions, each scored 0–4
    Output normalized 0–1
    """
    if len(answers) != 10:
        raise ValueError("Questionnaire must have 10 answers")

    if any(a < 0 or a > 4 for a in answers):
        raise ValueError("Answer must be between 0–4")

    return round(sum(answers) / 40.0, 3)

def risk_label(prob: float) -> str:
    if prob < 0.30:
        return "Low risk"
    elif prob < 0.60:
        return "Moderate risk"
    elif prob < 0.80:
        return "High risk"
    else:
        return "Very high risk"

def risk_breakdown(x: np.ndarray) -> Dict[str, float]:
    """
    Explainability (simple heuristic)
    """
    return {
        "image_risk": round(np.mean(x[:4]) / 100, 3),
        "sensor_risk": round(np.mean(np.abs(x[4:8])), 3),
        "questionnaire_risk": round(float(x[8]), 3)
    }

def train_synthetic_model(
    n_samples: int = 1000,
    seed: int = 42
) -> None:
    """
    Synthetic training for prototype only
    """
    np.random.seed(seed)
    X, y = [], []

    for _ in range(n_samples):
        yellow = np.clip(np.random.normal(15, 10), 0, 60)
        h = np.random.uniform(15, 35)
        s = np.random.uniform(80, 255)
        v = np.random.uniform(80, 255)

        mq = {
            "MQ_135": np.random.normal(200, 60),
            "MQ_136": np.random.normal(50, 20),
            "MQ_137": np.random.normal(25, 10),
            "MQ_138": np.random.normal(30, 15)
        }

        q_score = np.random.uniform(0, 1)

        features = [
            yellow, h, s, v,
            (mq["MQ_135"] - 200) / 200,
            (mq["MQ_136"] - 50) / 50,
            (mq["MQ_137"] - 25) / 25,
            (mq["MQ_138"] - 30) / 30,
            q_score
        ]

        risk_index = yellow * 0.4 + abs(mq["MQ_136"] - 50) * 0.3 + q_score * 40
        p = 1 / (1 + np.exp(-(risk_index - 35) / 8))
        label = np.random.binomial(1, p)

        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=3000)
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    logger.info("Model trained and saved successfully")

def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        logger.warning("Model not found — training synthetic model")
        train_synthetic_model()

    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

def build_feature_vector(
    image_path: str,
    sensor_csv: str,
    answers: List[int]
) -> np.ndarray:

    img_feat = extract_image_features(image_path)
    sensor_feat = extract_sensor_features(sensor_csv)
    q_score = questionnaire_score(answers)

    x = np.array(img_feat + sensor_feat + [q_score], dtype=np.float32)

    if x.shape[0] != EXPECTED_FEATURE_DIM:
        raise ValueError("Feature dimension mismatch")

    if not np.isfinite(x).all():
        raise ValueError("NaN or Inf in features")

    return x

def run_harmomed_safe(
    input_images: List[str],
    ref_image: str,
    out_path: str
) -> str:

    result = HarmoMed_lir(input_images, ref_image, out_path)

    if not isinstance(result, str):
        raise TypeError("HarmoMed must return output image path")

    if not os.path.exists(result):
        raise FileNotFoundError(result)

    return result

def run_ai_screening(
    image_path: str,
    sensor_csv: str,
    questionnaire_answers: List[int]
) -> AnalysisResult:

    x = build_feature_vector(image_path, sensor_csv, questionnaire_answers)
    model, scaler = load_model_and_scaler()

    x_scaled = scaler.transform(x.reshape(1, -1))
    prob = float(model.predict_proba(x_scaled)[0, 1])

    return AnalysisResult(
        image_features=x[:4].tolist(),
        sensor_features=x[4:8].tolist(),
        questionnaire_score=float(x[8]),
        risk_probability=round(prob, 3),
        risk_level=risk_label(prob),
        model_version=MODEL_VERSION,
        risk_breakdown=risk_breakdown(x)
    )

# if __name__ == "__main__":
#     processed_image = run_harmomed_safe(
#         ["output.jpg"],
#         "wtest2.jpg",
#         "images/1.jpg"
#     )

#     result = run_ai_screening(
#         processed_image,
#         "sensor.csv",
#         [3, 2, 4, 1, 3, 2, 4, 3, 2, 1]
#     )

#     print(result)

def Breathescan_L(input_img,path_img,sensor,ans):
    processed_image = run_harmomed_safe(
        [input_img],
        "wtest2.jpg",
        path_img
    )
    result = run_ai_screening(
        processed_image,
        sensor,
        ans
    )
    print(result)