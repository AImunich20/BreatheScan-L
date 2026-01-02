import cv2
from datetime import datetime
import os
import joblib
import numpy as np
import pandas as pd
import logging
import json
from typing import List, Dict
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from HarmoMed import HarmoMed_lir

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ================= CONSTANTS =================
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

# ================= QUESTIONNAIRE CONFIG =================
QS_COLUMNS = [
    "alcohol",
    "fat_food",
    "sulfur_food",
    "fructose",
    "jaundice",
    "carotene_food",
    "breath_odor",
    "abdominal_pain",
    "fatigue",
    "diabetes"
]

QS_MAP = {
    "no": 0, "never": 0, "none": 0, "low": 0,
    "sometimes": 2, "medium": 2,
    "yes": 4, "often": 4, "high": 4, "daily": 4,
    0: 0, 1: 4, 2: 2, 3: 3, 4: 4
}

# ================= DATA CLASS =================
@dataclass
class AnalysisResult:
    image_features: List[float]
    sensor_features: List[float]
    questionnaire_score: float
    risk_probability: float
    risk_level: str
    model_version: str
    risk_breakdown: Dict[str, float]

# ================= IMAGE FEATURES =================
def extract_image_features(image_path: str) -> List[float]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Resize to a controlled size for stable stats (maintain aspect ratio)
    h0, w0 = img.shape[:2]
    max_dim = 1024
    if max(h0, w0) > max_dim:
        scale = max_dim / max(h0, w0)
        img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    # Enhance local contrast: CLAHE on the V channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Skin detection via YCrCb color space (more robust than simple thresholding)
    ycrcb = cv2.cvtColor(img_eq, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    # Empirical skin thresholds (tuned for varied illumination)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    # Convert to HSV and LAB for combined yellow detection
    hsv = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)
    lab2 = cv2.cvtColor(img_eq, cv2.COLOR_BGR2LAB)

    # HSV yellow range (tuned narrower for accuracy)
    hsv_lower = np.array([18, 60, 60])
    hsv_upper = np.array([35, 255, 255])
    mask_hsv = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # LAB: yellow often has high 'b' channel
    l2, a2, b2 = cv2.split(lab2)
    # Normalize and threshold b channel adaptively
    b2_norm = cv2.normalize(b2, None, 0, 255, cv2.NORM_MINMAX)
    _, mask_lab = cv2.threshold(b2_norm, 150, 255, cv2.THRESH_BINARY)

    # Combine masks and restrict to skin region (reduce false positives)
    yellow_mask = cv2.bitwise_and(mask_hsv, mask_lab)
    yellow_mask = cv2.bitwise_and(yellow_mask, skin_mask)

    # Morphological cleanup and connected components filtering
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel2)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel2)

    # Connected components: keep sufficiently large regions to avoid noise
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(yellow_mask, connectivity=8)
    final_mask = np.zeros_like(yellow_mask)
    min_area = max(50, (img.shape[0] * img.shape[1]) // 2000)  # adaptive min area
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            final_mask[labels == i] = 255

    # Compute stable statistics: percentage area and robust means (median)
    yellow_area = np.count_nonzero(final_mask)
    total_skin_area = max(1, np.count_nonzero(skin_mask))
    yellow_ratio = yellow_area / total_skin_area

    # Compute robust color statistics (median) over final_mask in HSV
    h, s, v = cv2.split(hsv)
    masked_h = h[final_mask > 0]
    masked_s = s[final_mask > 0]
    masked_v = v[final_mask > 0]

    if masked_h.size > 0:
        h_med = float(np.median(masked_h))
        s_med = float(np.median(masked_s))
        v_med = float(np.median(masked_v))
    else:
        h_med = s_med = v_med = 0.0

    # Return percentages and medians (suitable for downstream scaling)
    return [
        round(yellow_ratio * 100, 4),
        round(h_med, 3),
        round(s_med, 3),
        round(v_med, 3)
    ]

# ================= SENSOR FEATURES =================
def extract_sensor_features(csv_path: str) -> List[float]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    features = []

    for sensor, ref in STANDARD_SENSOR.items():
        if sensor not in df.columns:
            raise ValueError(f"Missing sensor: {sensor}")

        mean_val = df[sensor].mean()
        z = (mean_val - ref) / (ref + EPS)
        features.append(round(z, 3))

    return features

# ================= QUESTIONNAIRE =================
def load_questionnaire_answers(csv_path: str) -> List[int]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df = df[QS_COLUMNS]
    row = df.iloc[0]

    answers = []
    for v in row:
        if pd.isna(v):
            answers.append(0)
        else:
            answers.append(int(QS_MAP.get(str(v).lower(), 0)))

    if len(answers) != 10:
        raise ValueError(
            f"Questionnaire must have 10 answers, got {len(answers)}"
        )

    return answers

def questionnaire_score(answers: List[int]) -> float:
    if len(answers) != 10:
        raise ValueError("Questionnaire must have 10 answers")

    if any(a < 0 or a > 4 for a in answers):
        raise ValueError("Answer must be between 0–4")

    return round(sum(answers) / 40.0, 3)

# ================= MODEL =================
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
    return {
        "image_risk": round(np.mean(x[:4]) / 100, 3),
        "sensor_risk": round(np.mean(np.abs(x[4:8])), 3),
        "questionnaire_risk": round(float(x[8]), 3)
    }

def train_synthetic_model(n_samples=1000, seed=42):
    np.random.seed(seed)
    X, y = [], []

    for _ in range(n_samples):
        yellow = np.clip(np.random.normal(15, 10), 0, 60)
        h, s, v = np.random.uniform(15, 35), np.random.uniform(80, 255), np.random.uniform(80, 255)

        mq = {
            "MQ_135": np.random.normal(200, 60),
            "MQ_136": np.random.normal(50, 20),
            "MQ_137": np.random.normal(25, 10),
            "MQ_138": np.random.normal(30, 15)
        }

        q = np.random.uniform(0, 1)

        features = [
            yellow, h, s, v,
            (mq["MQ_135"] - 200) / 200,
            (mq["MQ_136"] - 50) / 50,
            (mq["MQ_137"] - 25) / 25,
            (mq["MQ_138"] - 30) / 30,
            q
        ]

        risk_index = yellow * 0.4 + abs(mq["MQ_136"] - 50) * 0.3 + q * 40
        p = 1 / (1 + np.exp(-(risk_index - 35) / 8))
        label = np.random.binomial(1, p)

        X.append(features)
        y.append(label)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(np.array(X))

    model = LogisticRegression(max_iter=3000)
    model.fit(Xs, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    logger.info("Synthetic model trained")


def retrain_model(n_samples=1000, seed=42):
    """Public retrain wrapper. Trains and re-saves model and scaler.

    Returns path to saved model and scaler.
    """
    train_synthetic_model(n_samples=n_samples, seed=seed)
    return MODEL_PATH, SCALER_PATH

def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        train_synthetic_model()
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

# ================= PIPELINE =================
def build_feature_vector(image_path, sensor_csv, answers):
    img = extract_image_features(image_path)
    sensor = extract_sensor_features(sensor_csv)
    q = questionnaire_score(answers)

    x = np.array(img + sensor + [q], dtype=np.float32)

    if x.shape[0] != EXPECTED_FEATURE_DIM:
        raise ValueError("Feature dimension mismatch")

    return x

def run_harmomed_safe(input_images, ref_image, out_path):
    result = HarmoMed_lir(input_images, ref_image, out_path)
    # expect dict with final_image and per_image
    if not isinstance(result, dict) or "final_image" not in result:
        raise RuntimeError("HarmoMed failed or returned unexpected structure")
    if not os.path.exists(result["final_image"]):
        raise RuntimeError("HarmoMed final image missing")
    return result

def run_ai_screening(image_path, sensor_csv, questionnaire_answers):
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

def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def Breathescan_L(input_img, path_img, sensor, ans, result_dir):
    os.makedirs(result_dir, exist_ok=True)

    # 1) Image preprocessing (HarmoMed)
    # Normalize input_img: allow single path or list
    if isinstance(input_img, (str, os.PathLike)):
        input_images = [str(input_img)]
    else:
        # ensure iterable -> list of str
        input_images = [str(p) for p in input_img]

    harmo_result = run_harmomed_safe(
        input_images,
        "wtest2.jpg",
        path_img
    )
    processed_image = harmo_result.get("final_image")

    per_image_info = harmo_result.get("per_image", [])

    # 2) Load & convert questionnaire CSV -> List[int] (10 answers)
    questionnaire_answers = load_questionnaire_answers(ans)

    # 3) Run AI screening
    result = run_ai_screening(
        processed_image,
        sensor,
        questionnaire_answers
    )

    # 4) Save JSON result (numpy-safe)
    # include additional metadata
    output_payload = to_json_safe(result.__dict__)
    output_payload["processed_image"] = processed_image
    output_payload["per_image"] = per_image_info
    output_payload["feature_vector"] = build_feature_vector(processed_image, sensor, questionnaire_answers).tolist()
    output_payload["timestamp"] = datetime.now().isoformat()

    # Build color_summary aggregation
    color_summary = {
        "per_image_colors": [],
        "aggregated": {}
    }

    b_list = []
    c_list = []
    color_diff_means = []
    for p in per_image_info:
        meta = p.get("processing_meta") or {}
        per = {
            "input": p.get("input"),
            "filename": p.get("filename"),
            "ref_avg_color": meta.get("ref_avg_color"),
            "corrected_avg_color": meta.get("corrected_avg_color"),
            "warm_corrected_avg_color": meta.get("warm_corrected_avg_color"),
            "color_diff_result_mean": meta.get("color_diff_result_mean")
        }
        color_summary["per_image_colors"].append(per)

        if meta.get("total_brightness_change") is not None:
            b_list.append(meta.get("total_brightness_change"))
        if meta.get("total_contrast_change") is not None:
            c_list.append(meta.get("total_contrast_change"))
        if meta.get("color_diff_result_mean") is not None:
            color_diff_means.append(meta.get("color_diff_result_mean"))

    # aggregated stats
    def mean_or_none(x):
        return None if not x else float(np.mean(x))

    color_summary["aggregated"] = {
        "mean_total_brightness_change": mean_or_none(b_list),
        "mean_total_contrast_change": mean_or_none(c_list),
        "mean_color_diff_per_channel": None
    }

    if color_diff_means:
        arr = np.array(color_diff_means, dtype=float)
        # mean across images -> per-channel mean
        color_summary["aggregated"]["mean_color_diff_per_channel"] = list(np.mean(arr, axis=0).astype(float))

    output_payload["color_summary"] = color_summary

    result_path = os.path.join(result_dir, "analysis_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=4)

    return result
