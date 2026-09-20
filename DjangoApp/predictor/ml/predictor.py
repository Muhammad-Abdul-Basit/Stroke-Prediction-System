import os
import joblib
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_model_SVM.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_SVM.pkl")


try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.warning(f"Model could not be loaded: {e}")
    model = None


try:
    scaler = joblib.load(SCALER_PATH)
    logger.info(f"Scaler loaded from {SCALER_PATH}")
except Exception as e:
    logger.warning(f"Scaler could not be loaded: {e}")
    scaler = None


def predict_stroke(data):

    if model is None:
        raise RuntimeError("ML model is not loaded.")

    if scaler is None:
        raise RuntimeError("Scaler is not loaded.")

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)

    return int(prediction[0])