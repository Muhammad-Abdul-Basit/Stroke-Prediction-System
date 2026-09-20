import os
import joblib
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_model_SVM.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_SVM.pkl")


try:
    model = joblib.load(MODEL_PATH)

    print("MODEL LOADED SUCCESSFULLY")
    print("Model:", type(model))
    print("Model path:", MODEL_PATH)

    logger.info(f"Model loaded from {MODEL_PATH}")

except Exception as e:

    print("MODEL LOAD FAILED:", e)

    logger.warning(f"Model could not be loaded: {e}")
    model = None


try:
    scaler = joblib.load(SCALER_PATH)

    print("SCALER LOADED SUCCESSFULLY")
    print("Scaler:", type(scaler))
    print("Scaler path:", SCALER_PATH)

    logger.info(f"Scaler loaded from {SCALER_PATH}")

except Exception as e:

    print("SCALER LOAD FAILED:", e)

    logger.warning(f"Scaler could not be loaded: {e}")
    scaler = None


def predict_stroke(data):

    print("\n========== ML PREDICTION STARTED ==========")

    if model is None:
        raise RuntimeError("ML model is not loaded.")

    if scaler is None:
        raise RuntimeError("Scaler is not loaded.")

    print("Input data:")
    print(data)

    scaled_data = scaler.transform(data)

    print("Scaling completed.")
    print("Scaled data:")
    print(scaled_data)

    prediction = model.predict(scaled_data)

    print("Model prediction:", prediction)

    result = int(prediction)

    print("Final prediction:", result)
    print("========== ML PREDICTION FINISHED ==========\n")

    return result