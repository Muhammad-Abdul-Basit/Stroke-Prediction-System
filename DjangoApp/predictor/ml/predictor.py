from django.conf import settings
import os
import joblib
import numpy as np
import logging

logger=logging.getLogger(__name__)

#Getting path for each model of ML
MODEL_PATH =os.path.join(settings.BASE_DIR, "best_model_SVM.pkl")
SCALER_PATH=os.path.join(settings.BASE_DIR, "scaler_SVM.pkl")

#Loading model file of ML
try:
    model=joblib.load(MODEL_PATH)
    logger.info(f"Model file is loading from {MODEL_PATH}")
except Exception as e:
    logger.warning(f"Model file is not loading {e}")
    model=None

#Loading scaler file of ML
try:
    scaler=joblib.load(SCALER_PATH)
    logger.info(f"Scaler file is loading from {SCALER_PATH}")
except Exception as e:
    scaler=None
    logger.warning(f"Scaler file is not loading {e}")