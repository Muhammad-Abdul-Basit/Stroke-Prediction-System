import pandas as pd


# --------------------------------------------------
# Min-Max values from the training dataset
# --------------------------------------------------

AGE_MIN = 27.70
AGE_MAX = 99.49

BMI_MIN = 15.03
BMI_MAX = 47.49

GLUCOSE_MIN = 45.56
GLUCOSE_MAX = 176.180


def preprocess_data(form_data):

    # ----------------------------------------------
    # Get raw values from the Django form
    # ----------------------------------------------

    age = float(form_data["age"])
    hypertension = int(form_data["hypertension"])
    heart_disease = int(form_data["heart_disease"])
    avg_glucose = float(form_data["avg_glucose"])
    bmi = float(form_data["bmi"])

    gender_map = int(form_data["gender_map"])
    ses_map = int(form_data["ses_map"])
    smoking_status_map = int(
        form_data["smoking_status_map"]
    )


    # ----------------------------------------------
    # Min-Max normalization
    # ----------------------------------------------

    age = (
        age - AGE_MIN
    ) / (
        AGE_MAX - AGE_MIN
    )

    bmi = (
        bmi - BMI_MIN
    ) / (
        BMI_MAX - BMI_MIN
    )

    avg_glucose = (
        avg_glucose - GLUCOSE_MIN
    ) / (
        GLUCOSE_MAX - GLUCOSE_MIN
    )


    # ----------------------------------------------
    # Create DataFrame
    # ----------------------------------------------

    data = pd.DataFrame([{
        "Age": age,
        "Hypertension": hypertension,
        "Heart_Disease": heart_disease,
        "Avg_Glucose": avg_glucose,
        "BMI": bmi,
        "gender_map": gender_map,
        "SES_map": ses_map,
        "Smoking_Status_map": smoking_status_map
    }])


    # ----------------------------------------------
    # Return data
    # ----------------------------------------------

    return data