from django.shortcuts import render

from .preprocessing import preprocess_data
from .ml.predictor import predict_stroke


def stroke_prediction(request):

    if request.method == "POST":
        try:

            # These are only for the report
            user_name = request.POST.get("user_name")
            cnic = request.POST.get("cnic")

            # Only ML-related fields go into preprocessing
            form_data = {
                "age": request.POST.get("age"),
                "hypertension": request.POST.get("hypertension"),
                "heart_disease": request.POST.get("heart_disease"),
                "avg_glucose": request.POST.get("avg_glucose"),
                "bmi": request.POST.get("bmi"),
                "gender_map": request.POST.get("gender_map"),
                "ses_map": request.POST.get("ses_map"),
                "smoking_status_map": request.POST.get("smoking_status_map"),
            }

            processed_data = preprocess_data(form_data)

            prediction = predict_stroke(processed_data)

            return render(
                request,
                "report.html",
                {
                    "user_name": user_name,
                    "cnic": cnic,
                    "prediction": prediction
                }
            )

        except Exception as e:
            return render(
                request,
                "form.html",
                {
                    "error": str(e)
                }
            )

    return render(request, "form.html")