from django.urls import path
from . import views


urlpatterns = [
    path("", views.stroke_prediction, name="stroke_prediction"),
]