# matutor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="matutor-index"),
    path("api/tts/", views.synthesize, name="tts-synthesize"),
    path("media/<str:filename>", views.get_audio, name="tts-get-audio"),  # optional: direct app route
    path("api/image-to-text/", views.image_to_text, name="image-to-text"),
    path("api/solve-problem/", views.solve_problem, name="solve-problem"),
]
