# matutor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="matutor-index"),
    # TTS endpoints removed
    path("api/image-to-text/", views.image_to_text, name="image-to-text"),
    path("api/solve-problem/", views.solve_problem, name="solve-problem"),
    path("api/generate-video/", views.generate_video, name="generate-video"),
    path('api/explain-problem/', views.explain_problem, name='explain_problem'),
    path('api/chat-simple/', views.chat_with_wolftor_simple, name='chat_simple'),
]
