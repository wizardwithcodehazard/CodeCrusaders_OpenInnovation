# matutor/views.py
import os
import json
import logging

from django.conf import settings
from django.http import (
    JsonResponse,
    HttpResponseBadRequest,
    HttpResponseNotFound,
    FileResponse,
)
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .tts_service import TTSService

logger = logging.getLogger(__name__)

# Single shared TTS service instance (lazy-loads the TTS model on first use)
tts_service = TTSService()

# Maximum allowed characters in the text payload (protects from very long requests)
MAX_TEXT_LENGTH = 5000


def index(request):
    """Simple index/help endpoint."""
    return JsonResponse(
        {
            "message": "matutor TTS API — POST /api/tts/ with JSON {'text':'...'}",
            "notes": "Generated audio is served from MEDIA_URL (development only).",
        }
    )


@csrf_exempt  # remove or secure in production
@require_POST
def synthesize(request):
    """
    POST /api/tts/
    Body (JSON): {"text": "some text to synthesize"}
    Response: {"success": True, "filename": "tts_xxx.wav", "url": "http://.../media/tts_xxx.wav"}
    """
    # parse JSON body
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return HttpResponseBadRequest("Invalid JSON body")

    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return HttpResponseBadRequest("Field 'text' is required and must be non-empty")

    if len(text) > MAX_TEXT_LENGTH:
        return HttpResponseBadRequest(f"Text too long (max {MAX_TEXT_LENGTH} characters)")

    try:
        # Generate WAV file (returns absolute filesystem path)
        out_path = tts_service.generate(text)
        filename = os.path.basename(out_path)

        # Build absolute URL so clients (Postman/browser) can open it
        media_url = settings.MEDIA_URL if hasattr(settings, "MEDIA_URL") else "/media/"
        # request.build_absolute_uri handles host/port
        file_url = request.build_absolute_uri(os.path.join(media_url, filename))

        return JsonResponse({"success": True, "filename": filename, "url": file_url})
    except Exception as exc:
        logger.exception("TTS generation failed")
        return JsonResponse({"success": False, "error": str(exc)}, status=500)


def get_audio(request, filename):
    """
    Optional direct file endpoint: /media/<filename> (or configured via urls)
    Returns the generated WAV file, or 404 if missing.
    """
    media_root = getattr(settings, "MEDIA_ROOT", os.path.join(os.path.dirname(os.path.dirname(__file__)), "media"))
    path = os.path.join(media_root, filename)

    if not os.path.exists(path):
        return HttpResponseNotFound("File not found")

    return FileResponse(open(path, "rb"), content_type="audio/wav")
