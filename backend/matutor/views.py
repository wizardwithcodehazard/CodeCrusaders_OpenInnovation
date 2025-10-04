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


# matutor/views.py
import os
import google.generativeai as genai

from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from django.conf import settings


# Configure Gemini with API key from settings
genai.configure(api_key=settings.GEMINI_API_KEY)


@csrf_exempt
@require_POST
def image_to_text(request):
    """
    Endpoint: POST /api/image-to-text/
    Accepts: multipart/form-data with 'image' field
    Returns: JSON { "text": "..." }
    """
    if "image" not in request.FILES:
        return HttpResponseBadRequest("No image uploaded")

    image_file = request.FILES["image"]

    try:
        # Convert uploaded file into bytes
        image_bytes = image_file.read()

        # Load model (Gemini Pro Vision)
        model = genai.GenerativeModel("gemini-2.5-flash")

        # Call Gemini with image
        response = model.generate_content([
        {"mime_type": image_file.content_type, "data": image_bytes},
        "Extract ONLY the exact text from this image. Do not add descriptions, interpretations, or summaries. If no text is found, return an empty string."])


        # Extract text response
        text_output = response.text if response and hasattr(response, "text") else ""

        return JsonResponse({"text": text_output})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


import os
import json
import subprocess
import tempfile
import google.generativeai as genai

from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST



def build_gemini_prompt(problem_text: str) -> str:
    return f"""
You are an expert in mathematics, physics, electrical engineering, mechanical engineering, and symbolic computation. 
You have access to the Wolfram Engine through the Python WolframClient.

Your job:
1. Read the following problem:
---
{problem_text}
---

2. Decide what type of computation is needed (algebra, calculus, circuit analysis, physics formula, mechanics, etc.).

3. Write a **Python script** that:
   - Uses wolframclient.evaluation.WolframLanguageSession and wolframclient.language.wlexpr.
   - Defines the problem in Wolfram Language.
   - Evaluates the required symbolic or numeric computations.
   - Converts results to human-readable strings using ToString[InputForm[expr]].
   - Returns all results in a single Python dictionary called `results`.
   - Prints ONLY the JSON string of this dictionary using json.dumps(results).
   - Ensure session.terminate() is called at the end.

4. Constraints:
- Output ONLY Python code. 
- No explanations, no markdown.
"""
import sys

@csrf_exempt
@require_POST
def solve_problem(request):
    problem_text = request.POST.get("problem")
    if not problem_text:
        return HttpResponseBadRequest("Missing 'problem' field")

    try:
        # Step 1: Generate Python code from Gemini
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(build_gemini_prompt(problem_text))

        # Step 2: Extract Python code (strip markdown if present)
        code_text = response.text.strip()
        if code_text.startswith("```python"):
            code_text = code_text[len("```python"):].strip()
        if code_text.endswith("```"):
            code_text = code_text[:-3].strip()

        tmp_path = None
        try:
            # Step 3: Write to temporary file (Windows-safe)
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as tmp:
                tmp.write(code_text)
                tmp_path = tmp.name

            # Step 4: Run the temp script using venv Python
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="replace"
            )

        finally:
            # Always clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Step 5: Handle script errors
        if proc.returncode != 0:
            return JsonResponse({"error": proc.stderr.strip()}, status=500)

        # Step 6: Extract JSON safely
        stdout_clean = proc.stdout.strip()
        json_start = stdout_clean.find("{")
        json_end = stdout_clean.rfind("}") + 1
        if json_start == -1 or json_end == -1:
            return JsonResponse({"error": "No JSON output found", "raw": stdout_clean}, status=500)

        try:
            results = json.loads(stdout_clean[json_start:json_end])
        except json.JSONDecodeError as e:
            return JsonResponse({"error": f"Invalid JSON: {str(e)}", "raw": stdout_clean}, status=500)

        return JsonResponse({"results": results})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)