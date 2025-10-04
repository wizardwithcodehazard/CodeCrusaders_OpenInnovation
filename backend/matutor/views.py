# matutor/views.py
import os
import json
import logging
import shutil
import time
import re
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
You are a Python programming expert specializing in physics and mathematics problems.

CRITICAL INSTRUCTIONS:
- Generate ONLY pure Python code
- DO NOT use Wolfram Language, Mathematica, or any symbolic math that isn't Python
- Use standard Python libraries: math, numpy, json
- The script MUST output results as valid JSON to stdout
- Use json.dumps() to print the final results

Problem to solve:
{problem_text}

Requirements:
1. Write a complete Python script that solves this problem
2. Calculate all requested values using Python math/numpy
3. At the end, print results as JSON using: print(json.dumps(results))
4. The JSON should contain all calculated values with descriptive keys
5. Use float values for all numbers

Example output format:
{{
    "equivalent_resistance": 2.73,
    "total_current": 10.99,
    "individual_currents": {{
        "R1_5ohm": 6.0,
        "R2_10ohm": 3.0,
        "R3_15ohm": 2.0
    }},
    "power_dissipated": {{
        "R1_5ohm": 180.0,
        "R2_10ohm": 90.0,
        "R3_15ohm": 60.0
    }}
}}

Generate the Python code now:
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
    

import os
import sys
import json
import tempfile
import subprocess
from django.http import JsonResponse, HttpResponseBadRequest, FileResponse
from django.conf import settings
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MEDIA_DIR = os.path.join(settings.BASE_DIR, "media", "videos")
os.makedirs(MEDIA_DIR, exist_ok=True)

# Prompt template for Manim video generation
def build_manim_prompt(problem_text: str, results_json: dict) -> str:
    results_str = json.dumps(results_json, indent=2)
    return f"""
You are a Manim animation expert and high school astronomy/mathematics educator. Your task is to take any input problem (a high school level astronomy or math calculation question) and directly generate a fully working Manim Community Edition v0.18+ Python script that can be copy-pasted and rendered to produce an educational video.

⚡ Key Instructions:

1. Output Format: Always reply ONLY with the Manim Python script in plain text (no explanations, no markdown fences, no extra commentary).
2. The user will provide the problem statement, and you must convert it directly into a complete Manim script.

## VIDEO CONFIGURATION

Always use these settings:

```python
from manim import *
import numpy as np

# Video quality settings - 480p at 15fps
config.tex_compiler = "pdflatex"
config.pixel_width = 854
config.pixel_height = 480
config.frame_rate = 15
```

## SCREEN LAYOUT REQUIREMENTS

Divide screen into 3 distinct sections:

- TOP SECTION (0.8 to 1.0 of screen height): Main titles and headers only
- MIDDLE SECTION (0.2 to 0.8 of screen height): Main content, calculations, diagrams, processes
- BOTTOM SECTION (0.0 to 0.2 of screen height): Subtitles and explanatory text only

Proper positioning:

```python
# Top section positioning
title.to_edge(UP, buff=0.1)

# Middle section positioning
content.move_to(ORIGIN).shift(UP * 0.1)

# Bottom section positioning
subtitle.to_edge(DOWN, buff=0.1)
```

## ANIMATION CLEANUP RULES

CRITICAL: Always clean up before new content:

```python
# Before showing new content, ALWAYS remove old elements:
self.play(FadeOut(old_title, old_content, old_subtitle))
self.remove(old_title, old_content, old_subtitle)  # Ensure complete removal

# Then add new content
self.play(FadeIn(new_title, new_content, new_subtitle))
```

Sequential content management:
- Each frame should only show relevant content
- Remove ALL previous elements before introducing new ones
- Use FadeOut() followed by self.remove() for complete cleanup
- Never let elements overlap or accumulate

## API COMPATIBILITY RULES

Use current Manim Community syntax:
- Use self.add() instead of self.add_fixed_in_frame_mobjects()
- Use self.remove() instead of self.remove_fixed_in_frame_mobjects()
- Import from manim import * (not from manimlib import *)
- Use Scene class (not ThreeDScene unless specifically needed for 3D)

## SCENE STRUCTURE TEMPLATE

Follow this structure:

```python
class AstronomyProblem(Scene):
    def construct(self):
        # Set background
        self.camera.background_color = BLACK
        
        # Section 1: Title
        title = Text("Title", font_size=36).to_edge(UP, buff=0.1)
        self.play(Write(title))
        
        # Section 2: Main content
        content = MathTex("formula").move_to(ORIGIN)
        self.play(Write(content))
        
        # Section 3: Subtitle
        subtitle = Text("explanation").to_edge(DOWN, buff=0.1)
        self.add(subtitle)
        self.play(FadeIn(subtitle))
        
        # Cleanup before next section
        self.play(FadeOut(title, content, subtitle))
        self.remove(title, content, subtitle)
        
        # Next section...
```

## HELPER FUNCTIONS

Subtitle function:

```python
def create_subtitle(text_str, position=DOWN*3.2):
    \"\"\"Create subtitle text at bottom of the screen\"\"\"
    return Text(
        text_str,
        font_size=24,
        color=WHITE
    ).move_to(position).add_background_rectangle(
        color=BLACK,
        opacity=0.8,
        buff=0.2
    )
```

3D Surface function (if needed):

```python
def param_surface_from_func(f, u_range, v_range, resolution=30, color=BLUE_D, opacity=0.9):
    \"\"\"Return a 3D surface defined by z = f(x,y).\"\"\"
    return Surface(
        lambda u, v: np.array([u, v, f(u, v)]),
        u_range=u_range,
        v_range=v_range,
        checkerboard_colors=[BLUE_D, BLUE_E],
        resolution=(resolution, resolution),
        fill_opacity=opacity,
        stroke_width=0.5,
        stroke_color=WHITE,
    ).set_color(color)
```

## TEXT AND LAYOUT BEST PRACTICES

Proper text sizing for 480p:

```python
# Titles (top section)
title = Text("Main Title", font_size=36, color=YELLOW).to_edge(UP, buff=0.1)

# Main content (middle section)
content = MathTex(r"\\theta = 2 \\arctan\\left(\\frac{{D}}{{2d}}\\right)", font_size=32, color=TEAL).move_to(ORIGIN)

# Subtitles (bottom section)
subtitle = Text("Explanation text", font_size=24, color=WHITE).to_edge(DOWN, buff=0.1)
```

Avoid overlapping:
- Use .next_to() with proper buff values
- Use .align_to() for consistent alignment
- Use VGroup() to group related elements
- Test positioning with different screen sizes
- For bigger texts, don't make them go over the screen width - split text into multiple lines

## ANIMATION BUILDING BLOCKS

Math Expressions:

```python
equation = MathTex(r"E = mc^2", font_size=32).move_to(ORIGIN)
self.play(Write(equation))

# To highlight results:
box = SurroundingRectangle(equation, buff=0.15, color=YELLOW, stroke_width=3)
self.play(Create(box))
```

Showing Steps with Transform:

```python
eq1 = MathTex(r"E = mc^2")
eq2 = MathTex(r"E = (2)(3)^2")
self.play(Write(eq1))
self.wait(1)
self.play(Transform(eq1, eq2))
```

Shapes & Geometry:

```python
circle = Circle(radius=2, color=TEAL)
line = Line(LEFT, RIGHT, color=YELLOW)
arc = Arc(radius=1, start_angle=0, angle=PI/3, color=ORANGE)
dot = Dot(color=RED)
```

3D Objects (when needed):

```python
# Use Sphere() with proper resolution
sphere = Sphere(radius=0.5, resolution=(20, 20), color=BLUE_E)
axes3d = ThreeDAxes()
self.play(Create(axes3d), Create(sphere))
```

## SCENE FLOW / BEST PRACTICES

The script should follow this sequence:

1. Intro Title & Subtitle (problem statement)
2. Display Givens (MathTex list of knowns)
3. Show Formula (highlighted, maybe boxed)
4. Step-by-Step Substitution (use Transform to morph equations)
5. Final Result (boxed, large font, color highlight)
6. Optional Visualization (Geometry: circles, arcs, lines; Astronomy: planets, stars, orbits)
7. Summary Slide (Text + results)

## VALID ANIMATIONS TO USE

- Write()
- FadeIn()
- FadeOut()
- Transform()
- ReplacementTransform()
- Create() (for lines/shapes)
- GrowFromCenter()
- GrowArrow()
- self.wait(time) for pacing
- SurroundingRectangle() for highlights

## TEXT AND MATHTEX FORMATTING

- Use raw strings for LaTeX: r"\\theta = 2 \\arctan\\left(\\frac{{D}}{{2d}}\\right)"
- Escape backslashes properly in f-strings: f"{{value:.3f}}^\\\\circ"
- Use MathTex for mathematical expressions
- Use Text for regular text

## TIMING/PACING

- Animations 0.5–2s each
- Use self.wait(0.5–1) sparingly
- Total runtime optimized for 5–8 minutes at 1x

## CRITICAL: MATH TEX STRING FORMATTING RULES

NEVER use .format() with MathTex when mixing LaTeX and variables:

❌ WRONG - This causes KeyError:
```python
step3 = MathTex(r"\\theta = 2 \\arctan\\left({{:f}}\\right) \\quad (\\text{{radians}})".format(val_inside_arctan))
```

✅ CORRECT - Use string concatenation:
```python
step3 = MathTex(r"\\theta = 2 \\arctan\\left(" + f"{{val_inside_arctan:.6f}}" + r"\\right) \\quad (\\text{{radians}})")
```

### PROPER MATH TEX FORMATTING PATTERNS

For single variables:
```python
# Correct
result = MathTex(r"\\theta \\approx " + f"{{theta_deg:.2f}}" + r"^\\circ")

# Wrong
result = MathTex(r"\\theta \\approx {{:.2f}}^\\circ".format(theta_deg))
```

For multiple variables:
```python
# Correct
equation = MathTex(r"E = " + f"{{mass:.1f}}" + r" \\times " + f"{{speed:.0f}}" + r"^2")
```

### WHY THIS MATTERS

- LaTeX uses {{}} for grouping, which conflicts with Python's .format() syntax
- String concatenation with f-strings is the safest approach
- This prevents KeyError exceptions during rendering

## ERROR PREVENTION CHECKLIST

Before using MathTex with variables:
1. ✅ Use string concatenation instead of .format()
2. ✅ Use f-strings for number formatting
3. ✅ Keep LaTeX and Python formatting separate

Common parameter issues to avoid:
- ❌ stroke_dash_offset=0.2 parameter doesn't exist - remove it
- ❌ align_to=LEFT in next_to() - use aligned_edge=LEFT instead

## CRITICAL MISTAKES TO AVOID

NEVER:
- Let text overlap between sections
- Skip cleanup between scenes
- Use deprecated API methods (add_fixed_in_frame_mobjects, remove_fixed_in_frame_mobjects)
- Mix old and new content without proper removal
- Use incorrect video dimensions (always 480p, 15fps)
- Use deprecated syntax like TexText, ShowCreation
- Leave undefined variables (all colors, positions, constants must be declared)
- Use .format() with MathTex containing LaTeX

ALWAYS:
- Clean up completely before new content
- Use proper section positioning
- Test layout on 480p resolution
- Use appropriate font sizes for 480p
- Remove elements with both FadeOut() and self.remove()
- Use current Manim Community API
- Include proper error handling
- Use consistent naming conventions
- Use string concatenation with f-strings for MathTex

## OUTPUT REQUIREMENTS

Always provide:
1. Complete, runnable code with 480p/15fps settings
2. Proper 3-section layout implementation
3. Complete cleanup between scenes
4. Current API methods only
5. No overlapping text elements
6. Clear comments for each section
7. Equations that compile with LaTeX
8. No undefined variables or syntax errors
9. Final script must run standalone with Manim CE

## SPECIAL NOTE

If the user is asking a general question or wants to learn about something theoretical, explain the concept with real-life examples. Keep the language simple for the user to understand effectively.

---

Remember: Clean up completely before each new scene, maintain 3-section layout, always use 480p/15fps configuration, and use string concatenation for MathTex with variables.

Problem Text:
{problem_text}

Results JSON:
{results_str}
"""
@csrf_exempt
@require_POST
def generate_video(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST allowed")

    # Try to parse JSON body first, then fall back to form data
    problem_text = None
    
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body.decode('utf-8'))
            problem_text = data.get("problem")
        else:
            problem_text = request.POST.get("problem")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return HttpResponseBadRequest(f"Invalid request format: {str(e)}")

    if not problem_text:
        return HttpResponseBadRequest("Missing 'problem' field")

    tmp_solver_path = None
    tmp_manim_path = None
    
    try:
        # ===== STEP 1: SOLVE THE PROBLEM (like solve_problem) =====
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(build_gemini_prompt(problem_text))

        # Extract Python code (strip markdown if present)
        code_text = response.text.strip()
        if code_text.startswith("```python"):
            code_text = code_text[len("```python"):].strip()
        if code_text.endswith("```"):
            code_text = code_text[:-3].strip()

        # Write solver code to temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as tmp:
            tmp.write(code_text)
            tmp_solver_path = tmp.name

        # Run the solver script
        proc = subprocess.run(
            [sys.executable, tmp_solver_path],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace"
        )

        # Handle solver errors
        if proc.returncode != 0:
            return JsonResponse({"error": "Solver failed", "details": proc.stderr.strip()}, status=500)

        # Extract JSON from solver output
        stdout_clean = proc.stdout.strip()
        json_start = stdout_clean.find("{")
        json_end = stdout_clean.rfind("}") + 1
        if json_start == -1 or json_end == -1:
            return JsonResponse({"error": "No JSON output found", "raw": stdout_clean}, status=500)

        try:
            results_json = json.loads(stdout_clean[json_start:json_end])
        except json.JSONDecodeError as e:
            return JsonResponse({"error": f"Invalid JSON: {str(e)}", "raw": stdout_clean}, status=500)

        # ===== STEP 2: GENERATE MANIM SCRIPT =====
        manim_prompt = build_manim_prompt(problem_text, results_json)
        manim_response = model.generate_content(manim_prompt)
        script_text = manim_response.text.strip()

        # Remove markdown fences
        if script_text.startswith("```python"):
            script_text = script_text[len("```python"):].strip()
        if script_text.endswith("```"):
            script_text = script_text[:-3].strip()

        # ===== STEP 2.5: EXTRACT CLASS NAME FROM SCRIPT =====
        class_match = re.search(r'class\s+(\w+)\s*\(Scene\)', script_text)
        if not class_match:
            return JsonResponse({"error": "Could not find Scene class in generated script"}, status=500)
        
        class_name = class_match.group(1)

        # Write Manim script to temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as tmp:
            tmp.write(script_text)
            tmp_manim_path = tmp.name

        # ===== STEP 3: RENDER VIDEO WITH MANIM =====
        timestamp = int(time.time())
        video_filename = f"manim_video_{timestamp}.mp4"
        
        # Manim outputs to media/videos/<script_name>/<quality>/
        script_basename = os.path.splitext(os.path.basename(tmp_manim_path))[0]
        manim_output_dir = os.path.join("media", "videos", script_basename, "480p15")
        
        # Run manim CLI with extracted class name
        manim_proc = subprocess.run(
            [sys.executable, "-m", "manim",
             tmp_manim_path,
             class_name,
             "-ql",
             "--format", "mp4",
             "--media_dir", "media"],
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="replace"
        )

        # Handle render errors
        if manim_proc.returncode != 0:
            return JsonResponse({
                "error": "Manim render failed",
                "details": manim_proc.stderr.strip()
            }, status=500)

        # Find the generated video (using dynamic class name)
        expected_video = os.path.join(manim_output_dir, f"{class_name}.mp4")
        
        if not os.path.exists(expected_video):
            return JsonResponse({
                "error": "Video file not found after rendering",
                "expected_path": expected_video,
                "class_name": class_name
            }, status=500)

        # Move video to final location
        final_video_path = os.path.join(MEDIA_DIR, video_filename)
        os.makedirs(MEDIA_DIR, exist_ok=True)
        shutil.move(expected_video, final_video_path)

        return JsonResponse({
            "results": results_json,
            "video_file": video_filename,
            "video_url": f"/media/{video_filename}",
            "class_name": class_name
        })

    except subprocess.TimeoutExpired:
        return JsonResponse({"error": "Process timed out"}, status=500)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    finally:
        # Clean up both temp files
        if tmp_solver_path and os.path.exists(tmp_solver_path):
            os.unlink(tmp_solver_path)
        if tmp_manim_path and os.path.exists(tmp_manim_path):
            os.unlink(tmp_manim_path)


def get_video(request, filename):
    video_path = os.path.join(MEDIA_DIR, filename)
    if not os.path.exists(video_path):
        return JsonResponse({"error": "File not found"}, status=404)
    return FileResponse(open(video_path, "rb"), content_type="video/mp4")
