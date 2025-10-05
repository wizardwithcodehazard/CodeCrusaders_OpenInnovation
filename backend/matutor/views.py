# matutor/views.py
import os
import json
import traceback
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

# from .tts_service import TTSService

logger = logging.getLogger(__name__)
# Define temp directory for Wolfram scripts
WOLFRAM_TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(WOLFRAM_TEMP_DIR, exist_ok=True)

MEDIA_DIR = os.path.join(os.path.dirname(__file__), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)
# Single shared TTS service instance (lazy-loads the TTS model on first use)
# tts_service = TTSService()

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


# @csrf_exempt  # remove or secure in production
# @require_POST
# def synthesize(request):
#     """
#     POST /api/tts/
#     Body (JSON): {"text": "some text to synthesize"}
#     Response: {"success": True, "filename": "tts_xxx.wav", "url": "http://.../media/tts_xxx.wav"}
#     """
#     # parse JSON body
#     try:
#         payload = json.loads(request.body.decode("utf-8"))
#     except Exception:
#         return HttpResponseBadRequest("Invalid JSON body")

#     text = payload.get("text", "")
#     if not isinstance(text, str) or not text.strip():
#         return HttpResponseBadRequest("Field 'text' is required and must be non-empty")

#     if len(text) > MAX_TEXT_LENGTH:
#         return HttpResponseBadRequest(f"Text too long (max {MAX_TEXT_LENGTH} characters)")

#     try:
#         # Generate WAV file (returns absolute filesystem path)
#         out_path = tts_service.generate(text)
#         filename = os.path.basename(out_path)

#         # Build absolute URL so clients (Postman/browser) can open it
#         media_url = settings.MEDIA_URL if hasattr(settings, "MEDIA_URL") else "/media/"
#         # request.build_absolute_uri handles host/port
#         file_url = request.build_absolute_uri(os.path.join(media_url, filename))

#         return JsonResponse({"success": True, "filename": filename, "url": file_url})
#     except Exception as exc:
#         logger.exception("TTS generation failed")
#         return JsonResponse({"success": False, "error": str(exc)}, status=500)


# def get_audio(request, filename):
#     """
#     Optional direct file endpoint: /media/<filename> (or configured via urls)
#     Returns the generated WAV file, or 404 if missing.
#     """
#     media_root = getattr(settings, "MEDIA_ROOT", os.path.join(os.path.dirname(os.path.dirname(_file_)), "media"))
#     path = os.path.join(media_root, filename)

#     if not os.path.exists(path):
#         return HttpResponseNotFound("File not found")

#     return FileResponse(open(path, "rb"), content_type="audio/wav")


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


# def build_gemini_prompt(problem_text: str) -> str:
#     return f"""
# You are a Python programming expert specializing in physics and mathematics problems using Wolfram Cloud.

# ⚠️ CRITICAL OUTPUT REQUIREMENTS ⚠️
# 1. Output MUST be complete Python code - NO TRUNCATION ALLOWED
# 2. ALWAYS end with complete finally block: cloud.terminate()
# 3. Keep code CONCISE to avoid truncation - use SHORT variable names
# 4. Prioritize COMPLETENESS over verbosity
# 5. Test that you can fit the entire solution in your response

# ANTI-TRUNCATION STRATEGIES:
# - Use short variable names (r, i, v, p instead of resistance, current, voltage, power)
# - Minimize comments - only essential ones
# - Combine operations where possible
# - Keep the results dictionary simple
# - NO verbose variable names
# - NO unnecessary intermediate steps

# MANDATORY FORMAT (NO CODE FENCES):
# - Import statements
# - Initialize cloud session (5 lines max)
# - try block with computation
# - except block: print(json.dumps({{"error": str(e)}}))
# - finally block: cloud.terminate()  ← THIS MUST BE COMPLETE!

# Problem to solve:
# {problem_text}

# TEMPLATE - USE THIS EXACT STRUCTURE:

# import json
# import os
# from wolframclient.evaluation import SecuredAuthenticationKey, WolframCloudSession
# from wolframclient.language import wl, wlexpr

# sak = SecuredAuthenticationKey(os.getenv('WOLFRAM_CONSUMER_KEY'), os.getenv('WOLFRAM_CONSUMER_SECRET'))
# cloud = WolframCloudSession(credentials=sak)

# try:
#     # Your computation here - KEEP IT CONCISE
#     r = cloud.evaluate(wl.N(wlexpr('your_expression')))
#     val = float(r)
    
#     results = {{"result": val}}
#     print(json.dumps(results))
# except Exception as e:
#     print(json.dumps({{"error": str(e)}}))
# finally:
#     cloud.terminate()

# CRITICAL RULES:
# 1. NO markdown code fences (```)
# 2. Use wl.N() for all numerical evaluations
# 3. Convert to float() only after wl.N()
# 4. Use SHORT variable names (1-3 chars)
# 5. Minimize comments
# 6. ALWAYS complete the finally block: cloud.terminate()
# 7. Double curly braces in f-strings: {{{{ }}}}

# WOLFRAM PATTERNS (wrap in wl.N()):
# - Math: wl.N(wlexpr('1/3 + 1/5'))
# - Solve: sol = wlexpr('Solve[eq, x]'); x_val = float(cloud.evaluate(wl.N(wlexpr('x /. sol[[1]]'))))
# - Integrate: wl.N(wl.Integrate(wlexpr('x^2'), wlexpr('{{x, 0, 2}}')))
# - Lists: wl.N(wlexpr('{{1, 2, 3}}'))

# WRONG - WILL TRUNCATE:
# - Long variable names
# - Too many comments
# - Verbose code
# - Missing finally block

# RIGHT - COMPLETE CODE:
# - Short variable names
# - Essential code only
# - Complete finally block

# Generate COMPLETE Python code now (NO CODE FENCES, MUST END WITH cloud.terminate()):
# """
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
        if code_text.startswith("python"):
            code_text = code_text[len("python"):].strip()
        if code_text.endswith(""):
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

MEDIA_DIR = os.path.join(settings.BASE_DIR, "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

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

## VALID MANIM COLORS

ONLY use these predefined Manim color constants. NEVER invent color names:
- Basic: WHITE, BLACK, GRAY, GREY, RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, PINK
- Gray shades: LIGHT_GRAY, DARK_GRAY, DARKER_GRAY, LIGHT_GREY, DARK_GREY, DARKER_GREY
- Blue variants: BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E
- Teal variants: TEAL, TEAL_A, TEAL_B, TEAL_C, TEAL_D, TEAL_E
- Green variants: GREEN_A, GREEN_B, GREEN_C, GREEN_D, GREEN_E
- Yellow variants: YELLOW_A, YELLOW_B, YELLOW_C, YELLOW_D, YELLOW_E
- Gold variants: GOLD, GOLD_A, GOLD_B, GOLD_C, GOLD_D, GOLD_E
- Red variants: RED_A, RED_B, RED_C, RED_D, RED_E
- Maroon variants: MAROON, MAROON_A, MAROON_B, MAROON_C, MAROON_D, MAROON_E
- Purple variants: PURPLE_A, PURPLE_B, PURPLE_C, PURPLE_D, PURPLE_E
- Pink variants: PINK, LIGHT_PINK

⚠️ CRITICAL: Colors like GREEN_SCREEN, NEON_GREEN, BRIGHT_BLUE, etc. DO NOT EXIST. Use only the colors listed above.

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
Each frame should only show relevant content
Remove ALL non relevant previous elements before introducing new ones
Fade previous content before adding new content to avoid overlapping and messy visuals
Never let elements accumulate

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
class PhysicsProblem(Scene):
    def construct(self):
        # Set background
        self.camera.background_color = BLACK
        
        # Section 1: Title
        title = Text("Title", font_size=36, color=YELLOW).to_edge(UP, buff=0.1)
        self.play(Write(title))
        
        # Section 2: Main content
        content = MathTex("formula", color=TEAL).move_to(ORIGIN)
        self.play(Write(content))
        
        # Section 3: Subtitle
        subtitle = Text("explanation", font_size=24, color=WHITE).to_edge(DOWN, buff=0.1)
        self.add(subtitle)
        self.play(FadeIn(subtitle))
        
        # Cleanup before next section
        self.play(FadeOut(title, content, subtitle))
        self.remove(title, content, subtitle)
        
        # Next section...
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
- For bigger texts, split into multiple lines instead of exceeding screen width

## ANIMATION BUILDING BLOCKS

Math Expressions:

```python
equation = MathTex(r"E = mc^2", font_size=32, color=BLUE).move_to(ORIGIN)
self.play(Write(equation))

# To highlight results:
box = SurroundingRectangle(equation, buff=0.15, color=YELLOW, stroke_width=3)
self.play(Create(box))
```

Showing Steps with Transform:

```python
eq1 = MathTex(r"E = mc^2", color=WHITE)
eq2 = MathTex(r"E = (2)(3)^2", color=WHITE)
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

## SCENE FLOW / BEST PRACTICES

The script should follow this sequence:

1. Intro Title & Subtitle (problem statement)
2. Display Givens (list of knowns)
3. Show Formula (highlighted, maybe boxed)
4. Step-by-Step Substitution (use Transform to morph equations)
5. Final Result (boxed, large font, color highlight)
6. Optional Visualization (diagrams, circuits, etc.)
7. Summary Slide (results recap)

## VALID ANIMATIONS TO USE

- Write()
- FadeIn()
- FadeOut()
- Transform()
- ReplacementTransform()
- Create() (for lines/shapes)
- GrowFromCenter()
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
step3 = MathTex(r"\\theta = 2 \\arctan\\left({{:f}}\\right)".format(val))
```

✅ CORRECT - Use string concatenation:
```python
step3 = MathTex(r"\\theta = 2 \\arctan\\left(" + f"{{val:.6f}}" + r"\\right)")
```

### PROPER MATH TEX FORMATTING PATTERNS

For single variables:
```python
# Correct
result = MathTex(r"\\theta \\approx " + f"{{theta_deg:.2f}}" + r"^\\circ", color=GREEN)

# Wrong
result = MathTex(r"\\theta \\approx {{:.2f}}^\\circ".format(theta_deg))
```

For multiple variables:
```python
# Correct
equation = MathTex(r"E = " + f"{{mass:.1f}}" + r" \\times " + f"{{speed:.0f}}" + r"^2", color=BLUE)
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
4. ✅ Always specify a valid color from the approved list

Common issues to avoid:
- ❌ Undefined colors (GREEN_SCREEN, NEON_GREEN, etc.)
- ❌ stroke_dash_offset parameter (doesn't exist)
- ❌ align_to=LEFT in next_to() (use aligned_edge=LEFT)

## CRITICAL MISTAKES TO AVOID

NEVER:
- Use undefined color constants
- Let text overlap between sections
- Skip cleanup between scenes
- Use deprecated API methods
- Mix old and new content without proper removal
- Use incorrect video dimensions (always 480p, 15fps)
- Use deprecated syntax like TexText, ShowCreation
- Leave undefined variables
- Use .format() with MathTex containing LaTeX

Rules for generating Manim scripts:

Do not directly index into MathTex or Tex submobjects using numeric indices (like [0][4], [2][3]).

If you want to highlight or surround a part of an equation, use:

get_part_by_tex("symbol") for targeting a specific symbol, e.g. eq.get_part_by_tex("R").

Or surround the whole object with SurroundingRectangle(eq).

If you must use indexing, always check len(eq) first before accessing submobjects.

When animating highlights across multiple equations, prefer VGroup(eq1, eq2, eq3) instead of indexing into sub-elements.

Always prioritize readability and stability of the script over fancy indexing.

🔧 Example (before vs after)
❌ Bad (indexing, can crash):
SurroundingRectangle(currents_calc[0][4])

✅ Good (robust):
SurroundingRectangle(currents_calc[0])  
# or
currents_calc[0].get_part_by_tex("I")

ALWAYS:
- Use only valid Manim color constants
- Clean up completely before new content
- Use proper section positioning
- Use appropriate font sizes for 480p
- Remove elements with both FadeOut() and self.remove()
- Use current Manim Community API
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
9. Only valid Manim color constants
10. Final script must run standalone with Manim CE

---
⚠️ Do not use any undefined variables. Always reference actual objects created in the scene.
Remember: Use ONLY valid Manim colors, clean up completely before each new scene, maintain 3-section layout, always use 480p/15fps configuration, and use string concatenation for MathTex with variables.

Problem Text:
{problem_text}

Results JSON:
{results_str}
"""


def build_transcript_prompt(manim_script: str, problem_text: str, results_json: dict) -> str:
    results_str = json.dumps(results_json, indent=2)
    return f"""
You are an expert educational narrator. Analyze the provided Manim animation script and create a natural, engaging voiceover transcript that explains the problem and solution step-by-step.

INSTRUCTIONS:
1. Analyze the Manim script to understand the visual flow and timing
2. Create a clear, conversational narration that follows the animation
3. Keep it concise but educational - aim for 2-4 minutes of speech
4. Use simple language suitable for high school students
5. Explain each step as it appears in the animation
6. Mention key values and results from the calculations
7. Make it engaging and easy to follow
8. Do NOT include any markdown, labels, or formatting - just plain speech text

MANIM SCRIPT:
{manim_script}

ORIGINAL PROBLEM:
{problem_text}

CALCULATED RESULTS:
{results_str}

Generate a natural voiceover script that a teacher would use to narrate this animation. Output ONLY the transcript text, no markdown, no labels, just the speech text that should be spoken.
"""



# @csrf_exempt
# @require_POST
# def generate_video(request):
#     if request.method != "POST":
#         return HttpResponseBadRequest("Only POST allowed")

#     # Try to parse JSON body first, then fall back to form data
#     problem_text = None
    
#     try:
#         if request.content_type == 'application/json':
#             data = json.loads(request.body.decode('utf-8'))
#             problem_text = data.get("problem")
#         else:
#             problem_text = request.POST.get("problem")
#     except (json.JSONDecodeError, UnicodeDecodeError) as e:
#         return HttpResponseBadRequest(f"Invalid request format: {str(e)}")

#     if not problem_text:
#         return HttpResponseBadRequest("Missing 'problem' field")

#     # Generate unique filenames
#     timestamp = int(time.time() * 1000)
#     solver_filename = f"wolfram_solver_{timestamp}.py"
#     manim_filename = f"manim_script_{timestamp}.py"
#     solver_path = os.path.join(WOLFRAM_TEMP_DIR, solver_filename)
#     manim_path = os.path.join(WOLFRAM_TEMP_DIR, manim_filename)
    
#     try:
#         # ===== STEP 1: SOLVE THE PROBLEM WITH WOLFRAM =====
#         model = genai.GenerativeModel("gemini-2.5-flash")
#         response = model.generate_content(build_gemini_prompt(problem_text))

#         # Extract Python code (strip markdown if present)
#         code_text = response.text.strip()
#         if code_text.startswith("python"):
#             code_text = code_text[len("python"):].strip()
#         if code_text.endswith(""):
#             code_text = code_text[:-3].strip()

#         # Write solver code to temp folder
#         with open(solver_path, "w", encoding="utf-8") as f:
#             f.write(code_text)

#         # Run the Wolfram solver script
#         proc = subprocess.run(
#             [sys.executable, solver_path],
#             capture_output=True,
#             text=True,
#             timeout=60,  # Increased timeout for Wolfram
#             encoding="utf-8",
#             errors="replace"
#         )

#         # Handle solver errors
#         if proc.returncode != 0:
#             return JsonResponse({
#                 "error": "Wolfram solver failed",
#                 "details": proc.stderr.strip(),
#                 "solver_file": solver_filename
#             }, status=500)

#         # Extract JSON from solver output
#         stdout_clean = proc.stdout.strip()
#         json_start = stdout_clean.find("{")
#         json_end = stdout_clean.rfind("}") + 1
#         if json_start == -1 or json_end == -1:
#             return JsonResponse({
#                 "error": "No JSON output found",
#                 "raw": stdout_clean,
#                 "solver_file": solver_filename
#             }, status=500)

#         try:
#             results_json = json.loads(stdout_clean[json_start:json_end])
#         except json.JSONDecodeError as e:
#             return JsonResponse({
#                 "error": f"Invalid JSON: {str(e)}",
#                 "raw": stdout_clean,
#                 "solver_file": solver_filename
#             }, status=500)

#         # ===== STEP 2: GENERATE MANIM SCRIPT =====
#         manim_prompt = build_manim_prompt(problem_text, results_json)
#         manim_response = model.generate_content(manim_prompt)
#         script_text = manim_response.text.strip()

#         # Remove markdown fences
#         if script_text.startswith("python"):
#             script_text = script_text[len("python"):].strip()
#         if script_text.endswith(""):
#             script_text = script_text[:-3].strip()

#         # ===== STEP 2.5: EXTRACT CLASS NAME FROM SCRIPT =====
#         class_match = re.search(r'class\s+(\w+)\s*\(Scene\)', script_text)
#         if not class_match:
#             return JsonResponse({
#                 "error": "Could not find Scene class in generated script",
#                 "solver_file": solver_filename
#             }, status=500)
        
#         class_name = class_match.group(1)

#         # Write Manim script to temp folder
#         with open(manim_path, "w", encoding="utf-8") as f:
#             f.write(script_text)

#         # ===== STEP 3: RENDER VIDEO WITH MANIM =====
#         video_filename = f"manim_video_{timestamp}.mp4"
        
#         # Manim outputs to media/videos/<script_name>/<quality>/
#         script_basename = os.path.splitext(os.path.basename(manim_path))[0]
#         manim_output_dir = os.path.join("media", "videos", script_basename, "480p15")
        
#         # Run manim CLI with extracted class name
#         manim_proc = subprocess.run(
#             [sys.executable, "-m", "manim",
#              manim_path,
#              class_name,
#              "-ql",
#              "--format", "mp4",
#              "--media_dir", "media"],
#             capture_output=True,
#             text=True,
#             timeout=300,
#             encoding="utf-8",
#             errors="replace"
#         )

#         # Handle render errors
#         if manim_proc.returncode != 0:
#             return JsonResponse({
#                 "error": "Manim render failed",
#                 "details": manim_proc.stderr.strip(),
#                 "solver_file": solver_filename,
#                 "manim_file": manim_filename
#             }, status=500)

#         # Find the generated video (using dynamic class name)
#         expected_video = os.path.join(manim_output_dir, f"{class_name}.mp4")
        
#         if not os.path.exists(expected_video):
#             return JsonResponse({
#                 "error": "Video file not found after rendering",
#                 "expected_path": expected_video,
#                 "class_name": class_name,
#                 "solver_file": solver_filename,
#                 "manim_file": manim_filename
#             }, status=500)

#         # Move video to final location
#         final_video_path = os.path.join(MEDIA_DIR, video_filename)
#         os.makedirs(MEDIA_DIR, exist_ok=True)
#         shutil.move(expected_video, final_video_path)

#         # ===== STEP 4: GENERATE TRANSCRIPT FROM MANIM SCRIPT =====
#         transcript_prompt = build_transcript_prompt(script_text, problem_text, results_json)
#         transcript_response = model.generate_content(transcript_prompt)
#         transcript_text = transcript_response.text.strip()

#         return JsonResponse({
#             "results": results_json,
#             "video_file": video_filename,
#             "video_url": f"/media/{video_filename}",
#             "class_name": class_name,
#             "transcript": transcript_text,
#             "computation_engine": "Wolfram Engine",
#             "solver_file": solver_filename,
#             "manim_file": manim_filename
#         })

#     except subprocess.TimeoutExpired:
#         return JsonResponse({
#             "error": "Process timed out",
#             "solver_file": solver_filename if 'solver_filename' in locals() else None,
#             "manim_file": manim_filename if 'manim_filename' in locals() else None
#         }, status=500)
#     except Exception as e:
#         return JsonResponse({
#             "error": str(e),
#             "solver_file": solver_filename if 'solver_filename' in locals() else None,
#             "manim_file": manim_filename if 'manim_filename' in locals() else None
#         }, status=500)
#     # Note: Files are kept in temp folder for debugging


def get_video(request, filename):
    """Serve video files from multiple possible locations."""
    # Try direct MEDIA_DIR first
    video_path = os.path.join(MEDIA_DIR, filename)
    
    if not os.path.exists(video_path):
        # Try searching in media/videos subdirectories
        videos_dir = os.path.join("media", "videos","videos")
        if os.path.exists(videos_dir):
            for root, dirs, files in os.walk(videos_dir):
                if filename in files:
                    video_path = os.path.join(root, filename)
                    logger.info(f"Found video at: {video_path}")
                    break
    
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {filename}")
        logger.debug(f"Searched in: {MEDIA_DIR}")
        return JsonResponse({"error": "File not found", "filename": filename}, status=404)
    
    logger.info(f"Serving video: {video_path}")
    return FileResponse(open(video_path, "rb"), content_type="video/mp4")

def build_explanation_prompt(problem_text: str, results_json: dict) -> str:
    results_str = json.dumps(results_json, indent=2)
    return f"""
You are an expert teacher explaining a physics/mathematics problem to high school students. Your task is to provide a clear, step-by-step explanation of how to solve this problem.

PROBLEM:
{problem_text}

CALCULATED RESULTS:
{results_str}

INSTRUCTIONS:
1. Start with a brief introduction explaining what type of problem this is
2. List the given information clearly
3. Explain the formula or concept needed to solve the problem
4. Break down the solution into clear, numbered steps
5. Show the calculations with explanations for each step
6. Explain WHY each step is necessary (don't just show calculations)
7. State the final answer clearly
8. Add a brief conclusion or key takeaway

FORMATTING GUIDELINES:
- Use markdown formatting for better readability
- Use **bold** for important terms and values
- Use numbered lists for steps
- Use code blocks for formulas: `formula here`
- Keep language simple and conversational
- Avoid jargon; if you must use technical terms, explain them
- Use real-world analogies when helpful
- Make it educational and easy to follow

Generate a complete step-by-step explanation that a student can easily understand and learn from.
"""
# In-memory cache for storing computation results
# Key: problem_text hash, Value: {results, timestamp}
COMPUTATION_CACHE = {}
CACHE_EXPIRY = 3600  # 1 hour in seconds


def build_gemini_prompt(problem_text: str) -> str:
    return f"""
You are a Python programming expert specializing in physics and mathematics problems using Wolfram Cloud.

⚠️ CRITICAL: Keep code SHORT and COMPLETE. Use minimal variable names and comments.

Problem: {problem_text}

Output ONLY Python code (NO code fences). Must include:
1. Imports
2. Cloud session setup
3. try-except-finally with cloud.terminate()

TEMPLATE:
import json
import os
from wolframclient.evaluation import SecuredAuthenticationKey, WolframCloudSession
from wolframclient.language import wl, wlexpr

sak = SecuredAuthenticationKey(os.getenv('WOLFRAM_CONSUMER_KEY'), os.getenv('WOLFRAM_CONSUMER_SECRET'))
cloud = WolframCloudSession(credentials=sak)

try:
    r = cloud.evaluate(wl.N(wlexpr('your_math')))
    v = float(r)
    results = {{"result": v}}
    print(json.dumps(results))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
finally:
    cloud.terminate()

RULES:
- Use wl.N() for numbers
- Short names (r, v, i, p)
- Double braces in f-strings: {{{{ }}}}
- MUST end with: cloud.terminate()

Generate code:
"""


def fix_truncated_wolfram_code(code_text: str) -> str:
    """
    Post-processes generated code to fix common truncation issues.
    Ensures the code has proper try-except-finally structure.
    """
    lines = code_text.strip().split('\n')
    
    # Check if finally block exists and is complete
    has_finally = any('finally:' in line for line in lines)
    has_terminate = any('cloud.terminate()' in line for line in lines)
    
    logger.debug(f"Code has finally block: {has_finally}, has terminate: {has_terminate}")
    
    # If finally exists but incomplete, fix it
    if has_finally and not has_terminate:
        # Find the finally line
        for i, line in enumerate(lines):
            if 'finally:' in line:
                # Check if next line has incomplete terminate
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if 'terminat' in next_line and 'terminate()' not in next_line:
                        # Fix the truncated line
                        logger.warning(f"Fixed truncated line: {next_line} -> cloud.terminate()")
                        lines[i + 1] = '    cloud.terminate()'
                        break
                else:
                    # Add the terminate line
                    logger.warning("Added missing cloud.terminate() after finally")
                    lines.append('    cloud.terminate()')
                    break
    
    # If no finally block at all, add it
    elif not has_finally:
        logger.warning("Added complete finally block")
        lines.append('')
        lines.append('finally:')
        lines.append('    cloud.terminate()')
    
    return '\n'.join(lines)


def get_problem_cache_key(problem_text: str) -> str:
    """Generate a cache key from problem text."""
    import hashlib
    return hashlib.md5(problem_text.encode()).hexdigest()


def get_cached_results(problem_text: str):
    """Retrieve cached results if available and not expired."""
    cache_key = get_problem_cache_key(problem_text)
    if cache_key in COMPUTATION_CACHE:
        cached = COMPUTATION_CACHE[cache_key]
        age = time.time() - cached['timestamp']
        if age < CACHE_EXPIRY:
            logger.info(f"Cache hit for problem (age: {age:.1f}s)")
            return cached['results']
        else:
            logger.info(f"Cache expired for problem (age: {age:.1f}s)")
            del COMPUTATION_CACHE[cache_key]
    return None


def set_cached_results(problem_text: str, results: dict):
    """Store results in cache."""
    cache_key = get_problem_cache_key(problem_text)
    COMPUTATION_CACHE[cache_key] = {
        'results': results,
        'timestamp': time.time()
    }
    logger.info(f"Cached results for problem (cache size: {len(COMPUTATION_CACHE)})")


def run_wolfram_solver(problem_text: str, solver_path: str):
    """
    Run Wolfram solver and return results.
    Returns tuple: (success: bool, results: dict, error_info: dict)
    """
    try:
        # Generate Wolfram code
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={
                "max_output_tokens": 8192,
                "temperature": 0.1,
            }
        )
        logger.debug("Generating Wolfram code with Gemini...")
        response = model.generate_content(build_gemini_prompt(problem_text))

        # Extract Python code (strip markdown if present)
        code_text = response.text.strip()
        if code_text.startswith("```python"):
            code_text = code_text[len("```python"):].strip()
        if code_text.endswith("```"):
            code_text = code_text[:-3].strip()

        logger.debug(f"Generated code length: {len(code_text)} chars")

        # Fix truncation issues
        code_text = fix_truncated_wolfram_code(code_text)

        # Write solver code to temp folder
        with open(solver_path, "w", encoding="utf-8") as f:
            f.write(code_text)
        logger.debug(f"Wrote solver code to {solver_path}")

        # Run the Wolfram solver script
        logger.debug("Running Wolfram solver...")
        proc = subprocess.run(
            [sys.executable, solver_path],
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace"
        )

        # Handle solver errors
        if proc.returncode != 0:
            logger.error(f"Wolfram solver failed with return code {proc.returncode}")
            logger.error(f"stderr: {proc.stderr}")
            return False, None, {
                "error": "Wolfram solver failed",
                "details": proc.stderr.strip(),
                "generated_code": code_text,
                "return_code": proc.returncode
            }

        # Extract JSON from solver output
        stdout_clean = proc.stdout.strip()
        logger.debug(f"Solver stdout length: {len(stdout_clean)} chars")
        
        json_start = stdout_clean.find("{")
        json_end = stdout_clean.rfind("}") + 1
        
        if json_start == -1 or json_end == -1:
            logger.error("No JSON found in solver output")
            return False, None, {
                "error": "No JSON output found",
                "raw": stdout_clean,
                "generated_code": code_text
            }

        try:
            results_json = json.loads(stdout_clean[json_start:json_end])
            logger.info(f"Successfully parsed results: {list(results_json.keys())}")
            return True, results_json, None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return False, None, {
                "error": f"Invalid JSON: {str(e)}",
                "raw": stdout_clean,
                "generated_code": code_text
            }

    except Exception as e:
        logger.exception("Unexpected error in run_wolfram_solver")
        return False, None, {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@csrf_exempt
@require_POST
def explain_problem(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST allowed")

    # Parse JSON body
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body.decode('utf-8'))
            problem_text = data.get("problem")
        else:
            problem_text = request.POST.get("problem")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Request parsing error: {str(e)}")
        return HttpResponseBadRequest(f"Invalid request format: {str(e)}")

    if not problem_text:
        return HttpResponseBadRequest("Missing 'problem' field")

    logger.info(f"explain_problem called for: {problem_text[:50]}...")

    # Check cache first
    cached_results = get_cached_results(problem_text)
    if cached_results:
        logger.info("Using cached results")
        
        # Generate explanation for cached results
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            explanation_prompt = build_explanation_prompt(problem_text, cached_results)
            explanation_response = model.generate_content(explanation_prompt)
            explanation_text = explanation_response.text.strip()

            return JsonResponse({
                "problem": problem_text,
                "results": cached_results,
                "explanation": explanation_text,
                "computation_engine": "Wolfram Engine",
                "cached": True
            })
        except Exception as e:
            logger.error(f"Error generating explanation for cached results: {str(e)}")
            # Continue to return cached results even if explanation fails
            return JsonResponse({
                "problem": problem_text,
                "results": cached_results,
                "explanation": "Error generating explanation",
                "computation_engine": "Wolfram Engine",
                "cached": True
            })

    # Generate unique filename for this request
    timestamp = int(time.time() * 1000)
    solver_filename = f"wolfram_solver_{timestamp}.py"
    solver_path = os.path.join(WOLFRAM_TEMP_DIR, solver_filename)
    
    try:
        # Run Wolfram solver
        success, results_json, error_info = run_wolfram_solver(problem_text, solver_path)
        
        if not success:
            error_info["solver_file"] = solver_filename
            return JsonResponse(error_info, status=500)

        # Cache the results
        set_cached_results(problem_text, results_json)

        # Generate detailed explanation
        logger.debug("Generating explanation...")
        model = genai.GenerativeModel("gemini-2.5-flash")
        explanation_prompt = build_explanation_prompt(problem_text, results_json)
        explanation_response = model.generate_content(explanation_prompt)
        explanation_text = explanation_response.text.strip()

        return JsonResponse({
            "problem": problem_text,
            "results": results_json,
            "explanation": explanation_text,
            "computation_engine": "Wolfram Engine",
            "solver_file": solver_filename,
            "cached": False
        })

    except subprocess.TimeoutExpired:
        logger.error("Solver process timed out")
        return JsonResponse({
            "error": "Process timed out",
            "solver_file": solver_filename,
            "hint": "Wolfram Engine may take longer to initialize on first run"
        }, status=500)
    except Exception as e:
        logger.exception("Unexpected error in explain_problem")
        return JsonResponse({
            "error": str(e),
            "solver_file": solver_filename if 'solver_filename' in locals() else None,
            "traceback": traceback.format_exc()
        }, status=500)

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
        logger.error(f"Request parsing error: {str(e)}")
        return HttpResponseBadRequest(f"Invalid request format: {str(e)}")

    if not problem_text:
        return HttpResponseBadRequest("Missing 'problem' field")

    logger.info(f"generate_video called for: {problem_text[:50]}...")

    # Generate unique filenames
    timestamp = int(time.time() * 1000)
    solver_filename = f"wolfram_solver_{timestamp}.py"
    manim_filename = f"manim_script_{timestamp}.py"
    solver_path = os.path.join(WOLFRAM_TEMP_DIR, solver_filename)
    manim_path = os.path.join(WOLFRAM_TEMP_DIR, manim_filename)
    
    try:
        # ===== STEP 1: GET OR COMPUTE RESULTS =====
        cached_results = get_cached_results(problem_text)
        
        if cached_results:
            logger.info("Using cached Wolfram results for video generation")
            results_json = cached_results
        else:
            logger.info("No cached results found, computing with Wolfram...")
            success, results_json, error_info = run_wolfram_solver(problem_text, solver_path)
            
            if not success:
                error_info["solver_file"] = solver_filename
                return JsonResponse(error_info, status=500)
            
            # Cache the results
            set_cached_results(problem_text, results_json)

        # ===== STEP 2: GENERATE MANIM SCRIPT =====
        logger.debug("Generating Manim script...")
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={
                "max_output_tokens": 8192,
                "temperature": 0.1,
            }
        )
        
        manim_prompt = build_manim_prompt(problem_text, results_json)
        manim_response = model.generate_content(manim_prompt)
        script_text = manim_response.text.strip()

        # Remove markdown fences
        if script_text.startswith("```python"):
            script_text = script_text[len("```python"):].strip()
        if script_text.endswith("```"):
            script_text = script_text[:-3].strip()

        logger.debug(f"Generated Manim script length: {len(script_text)} chars")

        # ===== STEP 2.5: EXTRACT CLASS NAME FROM SCRIPT =====
        class_match = re.search(r'class\s+(\w+)\s*\(Scene\)', script_text)
        if not class_match:
            logger.error("Could not find Scene class in generated script")
            return JsonResponse({
                "error": "Could not find Scene class in generated script",
                "solver_file": solver_filename,
                "manim_script": script_text[:500]  # First 500 chars for debugging
            }, status=500)
        
        class_name = class_match.group(1)
        logger.info(f"Extracted class name: {class_name}")

        # Write Manim script to temp folder
        with open(manim_path, "w", encoding="utf-8") as f:
            f.write(script_text)
        logger.debug(f"Wrote Manim script to {manim_path}")

        # ===== STEP 3: RENDER VIDEO WITH MANIM =====
        video_filename = f"manim_video_{timestamp}.mp4"
        
        # Manim outputs to media/videos/<script_name>/<quality>/
        script_basename = os.path.splitext(os.path.basename(manim_path))[0]
        manim_output_dir = os.path.join("media", "videos", script_basename, "480p15")
        
        logger.debug(f"Rendering video with Manim (class: {class_name})...")
        
        # Run manim CLI with extracted class name
        manim_proc = subprocess.run(
            [sys.executable, "-m", "manim",
             manim_path,
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
            logger.error(f"Manim render failed with return code {manim_proc.returncode}")
            logger.error(f"Manim stderr: {manim_proc.stderr}")
            return JsonResponse({
                "error": "Manim render failed",
                "details": manim_proc.stderr.strip(),
                "solver_file": solver_filename,
                "manim_file": manim_filename,
                "manim_script": script_text[:500]
            }, status=500)

        # Find the generated video (using dynamic class name)
        expected_video = os.path.join(manim_output_dir, f"{class_name}.mp4")
        logger.debug(f"Looking for video at: {expected_video}")
        
        if not os.path.exists(expected_video):
            logger.error(f"Video file not found at expected path: {expected_video}")
            return JsonResponse({
                "error": "Video file not found after rendering",
                "expected_path": expected_video,
                "class_name": class_name,
                "solver_file": solver_filename,
                "manim_file": manim_filename
            }, status=500)

        # Save video to media/videos/ directory
        videos_dir = os.path.join(MEDIA_DIR, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        final_video_path = os.path.join(videos_dir, video_filename)
        
        # Copy the video file
        shutil.copy2(expected_video, final_video_path)
        logger.info(f"Video copied to: {final_video_path}")
        
        # Verify the file exists at the final location
        if not os.path.exists(final_video_path):
            logger.error(f"Video file not found at final path after copy: {final_video_path}")
            return JsonResponse({
                "error": "Video file not saved properly",
                "expected_path": final_video_path,
                "source_path": expected_video
            }, status=500)
        
        logger.info(f"Video successfully saved to: {final_video_path}")
        logger.info(f"Video URL will be: /media/videos/{video_filename}")

        # ===== STEP 4: GENERATE TRANSCRIPT FROM MANIM SCRIPT =====
        logger.debug("Generating transcript...")
        transcript_prompt = build_transcript_prompt(script_text, problem_text, results_json)
        transcript_response = model.generate_content(transcript_prompt)
        transcript_text = transcript_response.text.strip()

        return JsonResponse({
            "results": results_json,
            "video_file": video_filename,
            "video_url": f"/media/videos/{video_filename}",  # Updated URL path
            "class_name": class_name,
            "transcript": transcript_text,
            "computation_engine": "Wolfram Engine",
            "solver_file": solver_filename,
            "manim_file": manim_filename,
            "used_cached_results": cached_results is not None
        })

    except subprocess.TimeoutExpired:
        logger.error("Process timed out")
        return JsonResponse({
            "error": "Process timed out",
            "solver_file": solver_filename if 'solver_filename' in locals() else None,
            "manim_file": manim_filename if 'manim_filename' in locals() else None
        }, status=500)
    except Exception as e:
        logger.exception("Unexpected error in generate_video")
        return JsonResponse({
            "error": str(e),
            "solver_file": solver_filename if 'solver_filename' in locals() else None,
            "manim_file": manim_filename if 'manim_filename' in locals() else None,
            "traceback": traceback.format_exc()
        }, status=500)

def build_wolftor_system_prompt() -> str:
    """Build the system prompt for Wolftor chatbot"""
    return """
You are Wolftor, an expert mathematician chatbot powered by Gemini AI and Wolfram Engine for accuracy and validation.

YOUR IDENTITY:
- Name: Wolftor
- Specialty: Mathematics, Physics, Engineering, and Scientific Computing
- Powered by: Gemini AI + Wolfram Engine
- Personality: Concise, precise, friendly

CRITICAL RESPONSE RULES:
- Keep ALL responses to maximum 2-3 lines (50 words max)
- Be extremely concise and direct
- No long explanations unless specifically asked
- Give only the essential answer
- For complex problems, give just the final answer or key steps

YOUR BEHAVIOR:
1. STAY IN SCOPE: Only respond to mathematics, physics, engineering, and science-related queries
2. BE CONCISE: Maximum 2-3 lines, no exceptions
3. BE DIRECT: Answer first, explain only if asked
4. BE ACCURATE: Use Wolfram Engine for validation
5. BE FRIENDLY: But brief

WHEN QUERY IS OUT OF SCOPE:
Respond with ONLY:
"I'm Wolftor, a math bot! 🧮 I don't do [topic]. Let's solve some math instead?"

RESPONSE EXAMPLES (MAXIMUM 2-3 LINES):

User: "Hi!"
Wolftor: "Hey! I'm Wolftor, your math assistant powered by Wolfram Engine. What problem can I solve? 🧮"

User: "Solve x² - 5x + 6 = 0"
Wolftor: "x = 2 or x = 3. Factoring gives (x-2)(x-3) = 0."

User: "What is the derivative of x²?"
Wolftor: "d/dx(x²) = 2x using the power rule."

User: "Explain calculus"
Wolftor: "Calculus studies continuous change through derivatives (rates) and integrals (accumulation). Want a specific problem?"

User: "What's the weather today?"
Wolftor: "I'm Wolftor, a math bot! 🧮 I don't do weather. Let's solve some math instead?"

User: "Tell me a joke"
Wolftor: "I'm Wolftor, a math bot! 🧮 I don't do jokes. Let's solve some math instead?"

NOW RESPOND TO THE USER'S MESSAGE AS WOLFTOR (REMEMBER: MAXIMUM 2-3 LINES):
"""


def build_wolftor_prompt(user_message: str, conversation_history: list = None) -> str:
    """Build complete prompt with system instructions and conversation history"""
    
    prompt = build_wolftor_system_prompt()
    
    # Add conversation history if provided
    if conversation_history:
        prompt += "\n\nCONVERSATION HISTORY:\n"
        for msg in conversation_history[-5:]:  # Last 5 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{role.upper()}: {content}\n"
    
    # Add current user message
    prompt += f"\n\nUSER: {user_message}\n\nWOLFTOR:"
    
    return prompt

@csrf_exempt
@require_POST
def chat_with_wolftor_simple(request):
    """
    Ultra-simple chat API endpoint - just message in, response out
    
    POST /api/chat-simple/
    Body: {
        "message": "Your message here"
    }
    
    Returns: {
        "response": "Wolftor's response"
    }
    """
    
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST allowed")

    # Parse JSON body
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body.decode('utf-8'))
            user_message = data.get("problem")
        else:
            user_message = request.POST.get("problem")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return HttpResponseBadRequest(f"Invalid request format: {str(e)}")

    if not user_message:
        return HttpResponseBadRequest("Missing 'problem' field")

    try:
        # ===== CALL GEMINI WITH WOLFTOR PROMPT =====
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Build prompt with system instructions
        full_prompt = build_wolftor_prompt(user_message)
        
        # Generate response
        response = model.generate_content(full_prompt)
        
        # Extract response text
        bot_response = response.text.strip()

        return JsonResponse({
            "response": bot_response
        })

    except Exception as e:
        return JsonResponse({
            "error": str(e)
        }, status=500)