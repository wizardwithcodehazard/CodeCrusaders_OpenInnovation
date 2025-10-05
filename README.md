# CodeCrusaders Open Innovation - Math Tutor

A full-stack application that helps students solve mathematical and physics problems using AI, with the ability to generate explanatory videos.

## Features

- **Text Problem Solving**: Type mathematical problems and get step-by-step solutions
- **Image Processing**: Upload images of problems and extract text automatically
- **Video Generation**: Generate educational videos explaining the solution process
- **Interactive Chat Interface**: Clean, modern UI for problem solving

## Tech Stack

### Backend
- Django 5.2.7
- Google Gemini AI for problem solving and video script generation
- Manim for video animation
- SQLite database

### Frontend
- React with Vite
- Tailwind CSS for styling
- Axios for API calls

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd CodeCrusaders_OpenInnovation/backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the backend directory with:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

5. Run migrations:
```bash
python manage.py migrate
```

6. Start the Django server:
```bash
python manage.py runserver
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd CodeCrusaders_OpenInnovation/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## API Endpoints

- `POST /api/image-to-text/` - Extract text from uploaded images
- `POST /api/explain-problem/` - Get step-by-step problem explanation
- `POST /api/generate-video/` - Generate educational video for the problem

## Usage

1. **Text Problems**: Type your mathematical problem in the text area and click "Solve Problem Now"
2. **Image Problems**: Upload an image containing your problem, the system will extract text and solve it
3. **Video Generation**: After getting a solution, click "Explain with Video" to generate an educational video

## Workflow

1. User inputs problem (text or image)
2. If image: `image-to-text` endpoint extracts text
3. `explain_problem` endpoint provides solution and explanation
4. User can click "Explain with Video" to generate educational video
5. `generate-video` endpoint creates Manim animation with transcript

## Requirements

- Python 3.8+
- Node.js 16+
- Google Gemini API key
- Manim (included in requirements.txt)

## Notes

- The application uses CORS to allow frontend-backend communication
- Videos are generated using Manim and stored in the media directory
- All AI processing is done through Google Gemini API
