# Wolftor Frontend

A modern React.js frontend for the Wolftor AI Problem Solver application.

## Features

- **Landing Page**: Beautiful introduction to the application with feature highlights
- **Chat Interface**: Interactive problem-solving interface with:
  - Image upload with drag & drop
  - Text input for problems
  - Real-time chat with AI responses
  - Responsive design for desktop and mobile

## Tech Stack

- **React 18** with JSX
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **React Router** for navigation
- **Axios** for API calls
- **Lucide React** for icons

## Setup Instructions

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```

3. **Build for Production**
   ```bash
   npm run build
   ```

## API Integration

The frontend integrates with the Django backend APIs:

- `POST /api/image-to-text/` - Extract text from uploaded images
- `POST /api/solve-problem/` - Solve mathematical problems
- `POST /api/tts/` - Text-to-speech conversion (future feature)

## Color Scheme

The application uses a custom color palette defined in CSS variables:

- **Background**: Dark blue-purple gradient
- **Primary Orange**: #FF7F00
- **Secondary Orange**: #CD7F32
- **Card Background**: #2C2C4F
- **Text**: White and light gray variants

## Responsive Design

- Mobile-first approach
- Breakpoints: sm (640px), md (768px), lg (1024px), xl (1280px)
- Optimized for both desktop and mobile devices

## Development

The application is structured as follows:

```
src/
├── components/
│   ├── LandingPage.jsx    # Home page with features
│   └── ChatPage.jsx       # Main chat interface
├── App.jsx                # Main app component with routing
├── main.jsx              # Application entry point
└── style.css             # Global styles and Tailwind imports
```

## Backend Requirements

Make sure the Django backend is running on `http://localhost:8000` with the following endpoints available:

- `/api/image-to-text/` - For image text extraction
- `/api/solve-problem/` - For problem solving
- `/api/tts/` - For text-to-speech (optional)

## Environment Variables

No environment variables are required for the frontend. The API base URL is hardcoded to `http://localhost:8000` for development.
