# Roni Chat Frontend

A React TypeScript frontend for the Roni AI Finance Assistant chat application.

## Features

- 🤖 Chat interface matching the provided design
- 🔑 API key configuration modal
- 💬 Real-time messaging with loading states
- 📱 Responsive design
- 🎨 Modern UI with Tailwind-inspired styling

## Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm start
```

3. The app will open at `http://localhost:3000`

## Backend Integration

Make sure the FastAPI backend is running on `http://localhost:8000` before using the chat functionality.

The frontend will prompt for API keys on first launch:
- OpenAI API Key (required)
- Tavily API Key (required) 
- Cohere API Key (optional)

## Project Structure

```
src/
├── components/
│   ├── ApiKeyModal.tsx     # API key configuration modal
│   ├── ChatMessage.tsx     # Individual chat message component
│   ├── ChatInput.tsx       # Message input component
│   └── SuggestionButtons.tsx # Pre-defined question suggestions
├── services/
│   └── api.ts              # API service for backend communication
├── types.ts                # TypeScript type definitions
├── App.tsx                 # Main application component
└── index.tsx               # Application entry point
```

## Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm eject` - Eject from Create React App