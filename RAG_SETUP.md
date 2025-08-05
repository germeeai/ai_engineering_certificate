# RAG Chat Application Setup

This application connects the RAG functionality from your Jupyter notebook to a React frontend through a FastAPI backend.

## Architecture

- **Frontend**: React app with TypeScript (`frontend/`)
- **Backend**: FastAPI server with RAG agent (`backend.py`)
- **RAG Logic**: Extracted from `rag.ipynb` using LangGraph and LangChain

## Quick Start

### 1. Install Python Dependencies
```bash
# Install Python packages
uv sync
```

### 2. Install Frontend Dependencies
```bash
# Navigate to frontend directory and install packages
cd frontend
npm install
cd ..
```

### 3. Start Both Servers
```bash
# Option 1: Use the helper script
python start_servers.py

# Option 2: Start manually in separate terminals
# Terminal 1 - Backend
python backend.py

# Terminal 2 - Frontend
cd frontend && npm start
```

### 4. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Health**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs

## Configuration

1. Open the frontend at http://localhost:3000
2. Enter your API keys:
   - **OpenAI API Key**: Required for GPT responses
   - **Tavily API Key**: Required for web search functionality

## How It Works

1. **User Input**: User types a question in the React frontend
2. **API Call**: Frontend sends the question and API keys to the Python backend
3. **RAG Processing**: Backend uses the RAG agent to:
   - Search the web using Tavily for relevant information
   - Generate a response using OpenAI GPT with the search context
4. **Response**: Backend returns the AI-generated response to the frontend

## Key Features

- **Advanced RAG**: Uses LangGraph for sophisticated agent workflow
- **Web Search**: Integrated Tavily search for up-to-date information
- **Real-time Chat**: Responsive chat interface
- **API Key Security**: Keys are handled client-side and sent securely to backend
- **Error Handling**: Comprehensive error handling and user feedback

## File Structure

```
├── backend.py              # FastAPI server with RAG agent
├── start_servers.py        # Helper script to start both servers
├── rag.ipynb              # Original RAG notebook (reference)
├── pyproject.toml         # Python dependencies
├── frontend/              # React application
│   ├── src/
│   │   ├── App.tsx        # Main React component
│   │   ├── services/
│   │   │   └── api.ts     # API service (updated for backend)
│   │   ├── types/
│   │   │   └── index.ts   # TypeScript types
│   │   └── components/    # React components
│   └── package.json       # Frontend dependencies
└── RAG_SETUP.md          # This file
```

## Troubleshooting

### Backend Issues
- Ensure all Python dependencies are installed: `uv sync`
- Check that port 8000 is available
- Verify API keys are valid

### Frontend Issues
- Ensure Node.js dependencies are installed: `cd frontend && npm install`
- Check that port 3000 is available
- Verify backend is running at http://localhost:8000

### Connection Issues
- Ensure both servers are running
- Check browser console for error messages
- Verify CORS is properly configured (handled automatically)

## API Endpoints

### POST /chat
Send a message to the RAG agent.

**Request Body:**
```json
{
  "message": "Your question here",
  "openai_key": "your-openai-api-key",
  "tavily_key": "your-tavily-api-key"
}
```

**Response:**
```json
{
  "response": "AI-generated response with search context"
}
```

### GET /health
Check if the backend server is running.

**Response:**
```json
{
  "status": "healthy"
}
```