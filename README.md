# Backend for Large Model Web App

This backend provides a streaming API for large language model (LLM) chat completions, designed to work with the frontend in this project.

## Features

- Exposes a `/api/llm/stream` POST endpoint for streaming LLM responses.
- Reads model API key, base URL, and model name from a `.env` file.
- Supports CORS for frontend integration.
- Uses [FastAPI](https://fastapi.tiangolo.com/) and [OpenAI Python SDK](https://github.com/openai/openai-python).

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install fastapi uvicorn openai python-dotenv httpx
```

### 2. Create a `.env` file

In the `backend` directory, create a `.env` file with the following content:

```
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-3.5-turbo
```

### 3. Run the backend server

```bash
uvicorn main:app --host 0.0.0.0 --port 7890
```

## API Usage

### POST `/api/llm/stream`

**Request Body:**

```json
{
  "prompt": "Your question here"
}
```

**Response:**

- Streams the LLM's response as plain text.

## Notes

- Do **not** expose your API key to the frontend.
- Make sure your `.env` file is **not** committed to version control.
- The backend must be running for the frontend to access model services.

---
