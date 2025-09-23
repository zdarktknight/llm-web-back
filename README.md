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
#### 3.1 使用OpenAI API
启动后端服务
```bash
python main.py
```
#### 3.2 采用OpenAI API部署自己的模型
```bash
python -m vllm.entrypoints.openai.api_server --host xx.x.xxx.xxx --port 7862  --model /home/bigue/Desktop/model/Qwen2.5-0.5B-Instruct --gpu-memory-utilization 0.7
```
验证模型是否启动
```
curl http://10.1.188.130:7862/v1/models
```
启动后端服务
```
python main.py
```

#### 3.3 部署自己的模型 (stream mode)
```bash
python vllm_stream.py
```
**Response:**

- Streams the LLM's response as plain text.

## Notes

- Do **not** expose your API key to the frontend.
- Make sure your `.env` file is **not** committed to version control.

---
