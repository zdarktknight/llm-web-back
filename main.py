import io
import os
import sys

import httpx
import openai
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()  # 加载.env文件

app = FastAPI()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    http_client=httpx.Client(verify=False),
)


@app.post("/api/llm/stream")
async def llm_stream(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    def stream():
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    return StreamingResponse(stream(), media_type="text/plain;charset=utf-8")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7890, lifespan="on", log_level="debug")
