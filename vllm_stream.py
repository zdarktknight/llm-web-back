import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer
import uvicorn
from vllm import LLM, SamplingParams

# 模型路径和参数
MODEL_PATH = "/home/bigue/Desktop/model/Qwen2.5-0.5B-Instruct"
GPU_MEMORY_UTILIZATION = 0.7

# 初始化 vLLM 引擎
llm = LLM(
    model=MODEL_PATH,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    trust_remote_code=True,  # 推荐加上
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# 初始化 FastAPI
app = FastAPI()

# 添加 CORS（可选）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或指定前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
async def get_models():
    # 模拟 OpenAI 的 /v1/models 接口
    return {
        "data": [{"id": os.path.basename(MODEL_PATH).lower().replace("_", "-"), "object": "model"}]
    }


@app.post("/api/llm/stream")
async def chat_completions(request: Request):
    data = await request.json()
    content = data.get("prompt", "")
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 1024)
    stream = data.get("stream", False)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # log.info(tp.substitute(msg=f"msg: {messages}"))
    input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )

    def stream_generator():
        results_generator = llm.generate(input_text, sampling_params)
        for request_output in results_generator:
            for output in request_output.outputs:
                yield output.text

    return StreamingResponse(stream_generator(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7890)
