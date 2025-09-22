# vllm_stream.py
import io
import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

app = FastAPI()

stream_engine: AsyncLLM = None  # will be initialized in startup
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    global stream_engine
    engine_args = AsyncEngineArgs(
        model="/home/bigue/Desktop/model/Qwen2.5-0.5B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.7,  # Adjusted to fit your GPU
    )
    stream_engine = AsyncLLM.from_engine_args(engine_args)


async def stream_response(prompt: str, request_id: str):
    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0.8,
        top_p=0.95,
        seed=42,
        output_kind=RequestOutputKind.DELTA,
    )

    async for output in stream_engine.generate(
        request_id=request_id,
        prompt=prompt,
        sampling_params=sampling_params,
    ):
        for completion in output.outputs:
            yield completion.text
        if output.finished:
            break


@app.post("/api/llm/stream")
async def llm_stream(request: Request):
    request = await request.json()
    session_id = request.get("session_id", "ts")
    agent_id = request.get("agent_id", "tt")
    content = request.get("prompt", "")
    generator = stream_response(content, session_id + "_" + agent_id)
    return StreamingResponse(generator, media_type="text/plain")
