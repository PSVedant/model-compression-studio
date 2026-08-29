import os
import time
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
import gdown
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

APP_DIR = Path(__file__).parent.parent
ONNX_PATH = APP_DIR / "models" / "onnx" / "model.onnx"
BENCHMARKS_PATH = APP_DIR / "results" / "day5_final_comparison.json"
LABELS = {0: "negative", 1: "positive"}

ONNX_MODEL_ID = "12hsngUtxFaqTOcgGi4aBlO_VrJ0Pt9F-"
ONNX_DATA_ID = "15FIzhmzKriI6dCHb1rDRmcw4K2dOXBfN"


def ensure_model_downloaded():
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ONNX_PATH.exists():
        gdown.download(id=ONNX_MODEL_ID, output=str(ONNX_PATH), quiet=False)
    data_path = ONNX_PATH.parent / "model.onnx.data"
    if not data_path.exists():
        gdown.download(id=ONNX_DATA_ID, output=str(data_path), quiet=False)


app = FastAPI(
    title="Model Compression Studio API",
    version="1.0",
    description="Serves an ONNX-exported, benchmarked BERT-SST2 sentiment model",
)

session = None
tokenizer = None


@app.on_event("startup")
def load_model():
    global session, tokenizer
    ensure_model_downloaded()
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "bert-base-uncased-SST-2 (ONNX FP32)",
        "memory_usage_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 2),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    start = time.perf_counter()

    inputs = tokenizer(request.text, return_tensors="np", truncation=True, max_length=128)
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    logits = session.run(["logits"], ort_inputs)[0][0]

    probs = np.exp(logits) / np.sum(np.exp(logits))
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        label=LABELS[pred_idx],
        confidence=round(confidence, 4),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/benchmarks")
def benchmarks():
    if not BENCHMARKS_PATH.exists():
        raise HTTPException(status_code=404, detail="Benchmark results not found")
    with open(BENCHMARKS_PATH) as f:
        return json.load(f)


@app.get("/metrics")
def metrics():
    process = psutil.Process()
    return {
        "memory_usage_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        "cpu_percent": process.cpu_percent(interval=0.1),
    }