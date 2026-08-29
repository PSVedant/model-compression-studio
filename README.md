# Model Compression & Deployment Studio (MCDS)

A production-style ML platform that compresses, benchmarks, exports, and serves a HuggingFace BERT model — built to demonstrate real MLOps/AIML SDE skills: quantization, pruning, ONNX export, FastAPI serving, and honest evaluation methodology.

**Live demo:** https://model-compression-studio-1.onrender.com
**API docs:** https://model-compression-studio.onrender.com/docs

## What this does

Takes `textattack/bert-base-uncased-SST-2` (a BERT model fine-tuned for sentiment classification) and:
1. Compresses it via INT8 dynamic quantization and L1 magnitude pruning
2. Exports it to ONNX and benchmarks it through ONNX Runtime
3. Serves the best-performing variant via a FastAPI backend
4. Presents live inference + benchmark results through a Gradio dashboard

Every number below is measured, not estimated — see `results/` for raw output.

## Results (872-example SST-2 validation set)

| Method | Size | Accuracy | Latency (CPU) |
|---|---|---|---|
| Baseline (FP32) | 418.35 MB | 92.43% | 158.42 ms |
| Quantized (INT8) | 173.09 MB | 90.71% | 52.59 ms |
| Pruned (L1, 30%) | 417.72 MB | 91.28% | 110.55 ms |
| ONNX (FP32) | 419.09 MB | 92.43% | 71.69 ms |

**Deployed variant:** ONNX (FP32) — identical accuracy to baseline, 54.8% latency reduction from graph-level optimization alone, no accuracy trade-off to justify.

### Key findings

- **Quantization is the strongest lever:** 58.6% smaller, 66.8% faster, for a 1.72-point accuracy cost.
- **Pruning alone gave no real benefit here:** on a 200-example sample it looked accuracy-free, but the full 872-example validation set revealed a real 1.15-point accuracy cost with no size or speed gain — PyTorch's dense tensor format doesn't skip zeroed weights without a sparse-aware kernel. Included as an honest negative result, not hidden.
- **Out-of-domain generalization:** accuracy dropped from ~92% (SST-2) to ~84-85% on a 200-example IMDB sample (long-form reviews vs. SST-2's short phrases). The drop was consistent across baseline/quantized/ONNX — compression didn't disproportionately hurt generalization. Part of the gap is likely a 128-token truncation artifact, since IMDB reviews average well over that length; not claimed as a clean domain-robustness result.
- **GPU (T4) ONNX benchmarking was attempted** but blocked by a Colab CUDA/cuDNN dependency conflict that persisted through multiple fix attempts. All reported results are CPU inference, which matches this project's actual deployment target.

## Architecture

- **Compression/training:** Google Colab (quantization, pruning, ONNX export, all benchmarking)
- **Backend:** FastAPI + ONNX Runtime, serves `/predict`, `/health`, `/benchmarks`, `/metrics`
- **Frontend:** Gradio — Playground (live inference) + Benchmarks (results table)
- **Deployment:** Render (both services, free tier)

## Running locally

```bash
pip install -r requirements-backend.txt
uvicorn app.serve:app --reload --port 8000
```

In a second terminal:

```bash
pip install -r requirements-dashboard.txt
python app/dashboard.py
```

## Limitations (stated honestly)

- Evaluated on SST-2 (short phrases); IMDB test shows a real accuracy drop on longer, out-of-domain text.
- Pruning implementation is unstructured (no sparse-kernel runtime), so its size/latency benefit is theoretical, not realized here.
- GPU inference benchmarking not completed due to environment tooling issues, documented rather than omitted.
- Free-tier Render hosting spins down on inactivity; first request after idle time will be slow (cold start), not representative of steady-state latency.
