import json
from pathlib import Path

import gradio as gr
import requests
import os

API_BASE = os.environ.get("API_BASE","http://localhost:8000")
RESULTS_PATH = Path(__file__).parent.parent / "results" / "day5_final_comparison.json"

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tomorrow:wght@400;500&family=Inter:wght@400;500;600&display=swap');

:root {
    --color-ignition-orange: #ffa41c;
    --color-obsidian: #000000;
    --color-charcoal: #111111;
    --color-graphite: #1d1d1d;
    --color-tarmac: #262626;
    --color-stark-white: #ffffff;
    --color-ash: #aaaaaa;
    --color-dusk-gray: #999999;
}

.gradio-container {
    background-color: var(--color-obsidian) !important;
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3, .eyebrow {
    font-family: 'Tomorrow', sans-serif !important;
    color: var(--color-stark-white) !important;
}

.eyebrow {
    color: var(--color-ignition-orange) !important;
    text-transform: uppercase;
    letter-spacing: 0.84px;
    font-size: 12px;
    font-weight: 500;
}

.gr-button-primary {
    background-color: var(--color-ignition-orange) !important;
    color: var(--color-stark-white) !important;
    border-radius: 8px !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}

.gr-button-secondary {
    background-color: var(--color-charcoal) !important;
    color: var(--color-stark-white) !important;
    border: 1px solid var(--color-tarmac) !important;
    border-radius: 8px !important;
}

.gr-box, .gr-form, .gr-panel {
    background-color: var(--color-graphite) !important;
    border: 1px solid var(--color-tarmac) !important;
    border-radius: 8px !important;
}

body, p, span, label {
    color: var(--color-ash) !important;
}

footer {
    display: none !important;
}

.settings, [aria-label='Settings'] {
    display: none !important;
}
"""


def predict_sentiment(text: str):
    if not text.strip():
        return "—", "—", "—"
    try:
        resp = requests.post(f"{API_BASE}/predict", json={"text": text}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data["label"], f"{data['confidence']*100:.2f}%", f"{data['latency_ms']} ms"
    except requests.exceptions.ConnectionError:
        return "ERROR", "Is uvicorn running on port 8000?", "—"
    except Exception as e:
        return "ERROR", str(e), "—"


def load_benchmark_table():
    if not RESULTS_PATH.exists():
        return [["No benchmark data found", "-", "-", "-"]]
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    rows = []
    labels = {
        "baseline_fp32": "Baseline (FP32)",
        "quantized_int8": "Quantized (INT8)",
        "pruned_l1_30pct": "Pruned (L1, 30%)",
        "onnx_fp32": "ONNX (FP32)",
    }
    for key, label in labels.items():
        if key in data:
            entry = data[key]
            rows.append([
                label,
                f"{entry['size_mb']} MB",
                f"{entry['accuracy_full_872']*100:.2f}%",
                f"{entry['latency_ms']} ms",
            ])
    return rows
GENERALIZATION_PATH = Path(__file__).parent.parent / "results" / "generalization_imdb.json"


def load_generalization_table():
    if not GENERALIZATION_PATH.exists():
        return [["No generalization data found", "-", "-"]]
    with open(GENERALIZATION_PATH) as f:
        data = json.load(f)

    rows = []
    labels = {
        "baseline_fp32": "Baseline (FP32)",
        "quantized_int8": "Quantized (INT8)",
        "onnx_fp32": "ONNX (FP32)",
    }
    for key, label in labels.items():
        if key in data:
            entry = data[key]
            rows.append([
                label,
                f"{entry['sst2_accuracy']*100:.2f}%",
                f"{entry['imdb_accuracy']*100:.2f}%",
            ])
    return rows

with gr.Blocks(title="Model Compression Studio") as demo:
    gr.Markdown('<div class="eyebrow">MODEL COMPRESSION STUDIO</div>')
    gr.Markdown("# Compress, Benchmark, Serve")

    with gr.Tabs():
        with gr.TabItem("Playground"):
            gr.Markdown("Test the deployed ONNX model live — real inference, real latency.")
            with gr.Row():
                input_text = gr.Textbox(
                    label="Input text",
                    placeholder="Type a sentence to classify...",
                    lines=3,
                )
            predict_btn = gr.Button("Run Inference", variant="primary")
            with gr.Row():
                label_out = gr.Textbox(label="Prediction")
                confidence_out = gr.Textbox(label="Confidence")
                latency_out = gr.Textbox(label="Latency")

            predict_btn.click(
                fn=predict_sentiment,
                inputs=[input_text],
                outputs=[label_out, confidence_out, latency_out],
            )

        with gr.TabItem("Benchmarks"):
            gr.Markdown("Real, measured results across compression techniques — full 872-example SST-2 validation set.")
            gr.Dataframe(
                headers=["Method", "Size", "Accuracy", "Latency"],
                value=load_benchmark_table(),
                interactive=False,
            )
                        gr.Markdown("### Out-of-domain generalization (SST-2 → IMDB)")
            gr.Markdown("Same models, tested on 200 IMDB movie reviews (long-form) instead of SST-2 (short phrases) — checks whether compression hurts generalization beyond the benchmark it was measured on.")
            gr.Dataframe(
                headers=["Method", "SST-2 Accuracy", "IMDB Accuracy"],
                value=load_generalization_table(),
                interactive=False,
            )
            gr.Markdown(
                "*Accuracy drops ~8 points on out-of-domain text, consistently across all three variants — "
                "compression did not disproportionately hurt generalization. Part of the gap is likely a "
                "128-token truncation artifact, since IMDB reviews average well over that length.*"
            )
            gr.Markdown(
                "*Quantization: -58.6% size, -66.8% latency, -1.72pt accuracy. "
                "Pruning alone: no size/latency benefit without a sparse-aware kernel. "
                "ONNX: same accuracy as baseline, -54.8% latency from graph optimization.*"
            )


if __name__ == "__main__":
    demo.launch(
        css=CUSTOM_CSS,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )