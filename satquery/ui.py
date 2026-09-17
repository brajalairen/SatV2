"""Gradio web UI (D-004/D-005): builds an AnalysisRequest, calls the analysis function, renders the response."""

from collections.abc import Callable
from pathlib import Path

import gradio as gr

from satquery.api import analyze
from satquery.examples import EXAMPLE_QUERIES, EXAMPLES_DIR, load_scenarios
from satquery.schemas import AnalysisRequest, AnalysisResponse, ImageInput

STEP_HEADERS = ["step", "tool", "model", "permitted parameters", "status", "time (s)", "note"]


def _format(response: AnalysisResponse):
    icon = {"ok": "✅", "partial": "⚠️", "invalid_input": "⛔", "error": "❌"}[response.status]
    answer = f"### {icon} {response.answer}\n\n**Task:** `{response.task}` · **Input:** `{response.trace.input_config}`"
    if response.trace.intent:
        answer += f" · **Routing rule:** {response.trace.intent.matched_rule}"
    gallery = [(e.file, e.label) for e in response.evidence if e.kind == "overlay" and e.file]
    c = response.confidence
    confidence = (f"**Confidence:** {c.value if c.value is not None else 'n/a'} · *method:* {c.method}"
                  + (f"\n\n> {c.note}" if c.note else "")) if c else "**Confidence:** n/a"
    warnings = [f"- **{i.severity}** `{i.code}`: {i.message}" for i in response.trace.validation]
    if warnings:
        confidence += "\n\n**Input checks**\n" + "\n".join(warnings)
    rows = [[s.step_id, s.tool, s.model or "-", str(s.params), s.status, s.duration_s, s.error or ""] for s in response.trace.steps]
    return answer, gallery, confidence, rows, response.trace.model_dump(mode="json"), [response.report_html, response.report_json]


def load_example_rows() -> list[list]:
    """Demo scenarios from demo/examples/examples.json (format in demo/examples/README.md)."""
    rows = []
    for example in load_scenarios():
        slots = (example["images"] + [{}, {}])[:2]
        row = []
        for image in slots:
            row += [str(EXAMPLES_DIR / image["path"]) if image.get("path") else None,
                    image.get("modality", "optical"), image.get("acquired") or ""]
        rows.append(row + [example["query"]])
    return rows


def build_demo(run_analysis: Callable[[AnalysisRequest], AnalysisResponse] = analyze) -> gr.Blocks:
    def run(file_a, modality_a, date_a, file_b, modality_b, date_b, query):
        images = [ImageInput(path=f, modality=m, acquired=(d or "").strip() or None)
                  for f, m, d in ((file_a, modality_a, date_a), (file_b, modality_b, date_b)) if f]
        return _format(run_analysis(AnalysisRequest(query=query or "", images=images)))

    with gr.Blocks(title="SatQuery AI") as demo:
        gr.Markdown("# 🛰️ SatQuery AI\nAsk questions about satellite images. The agent validates the inputs, picks specialist "
                    "tools, runs them, and shows its evidence, confidence, and full execution trace. *SIH26167 prototype.*")
        with gr.Row():
            with gr.Column(scale=1):
                file_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
                file_a = gr.File(label="Image 1 (before, or optical)", file_types=file_types, type="filepath")
                with gr.Row():
                    modality_a = gr.Radio(["optical", "sar"], value="optical", label="Modality")
                    date_a = gr.Textbox(label="Date (optional)", placeholder="YYYY-MM-DD")
                file_b = gr.File(label="Image 2 (optional: after, or SAR)", file_types=file_types, type="filepath")
                with gr.Row():
                    modality_b = gr.Radio(["optical", "sar"], value="optical", label="Modality")
                    date_b = gr.Textbox(label="Date (optional)", placeholder="YYYY-MM-DD")
                query = gr.Textbox(label="Question or instruction", lines=2)
                gr.Examples(EXAMPLE_QUERIES, inputs=query, label="Example queries")
                button = gr.Button("Analyze", variant="primary")
                scenario_rows = load_example_rows()
                if scenario_rows:
                    gr.Examples(scenario_rows, inputs=[file_a, modality_a, date_a, file_b, modality_b, date_b, query],
                                label="Demo scenarios (click to load)")
            with gr.Column(scale=2):
                answer = gr.Markdown()
                gallery = gr.Gallery(label="Visual evidence", columns=1, height="auto")
                confidence = gr.Markdown()
                with gr.Accordion("Execution trace: plan and tool calls", open=True):
                    steps = gr.Dataframe(headers=STEP_HEADERS, wrap=True)
                    trace = gr.JSON(label="Full trace (JSON)")
                reports = gr.File(label="Download report (HTML + JSON)", file_count="multiple")
        button.click(run, [file_a, modality_a, date_a, file_b, modality_b, date_b, query],
                     [answer, gallery, confidence, steps, trace, reports])
    return demo
