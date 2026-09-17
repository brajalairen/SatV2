"""Visual evidence overlays and downloadable HTML/JSON reports (D-011)."""

import base64
import html
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from satquery.schemas import AnalysisResponse

COLORS = {"red": (230, 57, 70), "blue": (30, 120, 255), "cyan": (0, 210, 230), "yellow": (255, 205, 0),
          "orange": (255, 140, 0), "green": (46, 204, 113)}


def blend(rgb: np.ndarray, mask: np.ndarray, color: str, alpha: float = 0.45) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(COLORS[color], dtype=np.float32)
    return out.astype(np.uint8)


def overlay(rgb: np.ndarray, masks=(), boxes=()) -> np.ndarray:
    """masks: [(bool array, color)], boxes: [((x0, y0, x1, y1), color)]"""
    out = rgb
    for mask, color in masks:
        if mask is not None and mask.shape == rgb.shape[:2]:
            out = blend(out, mask, color)
    image = Image.fromarray(out)
    draw = ImageDraw.Draw(image)
    width = max(2, rgb.shape[1] // 200)
    for box, color in boxes:
        draw.rectangle(box, outline=COLORS[color], width=width)
    return np.asarray(image)


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 8) -> np.ndarray:
    height = max(left.shape[0], right.shape[0])
    canvas = np.full((height, left.shape[1] + gap + right.shape[1], 3), 255, dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + gap:] = right
    return canvas


def save_png(array: np.ndarray, path: Path) -> str:
    Image.fromarray(array).save(path)
    return str(path)


def write_reports(response: AnalysisResponse, run_dir: Path) -> tuple[str, str]:
    json_path = run_dir / "report.json"
    json_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")

    def embed(path: str) -> str:
        data = base64.b64encode(Path(path).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" style="max-width:100%;border:1px solid #ccc">'

    trace = response.trace
    esc = lambda value: html.escape(str(value))
    overlays = "".join(f"<figure>{embed(e.file)}<figcaption>{esc(e.label)}</figcaption></figure>"
                       for e in response.evidence if e.kind == "overlay" and e.file)
    steps = "".join(
        f"<tr><td>{esc(s.step_id)}</td><td>{esc(s.tool)}</td><td>{esc(s.model or '-')}</td><td><code>{esc(s.params)}</code></td>"
        f"<td>{esc(s.status)}</td><td>{s.duration_s:.2f}s</td><td>{esc(s.error or '')}</td></tr>" for s in trace.steps)
    issues = "".join(f"<li><b>{esc(i.severity)}</b> [{esc(i.code)}] {esc(i.message)}</li>" for i in trace.validation) or "<li>none</li>"
    images = "".join(f"<li>#{i.index + 1} {esc(i.name)}: {esc(i.modality)}, {i.width}x{i.height} px, bands {esc(i.bands)}, "
                     f"CRS {esc(i.crs)}, date {esc(i.acquired)}</li>" for i in trace.images)
    confidence = response.confidence
    confidence_text = (f"{confidence.value if confidence.value is not None else 'n/a'} ({esc(confidence.method)})"
                       + (f"<br><i>{esc(confidence.note)}</i>" if confidence.note else "")) if confidence else "n/a"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>SatQuery report {esc(trace.run_id)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #ddd;padding:4px 6px;text-align:left}}
figure{{margin:12px 0}}.answer{{background:#f3f7ff;padding:12px;border-radius:6px}}</style></head><body>
<h1>SatQuery AI: analysis report</h1>
<p><b>Run</b> {esc(trace.run_id)} · {esc(trace.created_at)} · status <b>{esc(response.status)}</b></p>
<p><b>Query:</b> {esc(trace.query)}</p>
<p><b>Input configuration:</b> {esc(trace.input_config)} · <b>Task:</b> {esc(response.task)}
 · <b>Rule:</b> {esc(trace.intent.matched_rule if trace.intent else '-')}</p>
<div class="answer"><b>Answer.</b> {esc(response.answer)}</div>
<p><b>Confidence:</b> {confidence_text}</p>
<h2>Visual evidence</h2>{overlays or '<p>none</p>'}
<h2>Execution trace</h2><table><tr><th>Step</th><th>Tool</th><th>Model</th><th>Parameters</th><th>Status</th><th>Time</th><th>Note</th></tr>{steps}</table>
<h2>Inputs</h2><ul>{images}</ul><h2>Validation</h2><ul>{issues}</ul>
<p style="font-size:12px;color:#666">Heuristic confidences are uncalibrated. Full machine-readable trace: report.json.</p>
</body></html>"""
    html_path = run_dir / "report.html"
    html_path.write_text(page, encoding="utf-8")
    return str(html_path), str(json_path)
