"""Create a self-contained HTML graph/report from ablation evaluation JSON files.

Example:
    python -m evaluation.plot_ablation outputs/ablation/groq_pilot --open

Reads metrics_<variant>.json files written by evaluation.evaluate, creates
ablation_report.html plus a CSV summary, and uses only the Python standard
library (no matplotlib/plotly installation needed).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import webbrowser

VARIANT_LABELS = {
    "full": "Full system",
    "no_eligibility_gates": "No eligibility gates",
    "no_gate1": "No Gate 1 (speaker)",
    "no_gate2": "No Gate 2 (pragmatic)",
    "no_gate3": "No Gate 3 (density)",
    "no_validation_redo": "No validation/redo",
    "no_editor": "No editor",
}

CHART_METRICS = (
    ("slang_coverage", "Slang coverage", "#2563eb"),
    ("event_recall_mean", "Event recall", "#16a34a"),
    ("slang_diversity", "Slang diversity", "#9333ea"),
    ("slang_validity", "Slang validity", "#ea580c"),
    ("changed_line_semantic_sim_mean", "Changed-line MuRIL similarity", "#0891b2"),
)
TABLE_METRICS = (
    ("slang_coverage", "Slang coverage"),
    ("event_recall_mean", "Event recall"),
    ("changed_line_count", "Changed lines scored"),
    ("changed_line_semantic_sim_mean", "Changed-line MuRIL similarity"),
    ("changed_line_semantic_sim_min", "Lowest changed-line MuRIL similarity"),
    ("slang_validity", "Slang validity"),
    ("slang_diversity", "Slang diversity"),
    ("semantic_sim_mean", "MuRIL similarity (coarse)"),
    ("slang_total", "Detected slang total"),
    ("n_scenes", "Scenes"),
)


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _load_reports(input_dir: Path) -> list[tuple[str, dict]]:
    reports = []
    for path in input_dir.glob("metrics_*.json"):
        variant = path.stem.removeprefix("metrics_")
        if variant.endswith("_linewise"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read {path}: {exc}") from exc
        reports.append((variant, data))
    order = {name: index for index, name in enumerate(VARIANT_LABELS)}
    reports.sort(key=lambda item: (order.get(item[0], len(order)), item[0]))
    return reports


def _make_chart(reports: list[tuple[str, dict]]) -> str:
    width, height = 960, 430
    left, right, top, bottom = 76, 24, 48, 96
    plot_height = height - top - bottom
    plot_width = width - left - right
    group_width = plot_width / max(len(reports), 1)
    bar_width = min(34, group_width / (len(CHART_METRICS) + 2))
    gap = 5
    bars_width = len(CHART_METRICS) * bar_width + (len(CHART_METRICS) - 1) * gap
    chunks = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Grouped bar chart comparing ablation metrics from zero to one">',
        f'<text x="{left}" y="24" class="axis-title">Score (0–1; higher is better)</text>',
    ]

    for step in range(5):
        score = step / 4
        y = top + plot_height * (1 - score)
        chunks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        chunks.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="tick">{score:.2f}</text>')

    for index, (variant, data) in enumerate(reports):
        center = left + group_width * (index + 0.5)
        start = center - bars_width / 2
        for metric_index, (key, label, color) in enumerate(CHART_METRICS):
            value = data.get(key)
            if value is None:
                continue
            value = min(max(float(value), 0.0), 1.0)
            bar_height = value * plot_height
            x = start + metric_index * (bar_width + gap)
            y = top + plot_height - bar_height
            chunks.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" rx="4" fill="{color}">'
                f'<title>{html.escape(label)}: {value:.3f}</title></rect>'
            )
            chunks.append(
                f'<text x="{x + bar_width/2:.1f}" y="{max(y-7, top+12):.1f}" '
                f'text-anchor="middle" class="value">{value:.2f}</text>'
            )
        label = html.escape(VARIANT_LABELS.get(variant, variant.replace("_", " ")))
        chunks.append(
            f'<text x="{center:.1f}" y="{top + plot_height + 25}" '
            f'text-anchor="middle" class="variant">{label}</text>'
        )

    legend_y = height - 19
    legend_width = len(CHART_METRICS) * 190
    legend_x = max(left, (width - legend_width) / 2)
    for index, (_, label, color) in enumerate(CHART_METRICS):
        x = legend_x + index * 190
        chunks.append(f'<rect x="{x:.1f}" y="{legend_y-12}" width="12" height="12" rx="2" fill="{color}"/>')
        chunks.append(f'<text x="{x+18:.1f}" y="{legend_y-2}" class="legend">{html.escape(label)}</text>')
    chunks.append("</svg>")
    return "\n".join(chunks)


def _make_table(reports: list[tuple[str, dict]]) -> str:
    head = "<thead><tr><th>Variant</th>" + "".join(
        f"<th>{html.escape(label)}</th>" for _, label in TABLE_METRICS
    ) + "</tr></thead>"
    rows = []
    for variant, data in reports:
        cells = [f"<th>{html.escape(VARIANT_LABELS.get(variant, variant))}</th>"]
        cells.extend(f"<td>{_fmt(data.get(key))}</td>" for key, _ in TABLE_METRICS)
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + head + "<tbody>" + "".join(rows) + "</tbody></table>"


def _write_csv(reports: list[tuple[str, dict]], path: Path) -> None:
    fields = ["variant"] + [key for key, _ in TABLE_METRICS]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant, data in reports:
            writer.writerow({"variant": VARIANT_LABELS.get(variant, variant),
                             **{key: data.get(key) for key, _ in TABLE_METRICS}})


def build_report(input_dir: Path, output_path: Path) -> Path:
    reports = _load_reports(input_dir)
    if not reports:
        raise ValueError(f"No metrics_*.json files found in {input_dir}")

    rows = _make_table(reports)
    chart = _make_chart(reports)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hinglish ablation results</title>
<style>
:root {{ color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; color: #172033; background: #f4f7fb; }}
body {{ max-width: 1120px; margin: 32px auto; padding: 0 22px 40px; }}
header, section {{ background: white; border: 1px solid #e3e8f0; border-radius: 14px; padding: 22px 26px; margin-bottom: 18px; box-shadow: 0 4px 18px #1720330b; }}
h1 {{ margin: 0 0 8px; font-size: 1.7rem; }}
p {{ line-height: 1.55; color: #526078; }}
.chart {{ display: block; width: 100%; height: auto; overflow: visible; }}
.grid {{ stroke: #e6ebf2; stroke-width: 1; }} .tick, .axis-title, .legend {{ fill: #526078; font-size: 12px; }}
.value {{ fill: #263248; font-size: 10px; font-weight: 700; }} .variant {{ fill: #263248; font-size: 12px; font-weight: 600; }}
table {{ width: 100%; border-collapse: collapse; font-size: .92rem; }}
th, td {{ text-align: left; padding: 11px 9px; border-bottom: 1px solid #e9edf3; }}
thead th {{ color: #526078; font-size: .82rem; }} tbody th {{ white-space: nowrap; }}
.note {{ background: #fff7e6; border-left: 4px solid #f59e0b; padding: 12px 15px; border-radius: 5px; }}
@media (max-width: 720px) {{ body {{ padding: 0 10px; }} section {{ padding: 16px; overflow-x: auto; }} table {{ min-width: 780px; }} }}
</style>
</head>
<body>
<header>
<h1>Hinglish editor ablation results</h1>
<p>Descriptive comparison of the generated variants. The chart shows coverage, event recall, diversity, validity, and MuRIL similarity of changed dialogue lines on a 0–1 scale. Changed-line similarity excludes unchanged turns that can inflate whole-scene scores. Validity or changed-line similarity is omitted where undefined.</p>
</header>
<section><h2>Metric comparison</h2>{chart}</section>
<section><h2>Detailed results</h2>{rows}
<p class="note"><strong>Interpret carefully:</strong> these are descriptive metrics, not statistical significance tests. The no-editor validity is N/A, not zero or perfect. MuRIL similarity is listed for reference but is coarse; use blind human ratings and more held-out stories for paper claims.</p>
</section>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    _write_csv(reports, csv_path)
    print(f"HTML graph/report: {output_path}")
    print(f"CSV summary:       {csv_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", default="outputs/ablation/groq_pilot",
                        help="directory containing metrics_<variant>.json files")
    parser.add_argument("--out", default=None, help="HTML report path (default: <input_dir>/ablation_report.html)")
    parser.add_argument("--open", action="store_true", help="open the generated HTML report in your browser")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.out) if args.out else input_dir / "ablation_report.html"
    try:
        report_path = build_report(input_dir, output_path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.open:
        webbrowser.open(report_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
