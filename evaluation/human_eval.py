"""Prepare blinded human-rating sheets from ablation traces and summarize ratings.

Create randomized rater CSVs:
    python -m evaluation.human_eval create outputs/ablation/groq_pilot \
        --out-dir outputs/ablation/groq_pilot/human_eval --raters 3 --seed 41

After collecting ratings, put completed CSV files in that folder and run:
    python -m evaluation.human_eval summarize outputs/ablation/groq_pilot/human_eval

Raters see neutral and edited dialogue, but not the system/variant name. Ratings
are ordinal 1-5. The summary includes descriptive means and Krippendorff alpha
for ordinal ratings when at least two raters have overlapping item ratings.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

RATING_FIELDS = (
    "meaning_preservation",
    "naturalness",
    "slang_appropriateness",
    "character_tone_fit",
)
VARIANT_LABELS = {
    "full": "Full system",
    "no_eligibility_gates": "No eligibility gates",
    "no_gate1": "No Gate 1 (speaker)",
    "no_gate2": "No Gate 2 (pragmatic)",
    "no_gate3": "No Gate 3 (density)",
    "no_validation_redo": "No validation/redo",
    "no_editor": "No editor",
}


def _turns(dialogue: str) -> list[tuple[str, str]]:
    turns = []
    for raw in (dialogue or "").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        speaker, text = line.split(":", 1)
        speaker, text = speaker.strip(), text.strip()
        if text:
            turns.append((speaker, text))
    return turns


def _read_trace(path: Path) -> list[dict[str, Any]]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    neutral_by_id = {s["scene_id"]: s for s in trace.get("neutral_script", [])}
    variant = trace.get("ablation", {}).get("variant") or path.stem.rsplit("_", 1)[-1]
    rows = []
    for scene in trace.get("slang_script", []):
        sid = scene.get("scene_id", "")
        neutral = neutral_by_id.get(sid, {})
        neutral_turns = _turns(neutral.get("dialogue", ""))
        edited_turns = _turns(scene.get("dialogue", ""))
        count = max(len(neutral_turns), len(edited_turns))
        for index in range(count):
            speaker = (edited_turns[index][0] if index < len(edited_turns)
                       else neutral_turns[index][0])
            neutral_text = neutral_turns[index][1] if index < len(neutral_turns) else ""
            edited_text = edited_turns[index][1] if index < len(edited_turns) else ""
            if not neutral_text and not edited_text:
                continue
            item_id = f"{path.stem}:{sid}:turn{index + 1}"
            rows.append({
                "item_id": item_id,
                "variant_hidden": variant,
                "source_trace": path.name,
                "scene_id": sid,
                "turn_index": index + 1,
                "speaker": speaker,
                "neutral_dialogue": neutral_text,
                "edited_dialogue": edited_text,
            })
    return rows


def create_sheets(input_dir: Path, out_dir: Path, raters: int, seed: int) -> None:
    traces = sorted(input_dir.glob("orch_full_trace_*.json"))
    if not traces:
        # Support arbitrary source trace stems: accept trace-compatible ablation files,
        # excluding reports/metrics/manifest.
        traces = [p for p in sorted(input_dir.glob("*.json"))
                  if "manifest" not in p.stem and not p.stem.startswith("metrics_")]
    if not traces:
        raise ValueError(f"No ablation trace JSON files found in {input_dir}")

    all_rows = []
    for path in traces:
        all_rows.extend(_read_trace(path))
    if not all_rows:
        raise ValueError("No dialogue turns found in the ablation trace files")

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest = {"seed": seed, "raters": raters, "items": len(all_rows),
                "rating_scale": "1-5; higher is better", "files": []}
    for rater in range(1, raters + 1):
        rows = [dict(row) for row in all_rows]
        rng.shuffle(rows)
        for row in rows:
            row["rater_id"] = f"R{rater:02d}"
            for field in RATING_FIELDS:
                row[field] = ""
            row["comment"] = ""
        dest = out_dir / f"rater_{rater:02d}.csv"
        fields = ["item_id", "rater_id", "scene_id", "turn_index", "speaker",
                  "neutral_dialogue", "edited_dialogue", *RATING_FIELDS, "comment"]
        with dest.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        manifest["files"].append(dest.name)

    # Keep the condition key separate from files given to raters.
    key_path = out_dir / "condition_key.json"
    key = {row["item_id"]: {"variant": row["variant_hidden"],
                             "source_trace": row["source_trace"]}
           for row in all_rows}
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "human_eval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {raters} blinded rating sheets with {len(all_rows)} paired turns each in {out_dir}")
    print("Do not give condition_key.json to raters. Share only their individual rater CSV.")


def _ordinal_alpha(items: list[dict[str, int]]) -> float | None:
    """Krippendorff alpha for ordinal data using squared normalized distance."""
    pairs = [(value, other) for item in items
             for value, other in combinations(item.values(), 2)]
    if not pairs:
        return None
    ratings = [value for item in items for value in item.values()]
    if len(ratings) < 2:
        return None
    levels = sorted(set(ratings))
    if len(levels) < 2:
        return 1.0
    low, high = min(levels), max(levels)
    scale = max(high - low, 1)
    observed = sum(((a - b) / scale) ** 2 for a, b in pairs) / len(pairs)
    expected_pairs = [(a, b) for a, b in combinations(ratings, 2)]
    if not expected_pairs:
        return None
    expected = sum(((a - b) / scale) ** 2 for a, b in expected_pairs) / len(expected_pairs)
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return 1 - observed / expected


def summarize(in_dir: Path, out_path: Path | None = None) -> dict:
    rating_files = [p for p in sorted(in_dir.glob("rater_*.csv"))
                    if p.stem[6:].isdigit()]
    if len(rating_files) < 2:
        raise ValueError("Need at least two completed rater_XX.csv files to summarize agreement")
    key_path = in_dir / "condition_key.json"
    if not key_path.exists():
        raise ValueError(f"Missing blinded condition key: {key_path}")
    condition_key = json.loads(key_path.read_text(encoding="utf-8"))

    raw: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for path in rating_files:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                item_id, rater_id = row.get("item_id", "").strip(), row.get("rater_id", "").strip()
                if not item_id or item_id not in condition_key:
                    continue
                for field in RATING_FIELDS:
                    value = row.get(field, "").strip()
                    if not value:
                        continue
                    try:
                        score = int(value)
                    except ValueError:
                        continue
                    if 1 <= score <= 5:
                        raw[(item_id, field)][rater_id] = score

    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    agreements: dict[str, list[dict[str, int]]] = defaultdict(list)
    item_scores: dict[tuple[str, str], list[int]] = {}
    for (item_id, field), per_rater in raw.items():
        if not per_rater:
            continue
        variant = condition_key[item_id]["variant"]
        grouped[variant][field].extend(per_rater.values())
        agreements[field].append(per_rater)
        item_scores[(item_id, field)] = list(per_rater.values())

    report = {
        "n_raters": len(rating_files),
        "n_dialogue_turns": len({item_id for item_id, _ in raw}),
        "scale": "ordinal 1-5; higher is better",
        "by_variant": {},
        "ordinal_krippendorff_alpha": {},
    }
    for variant, fields in grouped.items():
        report["by_variant"][variant] = {
            field: {"mean": round(mean(values), 3), "n_ratings": len(values)}
            for field, values in fields.items() if values
        }
    for field, items in agreements.items():
        report["ordinal_krippendorff_alpha"][field] = _ordinal_alpha(items)

    out_path = out_path or in_dir / "human_eval_summary.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved human-rating summary: {out_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create", help="create blinded CSV sheets for raters")
    create_parser.add_argument("input_dir", help="folder containing ablation trace JSONs")
    create_parser.add_argument("--out-dir", default=None)
    create_parser.add_argument("--raters", type=int, default=3)
    create_parser.add_argument("--seed", type=int, default=41)
    summary_parser = sub.add_parser("summarize", help="summarize completed blinded rating sheets")
    summary_parser.add_argument("input_dir", help="folder containing completed rater CSVs and condition key")
    summary_parser.add_argument("--out", default=None)
    args = parser.parse_args()

    try:
        if args.command == "create":
            if args.raters < 2:
                parser.error("use at least two raters to estimate agreement")
            source = Path(args.input_dir)
            destination = Path(args.out_dir) if args.out_dir else source / "human_eval"
            create_sheets(source, destination, args.raters, args.seed)
        else:
            summarize(Path(args.input_dir), Path(args.out) if args.out else None)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())