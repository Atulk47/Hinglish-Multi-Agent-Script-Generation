"""
evaluation/bench_editor.py
Time the Qwen LoRA slang editor on 3 lines WITHOUT running the pipeline - to see what
precision/speed this machine can do. Respects QWEN_DTYPE (auto | float32 | bfloat16).

PowerShell:
    python -m evaluation.bench_editor                       # whatever "auto" picks
    $env:QWEN_DTYPE="float32";  python -m evaluation.bench_editor   # force fp32 (needs ~6 GB free)
    $env:QWEN_DTYPE="bfloat16"; python -m evaluation.bench_editor
    Remove-Item Env:QWEN_DTYPE                              # back to auto
Tip: close Chrome / VS Code first and run from a separate terminal for the fp32 test.
"""
import time
from memory.scratchpad import Scratchpad
from agents.agent4_slang_rewriter import SlangRewriter


class _Clf:
    def classify(self, t):
        return "CASUAL_ASSERTION"


def main():
    t0 = time.time()
    sr = SlangRewriter(Scratchpad("bench_editor"), _Clf(), backend="qwen")
    if sr.backend != "qwen":
        print("Editor did not load as Qwen (fell back to", sr.backend, ") - see the message above.")
        return
    print(f"load + smoke test: {time.time() - t0:.0f}s\n")
    lines = ["Aaj ka match bahut accha tha.",
             "Yeh kaam baar baar bigad raha hai.",
             "Humari team final mein pahunch gayi."]
    sl = ["jhakaas", "mast", "solid", "kadak", "bindaas", "pak gaya", "dhamaal", "waat lag gayi"]
    times = []
    for ln in lines:
        t = time.time()
        out = sr._qwen_edit(ln, sl[: sr._qwen_k])
        times.append(time.time() - t)
        print(f"  [{times[-1]:5.1f}s] {ln}\n           -> {out}")
    print(f"\naverage {sum(times) / len(times):.1f}s per line "
          f"(~{sum(times) / len(times) * 8 / 60:.1f} min for an 8-line scene run, before retries)")


if __name__ == "__main__":
    main()
