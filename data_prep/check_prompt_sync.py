"""
data_prep/check_prompt_sync.py
Verify the v2 prompt strings are IDENTICAL in the three places that must agree:
  1. data_prep/build_lora_dataset.py        (what the training data contains)
  2. notebooks/train_slang_editor_lora_qwen_v3.py   (what the Kaggle notebook asserts/uses)
  3. agents/agent4_slang_rewriter.py        (what the pipeline sends at inference)
A mismatch means the adapter is trained on one prompt and queried with another, which
silently degrades output. Also verifies the dataset file on disk uses the v2 system prompt.

Usage: python -m data_prep.check_prompt_sync
"""
import ast, json, os, sys

NAMES = ("SYSTEM_V2", "USER_V2")


def from_source(path, names):
    """Read module-level string constants out of a file WITHOUT executing it."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
           and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            found[node.targets[0].id] = ast.literal_eval(node.value)
    return found


def main():
    builder = from_source("data_prep/build_lora_dataset.py", NAMES)
    notebook = from_source("notebooks/train_slang_editor_lora_qwen_v3.py", NAMES)
    agent = from_source("agents/agent4_slang_rewriter.py",
                        ("QWEN_SYSTEM_V2", "QWEN_USER_V2"))
    agent = {"SYSTEM_V2": agent.get("QWEN_SYSTEM_V2"), "USER_V2": agent.get("QWEN_USER_V2")}

    ok = True
    for n in NAMES:
        vals = {"builder": builder.get(n), "notebook": notebook.get(n), "agent4": agent.get(n)}
        same = len({v for v in vals.values()}) == 1 and None not in vals.values()
        print(f"{'OK  ' if same else 'FAIL'} {n}: builder/notebook/agent4 "
              f"{'identical' if same else 'DIFFER'}")
        if not same:
            ok = False
            for k, v in vals.items():
                print(f"       {k}: {v!r}")

    p = "data/felix_dataset/v3/lora_train.jsonl"
    if os.path.exists(p):
        first = json.loads(open(p, encoding="utf-8").readline())
        s = first["messages"][0]["content"]
        same = s == builder["SYSTEM_V2"]
        print(f"{'OK  ' if same else 'FAIL'} dataset on disk uses the v2 system prompt")
        ok = ok and same
    print("\nALL IN SYNC" if ok else "\nOUT OF SYNC - fix before training/running")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
