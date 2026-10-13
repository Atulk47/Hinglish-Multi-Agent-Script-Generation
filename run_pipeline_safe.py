"""
run_pipeline_safe.py - run run_pipeline.py and automatically retry if it dies with the intermittent native crash
(0xC0000005 inside torch_cpu.dll while the editor model loads). The crash happens within the first minute, before any
expensive work, so a retry costs ~1 min instead of failing the whole run.

    python run_pipeline_safe.py --story data/raw/demo_science_fair.txt --session final_a
(accepts exactly the same arguments as run_pipeline.py)
"""
import subprocess
import sys

CRASH_CODES = {3221225477, -1073741819}      # 0xC0000005 as unsigned / signed
MAX_TRIES = 4

for attempt in range(1, MAX_TRIES + 1):
    code = subprocess.call([sys.executable, "-u", "run_pipeline.py", *sys.argv[1:]])
    if code not in CRASH_CODES:
        sys.exit(code)
    print(f"\n[run_pipeline_safe] native crash on attempt {attempt}/{MAX_TRIES} - retrying ...\n", flush=True)
sys.exit(1)
