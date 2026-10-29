"""
tools/import_maya_zip.py  -  puts the clips rendered by a Kaggle batch notebook (maya_batch.zip) into the local TTS cache.

    python -m tools.import_maya_zip path/to/maya_batch.zip
After that,  `python -m audio.synthesize --trace ... --tts maya`  or the "Maya" voices in the UI play these stories with no GPU and no network.
"""
from __future__ import annotations
import json
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import soundfile as sf

from audio.tts import MAYA_CACHE

NAME = re.compile(r"^m[0-9a-f]{20}\.wav$")


def import_zip(path: str, dest: str = MAYA_CACHE) -> dict:
    os.makedirs(dest, exist_ok=True)
    res = {"imported": 0, "skipped_existing": 0, "bad": [], "kaggle_missing": []}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            base = os.path.basename(n)
            if base == "batch_log.json":
                try:
                    res["kaggle_missing"] = json.loads(z.read(n)).get("missing", [])
                except Exception:
                    pass
                continue
            if not NAME.match(base):
                continue
            target = os.path.join(dest, base)
            if os.path.exists(target) and os.path.getsize(target) > 2000:
                res["skipped_existing"] += 1
                continue
            data = z.read(n)
            tmp = target + ".part"
            open(tmp, "wb").write(data)
            try:
                ok = sf.info(tmp).duration >= 0.2
            except Exception:
                ok = False
            if ok:
                os.replace(tmp, target)
                res["imported"] += 1
            else:
                os.remove(tmp)
                res["bad"].append(base)
    return res


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    r = import_zip(sys.argv[1])
    print(f"imported {r['imported']} clips ({r['skipped_existing']} were already cached); unreadable: {len(r['bad'])}; "
          f"not rendered on Kaggle: {len(r['kaggle_missing'])}")
    if r["bad"] or r["kaggle_missing"]:
        print("  -> run  tools.export_maya_jobs  again for the same stories: it will list only the lines still missing.")


if __name__ == "__main__":
    main()
