"""Download all images referenced in the Labelbox NDJSON export.

Works locally and in Google Colab.
Colab: upload the .ndjson (or mount Drive), set NDJSON/OUT_DIR below, run.
"""
import json
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    BASE = Path(__file__).resolve().parent / "data"
except NameError:  # Colab / notebook
    BASE = Path("data")

NDJSON = BASE / "Export  project - Invasive Species - 11_10_2026.ndjson"
OUT_DIR = BASE / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
with open(NDJSON, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)["data_row"]
            rows.append((d["external_id"], d["row_data"]))
print(f"Found {len(rows)} images")


def fetch(item, tries=6):
    name, url = item
    dest = OUT_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        return name, "skipped"
    part = dest.with_suffix(dest.suffix + ".part")
    err = None
    for i in range(tries):
        try:
            urllib.request.urlretrieve(url, part)
            part.replace(dest)
            return name, "ok"
        except Exception as e:
            err = e
            time.sleep(2 * (i + 1))
    return name, f"FAILED: {err}"


with ThreadPoolExecutor(3) as ex:
    results = list(ex.map(fetch, rows))

failed = [r for r in results if r[1].startswith("FAILED")]
print(f"Done: {len(results) - len(failed)} ok, {len(failed)} failed -> {OUT_DIR}")
for n, s in failed:
    print(n, s)
