"""Rebuild everything in clean/ from source/. Usage: python pipeline/run_all.py"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["00_ct_crosswalk.py", "01_counties.py", "02_tracks.py", "03_storm_county.py",
         "04_loss_records.py", "05_loss_match.py", "06_losses.py", "07_final.py"]

for step in STEPS:
    path = HERE / step
    if not path.exists():
        print(f"--- {step}: not present, skipped")
        continue
    print(f"--- {step}")
    t = time.time()
    r = subprocess.run([sys.executable, str(path)], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"!!! {step} failed; stopping.")
    print(f"    ({time.time() - t:.0f}s)")
print("Done. Logs in clean/logs/")
