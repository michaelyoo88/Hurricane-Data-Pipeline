"""Rebuild everything in clean/ and final/ from source/ (about 2-3 minutes).

    python3 pipeline/run_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["00_ct_crosswalk.py", "01_counties.py", "01b_housing_by_year.py", "02_tracks.py", "03_storm_county.py",
         "04_loss_records.py", "05_losses.py", "06_final_tables.py", "07_county_year.py",
         "data_dictionary.py"]

for step in STEPS:
    print(f"--- {step}")
    subprocess.run([sys.executable, str(HERE / step)], cwd=HERE, check=True)
print("Done: tables are in final/")
