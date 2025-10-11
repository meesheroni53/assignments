import subprocess
subprocess.check_call(["pip", "install", "kaggle"])
import os
import subprocess
import zipfile
from pathlib import Path

COMPETITION = "spring-2025-classification-competition"

# Paths
script_dir = Path(__file__).resolve().parent
os.environ['KAGGLE_CONFIG_DIR'] = str(script_dir)  # kaggle.json is located here
rawdata_path = script_dir.parent / "RawData"
kaggle_json_path = script_dir / "kaggle.json"
rawdata_path.mkdir(parents=True, exist_ok=True)

# Skip download if CSV files already exist
if any(rawdata_path.glob("*.csv")):
    print("Data files already exist, skipping download.")
else:
    print(f"Downloading data for {COMPETITION} into {rawdata_path}...")
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", COMPETITION,
        "-p", str(rawdata_path)
    ], check=True)

    # Extract all zip files and then delete the zip file after the excel files are extracted
    for zip_file in rawdata_path.glob("*.zip"):
        print(f"Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(rawdata_path)
        zip_file.unlink()  
        print(f"Extracted and removed {zip_file.name}")

