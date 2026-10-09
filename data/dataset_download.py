from pathlib import Path
import subprocess
import zipfile
import os

DATA_DIR = Path("isic2024")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=== Downloading ISIC 2024 ===")

subprocess.run([
    "py", "-m", "kaggle",
    "competitions", "download",
    "-c", "isic-2024-challenge",
    "-p", str(DATA_DIR)
], check=True)

print("\n=== Extracting files ===")

zip_files = list(DATA_DIR.glob("*.zip"))

for zip_path in zip_files:
    print(f"\nExtracting: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print(f"Extracted: {zip_path.name}")

    # Удаляем ZIP после успешной распаковки
    zip_path.unlink()
    print(f"Deleted: {zip_path.name}")

print("\n=== DONE ===")

for file in DATA_DIR.iterdir():
    if file.is_file():
        size_gb = file.stat().st_size / (1024 ** 3)
        print(f"{file.name:40} {size_gb:.2f} GB")