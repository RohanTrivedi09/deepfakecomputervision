"""Download the 140k Real and Fake Faces dataset via the Kaggle API.

Requires Kaggle credentials to already be configured (either ~/.kaggle/kaggle.json
or the KAGGLE_USERNAME / KAGGLE_KEY environment variables). On Colab this is set
up from Colab Secrets in the training notebook before this script runs.
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from src.common.paths import DATA_RAW

DATASET = "xhlulu/140k-real-and-fake-faces"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(DATA_RAW))
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if any(dest.iterdir()):
        print(f"{dest} is not empty, skipping download.")
        return

    print(f"Downloading {DATASET} to {dest} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(dest)],
        check=True,
    )

    zip_path = dest / "140k-real-and-fake-faces.zip"
    if zip_path.exists():
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        zip_path.unlink()

    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
