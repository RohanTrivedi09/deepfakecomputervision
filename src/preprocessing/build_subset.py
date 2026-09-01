"""Build a stratified real/fake subset from the downloaded dataset into data/processed/.

Locates train/valid/test x real/fake source directories anywhere under data/raw/
(the Kaggle dataset nests them a few levels deep, e.g. under
real_vs_fake/real-vs-fake/<split>/<class>/, and class/split folder names may be
any case, e.g. REAL/FAKE), then copies a random, seeded sample of each into
data/processed/<split>/<class>/.
"""
import argparse
import random
import shutil
from pathlib import Path

from src.common.paths import DATA_PROCESSED, DATA_RAW

SPLITS = ("train", "valid", "test")
CLASSES = ("real", "fake")
SPLIT_ALIASES = {
    "train": "train",
    "valid": "valid",
    "validation": "valid",
    "val": "valid",
    "test": "test",
}


def find_split_dirs(raw_root: Path) -> dict:
    """Walk every directory under raw_root and match by lowercase name, so this
    is robust to whatever exact casing/nesting the Kaggle archive extracts to.
    """
    found = {split: {} for split in SPLITS}
    for p in raw_root.rglob("*"):
        if not p.is_dir():
            continue
        cls = p.name.lower()
        if cls not in CLASSES:
            continue
        split = SPLIT_ALIASES.get(p.parent.name.lower())
        if split is None:
            continue
        found[split].setdefault(cls, p)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class-train", type=int, default=10000)
    parser.add_argument("--per-class-valid", type=int, default=1000)
    parser.add_argument("--per-class-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    counts = {
        "train": args.per_class_train,
        "valid": args.per_class_valid,
        "test": args.per_class_test,
    }

    split_dirs = find_split_dirs(DATA_RAW)

    if not any(split_dirs[split] for split in SPLITS):
        print(f"No train/valid/test x real/fake directories found anywhere under {DATA_RAW}.")
        print("Directory tree found instead (up to 4 levels deep):")
        for p in sorted(DATA_RAW.rglob("*")):
            if p.is_dir() and len(p.relative_to(DATA_RAW).parts) <= 4:
                print(" ", p.relative_to(DATA_RAW))
        raise SystemExit(
            "Aborting: nothing to subsample. Check that download_data.py actually "
            "downloaded/extracted the dataset, and that the split/class folder names "
            "above look like train/valid/test and real/fake (any case)."
        )

    for split in SPLITS:
        for cls in CLASSES:
            src_dir = split_dirs[split].get(cls)
            if src_dir is None:
                print(f"WARNING: could not find {split}/{cls} under {DATA_RAW}, skipping.")
                continue

            images = sorted(src_dir.glob("*.*"))
            random.shuffle(images)
            n = min(counts[split], len(images))
            chosen = images[:n]

            dest_dir = DATA_PROCESSED / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in chosen:
                shutil.copy2(img, dest_dir / img.name)
            print(f"{split}/{cls}: copied {len(chosen)} of {len(images)} images -> {dest_dir}")


if __name__ == "__main__":
    main()
