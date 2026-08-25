"""
Download NSL-KDD, UNSW-NB15, and CIC-IDS-2017 datasets.
Run: python -m datasets.download [--dataset all|nsl-kdd|unsw-nb15|cic-ids-2017]
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "database"

DATASETS = {
    "nsl-kdd": {
        "files": {
            "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
            "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
        },
    },
    "unsw-nb15": {
        "files": {
            "UNSW_NB15_training-set.csv": "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv",
            "UNSW_NB15_testing-set.csv": "https://raw.githubusercontent.com/oshoyemi/project/master/UNSW_NB15_testing-set.csv",
        },
    },
    "cic-ids-2017": {
        "zip_url": "https://cse-cnc.ca/publications/2017/2017-004-ML-CIC-IDS-2017.zip",
        "extract_glob": "**/MachineLearningCVE/*.csv",
    },
}


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(100, block_num * block_size * 100 / total_size)
    sys.stdout.write(f"\r  {pct:5.1f}%")
    sys.stdout.flush()


def download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  skip (exists): {dest.name}")
        return True
    print(f"  downloading: {dest.name}")
    try:
        urlretrieve(url, dest, _progress)
        print()
        return True
    except Exception as exc:
        print(f"\n  FAILED: {exc}")
        return False


def download_nsl_kdd():
    print("\n[NSL-KDD]")
    ok = True
    for name, url in DATASETS["nsl-kdd"]["files"].items():
        ok &= download_file(url, DB_DIR / name)
    return ok


def download_unsw_nb15():
    print("\n[UNSW-NB15]")
    ok = True
    for name, url in DATASETS["unsw-nb15"]["files"].items():
        ok &= download_file(url, DB_DIR / name)
    return ok


def download_cic_ids_2017():
    print("\n[CIC-IDS-2017] (~350 MB zip, may take several minutes)")
    zip_path = DB_DIR / "CIC-IDS-2017.zip"
    extract_dir = DB_DIR / "CIC-IDS-2017"

    if extract_dir.exists() and any(extract_dir.rglob("*.csv")):
        print("  skip (already extracted)")
        return True

    if not zip_path.exists():
        if not download_file(DATASETS["cic-ids-2017"]["zip_url"], zip_path):
            print("  CIC-IDS-2017 download failed — try manual download from:")
            print("  https://www.unb.ca/cic/datasets/ids-2017.html")
            return False

    print("  extracting ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("  done")
        return True
    except Exception as exc:
        print(f"  extract FAILED: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download NIDS datasets")
    parser.add_argument(
        "--dataset",
        choices=["all", "nsl-kdd", "unsw-nb15", "cic-ids-2017"],
        default="all",
    )
    args = parser.parse_args()

    DB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {DB_DIR}")

    tasks = {
        "nsl-kdd": download_nsl_kdd,
        "unsw-nb15": download_unsw_nb15,
        "cic-ids-2017": download_cic_ids_2017,
    }

    targets = list(tasks.keys()) if args.dataset == "all" else [args.dataset]
    results = {name: tasks[name]() for name in targets}

    print("\n" + "=" * 50)
    for name, ok in results.items():
        print(f"  {name:15s} {'OK' if ok else 'FAILED'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
