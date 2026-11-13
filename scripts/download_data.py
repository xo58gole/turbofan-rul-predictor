"""Download the three FD001 files and check they're the right ones.

NASA's original download page moves around, so we pull the files from a
public GitHub copy of the dataset and verify each one against a SHA-256
checksum. The checksums match files with the published sizes
(100 training engines / 20,631 rows, 100 test engines / 13,096 rows).

Usage:  python scripts/download_data.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
BASE_URL = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData"
FILES = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = DATA_DIR / name
        if path.exists() and sha256(path) == expected:
            print(f"ok      {name} (already downloaded)")
            continue
        print(f"get     {name} ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", path)
        if sha256(path) != expected:
            path.unlink()
            print(f"ERROR   {name} checksum doesn't match, file deleted")
            return 1
        print(f"ok      {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
