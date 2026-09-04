#!/usr/bin/env python3
"""Verify the seven canonical hazard files against SHA256SUMS.txt."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    raw_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else package_root / "raw"
    checksum_path = package_root / "SHA256SUMS.txt"
    failed = False

    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        expected, filename = line.split(maxsplit=1)
        path = raw_dir / filename.strip()
        if not path.exists():
            print(f"MISSING  {path}")
            failed = True
            continue
        actual = sha256(path)
        if actual == expected:
            print(f"OK       {path.name}")
        else:
            print(f"FAILED   {path.name}: expected={expected}, actual={actual}")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
