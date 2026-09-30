"""Retire obsolete CoreML v1 packages during a stopped Påmin maintenance window."""

import argparse
import os
import re
import shutil
import stat
from pathlib import Path


def obsolete(models: Path) -> list[Path]:
    if models.name != "models":
        raise ValueError("pass the Påmin models directory, not a parent directory")
    root = models.resolve(strict=True)
    prepared = root / "prepared"
    if not prepared.exists():
        return []
    if prepared.is_symlink() or not prepared.is_dir():
        raise ValueError("models/prepared must be a real directory")
    return sorted(
        old
        for entry in prepared.iterdir()
        if re.fullmatch(r"[0-9a-f]{32}", entry.name)
        and entry.is_dir()
        and not entry.is_symlink()
        if (old := entry / "coreml-all-v1").is_dir() and not old.is_symlink()
    )


def logical_bytes(directory: Path) -> int:
    total = 0
    for root, _, files in os.walk(directory, followlinks=False):
        for name in files:
            entry = os.lstat(Path(root) / name)
            if stat.S_ISREG(entry.st_mode):
                total += entry.st_size
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, help="the local Påmin models directory")
    parser.add_argument("--apply", action="store_true", help="remove the listed v1 packages")
    parser.add_argument(
        "--all-processes-stopped",
        action="store_true",
        help="assert every process using this model directory has stopped",
    )
    args = parser.parse_args()
    if args.apply and not args.all_processes_stopped:
        parser.error("--apply requires --all-processes-stopped")
    try:
        packages = obsolete(args.models)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for package in packages:
        print(f"{package}: {logical_bytes(package)} logical bytes")
        if args.apply:
            shutil.rmtree(package)
    print(f"{'removed' if args.apply else 'found'} {len(packages)} v1 packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
