#!/usr/bin/env python3
"""Measure the engineering budgets and fail when a gated one regresses.

Gated metrics are deterministic. Build times are reported only: shared CI
runners vary too much for wall-clock timing to be a reliable failure signal,
and a flaky gate teaches people to ignore gates.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUDGETS = ROOT / "ci" / "budgets.json"
BINARY = ROOT / "target" / "release" / "pamin"


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def dependency_count() -> int:
    out = subprocess.run(
        ("cargo", "metadata", "--format-version", "1"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    packages = json.loads(out.stdout)["packages"]
    workspace = {p["id"] for p in packages if p["manifest_path"].startswith(str(ROOT))}
    return len([p for p in packages if p["id"] not in workspace])


def projection_containment() -> list[str]:
    """Return the crates that reach `zvec`, so the boundary stays a rule.

    The architecture decision that picked `zvec` mitigates its pre-1.0 risk by
    confining it: the engine appears only behind the projection boundary, and
    its types must not reach the domain layer. That was written as prose and
    was never enforced, so this checks it the way the SQL drift test checks
    the enum labels — mechanically, on every run.
    """
    out = subprocess.run(
        ("cargo", "tree", "--workspace", "--invert", "zvec-rust", "--prefix", "none"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    workspace = set()
    for line in out.stdout.splitlines():
        name = line.split(" ", 1)[0]
        if name.startswith("pamin-"):
            workspace.add(name)
    return sorted(workspace)


ALLOWED_TO_REACH_ZVEC = {"pamin-index", "pamin-engine", "pamin-cli"}

def directory_bytes(path: pathlib.Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# Extensions of a native library that ships beside the binary rather than
# inside it.
SIDECAR_SUFFIXES = (".so", ".dylib", ".dll")


def sidecar_bytes() -> int:
    """Return the bytes of the native libraries shipped beside the binary.

    The retrieval engine is a dynamic library. The linker records it by name,
    the build script copies it next to the binary, and an `$ORIGIN` rpath is
    what lets the pair be moved together -- so what a user downloads is both
    files, and `binary_bytes` alone has never been the shipped size. Measuring
    only the executable made every size comparison, including any comparison
    against a different retrieval engine, wrong by the larger of the two.
    """
    profile = BINARY.parent
    return sum(
        f.stat().st_size
        for f in profile.iterdir()
        if f.is_file() and f.suffix in SIDECAR_SUFFIXES
    )


def measure() -> dict[str, float]:
    target = ROOT / "target"
    if target.exists():
        shutil.rmtree(target)

    start = time.monotonic()
    run("cargo", "build", "--release", "--workspace")
    cold_build_seconds = time.monotonic() - start

    # A check straight after a build measures the first check, not an
    # incremental one. Touching the most-edited crate is what makes this the
    # number a developer actually waits on.
    run("cargo", "check", "--workspace")
    (ROOT / "crates" / "pamin-core" / "src" / "lib.rs").touch()

    start = time.monotonic()
    run("cargo", "check", "--workspace")
    incremental_check_seconds = time.monotonic() - start

    binary_bytes = BINARY.stat().st_size
    return {
        "binary_bytes": binary_bytes,
        "distribution_bytes": binary_bytes + sidecar_bytes(),
        "total_dependencies": dependency_count(),
        "cold_build_seconds": round(cold_build_seconds, 1),
        "incremental_check_seconds": round(incremental_check_seconds, 1),
        "target_bytes": directory_bytes(target),
    }


def main() -> int:
    budgets = json.loads(BUDGETS.read_text())
    measured = measure()

    print("\nEngineering budgets\n")
    failures = []
    for name, limit in budgets["gated"].items():
        value = measured[name]
        # A zero limit means the budget has not been set yet, so record the
        # measurement and let it pass rather than failing on an unset gate.
        ok = limit == 0 or value <= limit
        state = "unset" if limit == 0 else ("ok" if ok else "OVER")
        print(f"  {name:<26} {value:>12,}  limit {limit:>12,}  {state}")
        if not ok:
            failures.append(f"{name}: {value:,} exceeds budget {limit:,}")

    for name in budgets["reported"]:
        print(f"  {name:<26} {measured[name]:>12,}  (reported)")

    reaches_zvec = set(projection_containment())
    leaked = sorted(reaches_zvec - ALLOWED_TO_REACH_ZVEC)
    print(f"\n  zvec is reachable from: {', '.join(sorted(reaches_zvec)) or 'nothing'}")
    if leaked:
        failures.append(
            "zvec escaped the projection boundary and is now reachable from "
            + ", ".join(leaked)
        )

    summary = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if summary is not None:
        lines = ["| Metric | Value | Budget |", "| --- | ---: | ---: |"]
        for name, limit in budgets["gated"].items():
            lines.append(
                f"| {name} | {measured[name]:,} | {'unset' if limit == 0 else f'{limit:,}'} |"
            )
        for name in budgets["reported"]:
            lines.append(f"| {name} | {measured[name]:,} | reported |")
        summary.write_text("## Engineering budgets\n\n" + "\n".join(lines) + "\n")

    if failures:
        print("\nOver budget:")
        for failure in failures:
            print(f"  {failure}")
        print("\nRaise the budget deliberately and say why, or bring the cost back down.")
        return 1

    print("\nAll gated budgets within limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
