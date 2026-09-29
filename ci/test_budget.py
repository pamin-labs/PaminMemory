"""Run with python3 ci/test_budget.py; no build or third-party dependencies."""

import pathlib
import tempfile

from budget import require_runtime_library


with tempfile.TemporaryDirectory() as temporary:
    profile = pathlib.Path(temporary)
    try:
        require_runtime_library(profile)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a missing native runtime passed the distribution gate")
    for name in ("libzvec_c_api.dylib", "libzvec_c_api.so", "zvec_c_api.dll"):
        runtime = profile / name
        runtime.write_bytes(b"fixture")
        require_runtime_library(profile)
        runtime.unlink()
print("Native distribution gate rejects omission and accepts platform filenames.")
