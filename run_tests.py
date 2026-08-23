"""
run_tests.py
============
Runs every test_*.py in this directory, in one process invocation, with a
single pass/fail summary. This is what CI calls, and what you should call
locally before pushing anything that touches decision.py, safety.py,
classifier.py, auth.py, or db.py.

    venv\\Scripts\\python.exe run_tests.py

Each test_*.py file is self-contained (own stubs, own fixtures, own
sys.exit(1) on failure) and was written to be run standalone during
development. This runner does not change that -- it just also runs them in
subprocesses, so a sys.exit(1) inside one test file doesn't kill the whole
suite before the others get a chance to report.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = HERE / "venv" / "Scripts" / "python.exe"
if not PY.exists():
    # CI runners don't have this repo's local venv -- they set up their own
    # Python and it's just "python" on PATH. Local dev still prefers the venv
    # so it uses exactly the pinned dependency versions.
    PY = Path(sys.executable)

# Ordered deliberately: cheapest/most-isolated first, so a fast failure shows
# up before the slower ones (ONNX export validation, ~10s) run at all.
TEST_FILES = [
    "test_decision.py",       # pure logic, no I/O -- the safety-critical core
    "test_signal_floor.py",   # the "hi" bug and its fix
    "test_informational.py",  # "tel me about ptsd symptoms" -> trauma probe
    "test_reply_integrity.py",  # repeated question + half-sentence replies
    "test_ratelimit.py",
    "test_auth.py",
    "test_persistence.py",
    "test_onnx_path.py",      # needs the real model artifacts + onnxruntime
]

# Tests in here are SKIPPED (not failed) when their precondition isn't met.
# artifacts/*.keras and *.onnx are gitignored (see .gitignore) -- ~100MB of
# binary model files with no business in git history -- so a fresh clone, and
# every CI run unless the artifacts are fetched by a separate step, will not
# have them. Without this, every clone and every CI run would report a
# failure that has nothing to do with anyone's code change.
CONDITIONAL: dict[str, callable] = {
    "test_onnx_path.py": lambda: (
        (HERE / "artifacts" / "model_v5.onnx").exists()
        or (HERE / "artifacts" / "model_v5.keras").exists()),
}


def main() -> int:
    results = []
    t0 = time.time()
    for name in TEST_FILES:
        path = HERE / name
        if not path.exists():
            print(f"SKIP  {name} (file not found)")
            results.append((name, None, 0, "file not found"))
            continue

        precondition = CONDITIONAL.get(name)
        if precondition and not precondition():
            print(f"SKIP  {name} (model artifacts not present -- see "
                  f"artifacts/ in .gitignore. Run export_onnx.py after "
                  f"copying model_v5.keras in to enable this test.)")
            results.append((name, None, 0, "artifacts not present"))
            continue

        print(f"\n{'='*70}\nRUNNING {name}\n{'='*70}")
        t1 = time.time()
        proc = subprocess.run([str(PY), str(path)], cwd=HERE)
        elapsed = time.time() - t1
        results.append((name, proc.returncode == 0, elapsed, None))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    failed = []
    skipped = []
    for name, ok, elapsed, reason in results:
        if ok is None:
            print(f"  SKIP  {name}  ({reason})")
            skipped.append(name)
        elif ok:
            print(f"  PASS  {name}  ({elapsed:.1f}s)")
        else:
            print(f"  FAIL  {name}  ({elapsed:.1f}s)")
            failed.append(name)

    ran = len(results) - len(skipped)
    total = time.time() - t0
    print(f"\n{ran - len(failed)}/{ran} run passed, {len(skipped)} skipped, "
          f"in {total:.1f}s")

    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nALL RUNNABLE TEST FILES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
