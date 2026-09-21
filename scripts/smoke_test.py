#!/usr/bin/env python3
"""Run every numbered example script and report which ones fail.

Usage (from anywhere; paths are resolved relative to the repository root):

    python scripts/smoke_test.py                  # run everything
    python scripts/smoke_test.py --only attention # only paths containing "attention"
    python scripts/smoke_test.py --timeout 300    # seconds allowed per script
    python scripts/smoke_test.py --list           # show what would run

Scripts are discovered as  NN-module/code/**/*.py  (``__init__.py`` is ignored).
Matplotlib runs headless (MPLBACKEND=Agg) so plots never block the run.

If a script is too slow, needs a download, or needs command-line arguments,
add its repository-relative path to SKIP below.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Repository-relative POSIX paths to leave out, e.g.
#   "09-projects/code/03-long_training.py",
SKIP = set()


def find_scripts(only):
    scripts = []
    for path in sorted(ROOT.glob("[0-9][0-9]-*/code/**/*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if path.name == "__init__.py" or rel in SKIP:
            continue
        if only and only not in rel:
            continue
        scripts.append(path)
    return scripts


def run_script(path, timeout):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        status = "PASS" if proc.returncode == 0 else "FAIL"
        detail = proc.stderr if status == "FAIL" else ""
    except subprocess.TimeoutExpired:
        status, detail = "TIMEOUT", "exceeded {}s".format(timeout)
    return status, time.time() - start, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", default="", help="only run scripts whose path contains this text")
    parser.add_argument("--timeout", type=int, default=120, help="seconds allowed per script")
    parser.add_argument("--list", action="store_true", help="list scripts without running them")
    args = parser.parse_args()

    scripts = find_scripts(args.only)
    if not scripts:
        print("No scripts found. Expected files like 02-attention/code/01-example.py")
        return 1

    if args.list:
        for path in scripts:
            print(path.relative_to(ROOT).as_posix())
        return 0

    print("Running {} script(s) with a {}s timeout each\n".format(len(scripts), args.timeout))
    failures = []
    for path in scripts:
        rel = path.relative_to(ROOT).as_posix()
        status, seconds, detail = run_script(path, args.timeout)
        print("{:<8}{:>6.1f}s  {}".format(status, seconds, rel))
        if status != "PASS":
            failures.append((rel, status, detail))

    print("\n{} passed, {} failed".format(len(scripts) - len(failures), len(failures)))
    for rel, status, detail in failures:
        print("\n--- {} ({}) ---".format(rel, status))
        print("\n".join(detail.strip().splitlines()[-12:]))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
