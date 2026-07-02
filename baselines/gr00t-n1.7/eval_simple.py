#!/usr/bin/env python3
"""Launch the shared SIMPLE evaluator with the N1.7 preset directory."""

from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "gr00t-n1.6" / "eval_simple.py"

# The shared launcher resolves named presets relative to its own directory.
# Convert an N1.7 preset name into an absolute path before handing off.
if "--preset" in sys.argv:
    index = sys.argv.index("--preset") + 1
    value = Path(sys.argv[index])
    if not value.exists():
        value = HERE / "presets" / "eval" / f"{value}.yaml"
    sys.argv[index] = str(value.resolve())

runpy.run_path(str(SHARED), run_name="__main__")
