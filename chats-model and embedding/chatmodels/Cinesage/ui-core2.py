# Alias file for ui_core2.py
import runpy
import sys
from pathlib import Path

target_file = Path(__file__).parent / "ui_core2.py"
runpy.run_path(str(target_file), run_name="__main__")
