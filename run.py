"""Run any ocrfix command with the venv's Python, e.g. on Windows:

    .\\.venv\\Scripts\\python.exe run.py chunk data/real/clean data/real/chunks_clean
    .\\.venv\\Scripts\\python.exe run.py gold chunks data/real/gold data/real/gold_chunks
    .\\.venv\\Scripts\\python.exe run.py --help

Same commands and options as the `ocrfix` executable (`python -m ocrfix` works too), plus:
  - UTF-8 mode: restarts itself with ``python -X utf8`` when it isn't on, so files are read and
    written as UTF-8 on Windows too (Korean text, ⟨M1⟩ placeholders).
  - Profile: ``cpu`` (configs/profiles/cpu.yaml) unless ``--profile`` or ``OCRFIX_PROFILE``
    says otherwise.
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    if not sys.flags.utf8_mode:
        sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))
    os.environ.setdefault("OCRFIX_PROFILE", "cpu")
    from ocrfix.cli import app

    app(prog_name="run.py")
