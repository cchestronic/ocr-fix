"""Run any ocrfix command. Called by the launchers in this folder, so on every machine:

    ./ocrfix chunk data/real/clean data/real/chunks_clean
    ./ocrfix --help

(``ocrfix.cmd`` on Windows, ``ocrfix`` on Linux; both start ``.venv``'s Python with this file.)

Same commands and options as the installed ``ocrfix`` executable, plus:
  - UTF-8 mode: restarts itself with ``python -X utf8`` when it isn't on, so files are read and
    written as UTF-8 on Windows too (Korean text, ⟨M1⟩ placeholders).
  - Profile: ``--profile``, else ``OCRFIX_PROFILE``, else ``cpu``. Set ``OCRFIX_PROFILE`` once
    per GPU machine (e.g. in ``~/.bashrc``). The profile in use is printed on stderr.
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    if not sys.flags.utf8_mode:
        sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))
    os.environ.setdefault("OCRFIX_PROFILE", "cpu")
    args = sys.argv[1:]
    profile = os.environ["OCRFIX_PROFILE"]
    for i, a in enumerate(args):
        if a == "--profile" and i + 1 < len(args):
            profile = args[i + 1]
        elif a.startswith("--profile="):
            profile = a.split("=", 1)[1]
    if args and "--help" not in args:
        print(f"[ocrfix] profile: {profile}", file=sys.stderr)
    from ocrfix.cli import app

    app(prog_name="ocrfix")
