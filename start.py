#!/usr/bin/env python3
"""Run the NeuralNote API server from the repository root."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    src = root / "src"
    if not src.is_dir():
        print("Error: src/ directory not found.", file=sys.stderr)
        sys.exit(1)

    port = int(os.environ.get("PORT", 7860))

    # Prefer running in-process if uvicorn is importable (e.g. venv activated)
    try:
        import uvicorn
        os.chdir(src)
        sys.path.insert(0, str(src))
        from app import app
        uvicorn.run(app, host="0.0.0.0", port=port)
        return
    except ModuleNotFoundError:
        pass

    # Otherwise run via project venv so "py start.py" or system python still works
    if sys.platform == "win32":
        venv_python = root / ".env" / "Scripts" / "python.exe"
    else:
        venv_python = root / ".env" / "bin" / "python"

    if venv_python.is_file():
        env = os.environ.copy()
        env["PYTHONPATH"] = str(src)
        try:
            code = subprocess.run(
                [str(venv_python), "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(port)],
                cwd=src,
                env=env,
            ).returncode
        except KeyboardInterrupt:
            code = 0
        sys.exit(code)

    print("Error: uvicorn not found. Either:", file=sys.stderr)
    print("  1. Activate the venv and run:  python start.py", file=sys.stderr)
    print("  2. Or install deps at root:     pip install -r requirements.txt", file=sys.stderr)
    print("  3. Or create a venv:            python -m venv .env  then  .env\\Scripts\\pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
