import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import sys
from pathlib import Path

from dirspread.pipeline import run_tasks


def main():
    root = Path(__file__).resolve().parent
    spec = json.loads((root / "tasks.json").read_text())
    wanted = {int(item) for item in sys.argv[1:]}
    tasks = spec["tasks"]
    if wanted:
        tasks = [task for task in tasks if task["id"] in wanted]
        missing = wanted - {task["id"] for task in tasks}
        if missing:
            raise SystemExit(f"unknown segment {sorted(missing)}")
    run_tasks(
        spec["checkpoint"],
        spec["tokenizer"],
        tasks,
        root / "videos",
        int(spec["frames"]),
        skip_existing=not wanted,
    )


if __name__ == "__main__":
    main()
