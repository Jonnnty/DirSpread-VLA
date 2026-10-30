#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${OPENPI_ROOT:-}" ]]; then
  echo "set OPENPI_ROOT to the openpi checkout" >&2
  exit 1
fi
export PYTHONPATH="$OPENPI_ROOT/examples/libero:$OPENPI_ROOT/third_party/libero:$OPENPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
PY="${PYTHON:-$OPENPI_ROOT/.venv/bin/python}"
exec "$PY" "$ROOT/run.py" "$@"
