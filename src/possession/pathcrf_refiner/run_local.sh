#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
PY_EXEC="${TRANSITION_PYTHON:-/Users/jsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12}"
if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="/private/tmp/dfl-ha-torch:/private/tmp/mtah-dfl-runtime.cz3a8v:/private/tmp/pathcrf-runtime"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/pathcrf-mpl}"
if [ "${1:-}" = verify ]; then
    exec "$PY_EXEC" 'src/possession/pathcrf_refiner/verify.py'
fi
exec "$PY_EXEC" 'src/possession/pathcrf_refiner/run.py' "$@" --official-prepared data/pathcrf_dfl/transition_head_v1/prepared
