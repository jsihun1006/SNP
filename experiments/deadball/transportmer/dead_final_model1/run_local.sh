#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -P -- "$(dirname -- "$0")" && pwd)
repo=$(CDPATH= cd -- "$script_dir/../../../.." && pwd)
bundled_python="/Users/jsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12"
if [ -n "${MODEL1_PYTHON:-}" ]; then
    python="$MODEL1_PYTHON"
elif [ -x "$bundled_python" ]; then
    python="$bundled_python"
else
    python=python3.12
fi
trial_dependencies=""
for candidate in /private/tmp/dfl-deadball-lightgbm /private/tmp/pathcrf-transition-deps /private/tmp/dfl-ha-torch /private/tmp/mtah-dfl-runtime.cz3a8v /private/tmp/pathcrf-runtime; do
    if [ -d "$candidate" ]; then
        trial_dependencies="$trial_dependencies:$candidate"
    fi
done
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/pathcrf-mpl}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src:$repo${PYTHONPATH:+:$PYTHONPATH}$trial_dependencies"
cd "$repo"
exec "$python" -u "$repo/src/deadball/train_dead_final_model1.py" "$@"
