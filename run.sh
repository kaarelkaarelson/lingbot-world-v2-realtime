#!/usr/bin/env bash
# Generate a clip with the 16.2 FPS stack: a thin wrapper over `lingbot clip`. Any `python -m lingbot.generate` flag can be appended:
#   ./run.sh --frame_num 361 --bench
#   ./run.sh --image my.jpg --action_path my_poses/ --prompt "..." --preset exact
set -euo pipefail
cd "$(dirname "$0")"
. .venv/bin/activate
exec lingbot clip "$@"
