#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pip install -r requirements.txt
echo "setup done (no data to download: everything is simulated)"
