#!/usr/bin/env bash
# Rerun every simulation. About 8 minutes on a laptop. Outputs go to results/ and figures/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results figures
python3 src/corner_rule.py
python3 src/pgg_sweep.py
python3 src/formula_check.py
python3 src/sampled_learners.py
python3 src/figure.py
echo "all done"
