"""Outcome of the self-weighted dynamics in the N-player public goods game across endowment e,
multiplier r and group size N. 150 runs per setting from uniform random initial probabilities.
Also a replicator control. Writes results/pgg_sweep.csv and results/replicator_control.json."""
import json
import numpy as np, pandas as pd
from pathlib import Path
from dynamics import run

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(1)
rows = []
for e in [0.25, 1.0, 3.0, 10.0]:
    for r in [1.5, 3.0]:
        for N in [3, 5, 10, 20, 50, 100, 200]:
            if r >= N:
                continue
            ms, unsettled, rule_ok = [], 0, True
            for _ in range(150):
                p, _t = run(N, r, e, "self", rng)
                if np.any((p > 1e-6) & (p < 1 - 1e-6)):
                    unsettled += 1; continue
                m = int((p > 0.5).sum()); ms.append(m)
                if m > 0 and e - 1 + r * m / N <= 0:        # a locked-in contributor must earn a positive reward
                    rule_ok = False
            ms = np.array(ms)
            rows.append(dict(e=e, r=r, N=N, runs=len(ms), unsettled=unsettled,
                             all_contribute=np.mean(ms == N), all_defect=np.mean(ms == 0),
                             split=np.mean((ms > 0) & (ms < N)), contributor_share=(ms / N).mean(), rule_ok=rule_ok))
            print(rows[-1], flush=True)
pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / "pgg_sweep.csv", index=False)
ctrl = {}
for e in [0.25, 1.0, 3.0, 10.0]:
    ctrl[str(e)] = sorted({int((run(50, 3.0, e, "rep", rng)[0] > 0.5).sum()) for _ in range(20)})
json.dump(dict(description="replicator dynamics, N=50, r=3: set of final contributor counts over 20 runs, by e", result=ctrl),
          open(ROOT / "results" / "replicator_control.json", "w"), indent=1)
print("replicator control:", ctrl)
