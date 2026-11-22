"""Compare the simulated contributor share at N=200 with the frozen-field formula, and check that the
agents who lock into contributing are exactly the agents with the highest initial probabilities.
Writes results/formula_check.csv."""
import numpy as np, pandas as pd
from pathlib import Path
from dynamics import run, predicted_share

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)
rows = []
for e in [1.0, 1.5, 2, 3, 5, 10, 30]:
    for r in [1.5, 3.0, 6.0]:
        shares, ordered = [], 0
        for _ in range(20):
            p0 = rng.random(200); p, _t = run(200, r, e, "self", rng, init=p0)
            coop = p > 0.5; shares.append(coop.mean())
            if coop.any() and (~coop).any():
                ordered += p0[coop].min() > p0[~coop].max()
        rows.append(dict(e=e, r=r, sim_mean=np.mean(shares), sim_sd=np.std(shares), formula=predicted_share(e, r),
                         runs=20, runs_where_contributors_are_the_initially_most_willing=ordered))
        print(rows[-1], flush=True)
pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / "formula_check.csv", index=False)
