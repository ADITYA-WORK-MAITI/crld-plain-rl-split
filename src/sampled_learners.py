"""Real sampled learners (noisy actions and rewards), N=50. Writes results/sampled_learners.json."""
import json
import numpy as np
from pathlib import Path
from dynamics import sampled, predicted_share

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(3)
out = {}
for e, r in [(3.0, 3.0), (10.0, 3.0), (1.0, 3.0)]:
    shares, locked, corr = [], [], []
    for _ in range(12):
        p, p0 = sampled(50, r, e, rng)
        coop = p > 0.5
        shares.append(coop.mean()); locked.append(np.mean((p < 0.01) | (p > 0.99))); corr.append(np.corrcoef(p0, coop)[0, 1])
    out[f"plain_e={e}_r={r}"] = dict(contributor_share_mean=float(np.mean(shares)), sd=float(np.std(shares)),
                                      formula=predicted_share(e, r), share_of_agents_locked_in=float(np.mean(locked)),
                                      corr_initial_prob_vs_final_choice=float(np.mean(corr)))
    print(out[f"plain_e={e}_r={r}"])
p, _ = sampled(50, 3.0, 3.0, rng, alg="replicator")
out["importance_weighted_e=3_r=3"] = dict(contributor_share=float((p > 0.5).mean()))
print(out["importance_weighted_e=3_r=3"])
json.dump(out, open(ROOT / "results" / "sampled_learners.json", "w"), indent=1)
