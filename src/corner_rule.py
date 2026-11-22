"""Which pure action profiles attract, in random N-agent, M-action games?
Rule tested for the self-weighted dynamics: profile a attracts iff every agent's reward at a is positive.
Known rule for the replicator: a attracts iff a is a strict Nash equilibrium.
Each corner is perturbed by 1e-3 and iterated. A corner counts as attracting only if the state never
leaves its neighbourhood (every agent keeps >= 0.9 on its corner action throughout) and ends within 1e-6
of it. Checking only the final state is not enough: the replicator can travel round a cycle of corners
and be back near the start when the clock stops. Corners whose margin is below 0.1 are skipped as marginal.
Writes results/corner_rule.json."""
import itertools, json, string
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
L = string.ascii_lowercase

def mean_rewards(R, X):
    N, M = X.shape; out = np.empty((N, M))
    for i in range(N):
        idx = L[:N]; ops = [R[i]]; subs = [idx]
        for j in range(N):
            if j != i:
                ops.append(X[j]); subs.append(idx[j])
        out[i] = np.einsum(",".join(subs) + "->" + idx[i], *ops)
    return out

def attracts(R, a, kind, alpha, T):
    N, M = len(a), R.shape[1]
    X = np.zeros((N, M))
    for i in range(N):
        n = rng.random(M); n[a[i]] = 0; X[i] = 1e-3 * n / n.sum(); X[i, a[i]] = 1 - 1e-3
    for _ in range(T):
        Rx = mean_rewards(R, X)
        lw = np.log(np.maximum(X, 1e-300)) + alpha * (X * Rx if kind == "self" else Rx)
        lw -= lw.max(1, keepdims=True); X = np.exp(lw); X /= X.sum(1, keepdims=True)
        if any(X[i, a[i]] < 0.9 for i in range(N)):
            return False
    return all(X[i, a[i]] > 1 - 1e-6 for i in range(N))

games = []
for g in range(40):
    N = int(rng.integers(2, 5)); M = int(rng.integers(2, 4))
    games.append(rng.normal(0.3, 1, size=(N,) + (M,) * N))

def test(alpha, T):
    res = {"self": [0, 0], "rep": [0, 0]}; mism = []
    for gi, R in enumerate(games):
        N, M = R.shape[0], R.shape[1]
        for a in itertools.product(range(M), repeat=N):
            own = np.array([R[(i,) + a] for i in range(N)])
            gaps = np.array([min(R[(i,) + a] - R[(i,) + a[:i] + (b,) + a[i + 1:]] for b in range(M) if b != a[i]) for i in range(N)])
            for kind, margin, pred in [("self", np.abs(own).min(), own.min() > 0), ("rep", np.abs(gaps).min(), gaps.min() > 0)]:
                if margin < 0.1:
                    continue
                ok = attracts(R, a, kind, alpha, T) == pred
                res[kind][0] += ok; res[kind][1] += 1
                if not ok:
                    mism.append(dict(kind=kind, game=gi, profile=list(a), predicted=bool(pred)))
    return res, mism

out = {}
for alpha, T in [(1.0, 600), (0.2, 6000)]:
    res, mism = test(alpha, T)
    out[f"alpha={alpha},T={T}"] = dict(self_weighted=f"{res['self'][0]}/{res['self'][1]}", replicator=f"{res['rep'][0]}/{res['rep'][1]}", mismatches=mism)
    print(alpha, T, {k: f"{v[0]}/{v[1]}" for k, v in res.items()}, "mismatches:", len(mism))
json.dump(out, open(ROOT / "results" / "corner_rule.json", "w"), indent=1)
