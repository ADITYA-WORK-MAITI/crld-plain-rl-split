"""N-player linear public goods game and two learning dynamics.

Game. Each of N agents receives a flat endowment e, then chooses C (contribute c) or D.
Contributions are multiplied by r and shared equally:
    R_C = e - c + r c (1 + m_others) / N,   R_D = e + r c m_others / N,
where m_others is the number of other contributors. For 1 < r < N, D is dominant (R_D - R_C = c(1 - r/N) > 0).
Expected rewards are linear in the others' contribution probabilities, so the deterministic dynamics are exact.

Dynamics, in log-odds l_i = log(p_i / (1 - p_i)) of agent i's probability of contributing:
    self-weighted ('plain' RL, no baseline, no importance weighting):  l_i += alpha (p_i R_C - (1 - p_i) R_D)
    replicator   (importance-weighted RL):                              l_i += alpha (R_C - R_D)
These are the two-action forms of the collective-reinforcement-learning dynamics in log-weight form:
    self-weighted: log w_i(a) += alpha x_i(a) R_i(a)      replicator: log w_i(a) += alpha R_i(a).
"""
import numpy as np


def rewards(p, N, r, e, c=1.0):
    """Expected reward of C and of D for every agent, given everyone's current probabilities p."""
    others = p.sum() - p
    return e - c + r * c * (1 + others) / N, e + r * c * others / N


def run(N, r, e, kind, rng, alpha=0.05, c=1.0, T=200_000, init=None):
    """Iterate the deterministic dynamics. Returns final probabilities and the number of steps used."""
    p = rng.random(N) if init is None else np.asarray(init, float).copy()
    l = np.log(p) - np.log1p(-p)
    for t in range(T):
        p = 1 / (1 + np.exp(-l))
        RC, RD = rewards(p, N, r, e, c)
        l += alpha * ((p * RC - (1 - p) * RD) if kind == "self" else (RC - RD))
        np.clip(l, -60, 60, out=l)
        if t % 200 == 0 and np.all(np.abs(l) > 25):
            break
    return 1 / (1 + np.exp(-l)), t


def predicted_share(e, r, c=1.0):
    """Frozen-field prediction of the final share of contributors, for uniform random initial
    probabilities and large N. Agent i locks into C iff p_i R_C > (1 - p_i) R_D, i.e. p_i > R_D / (R_C + R_D).
    With the others' field frozen at its initial mean (N-1)/2 contributors and N large,
    R_C = e - c + r c / 2 and R_D = e + r c / 2, so the share above the threshold is
    1 - R_D / (R_C + R_D) = (e - c + r c / 2) / (2 e - c + r c)."""
    return (e - c + r * c / 2) / (2 * e - c + r * c)


def sampled(N, r, e, rng, alg="plain", alpha=0.02, beta=1.0, T=40_000, c=1.0, p0=None):
    """Real sampled learners. Each step every agent samples an action from softmax(beta Q) and adds
    alpha * reward to the Q value of the action taken. 'plain' = no weighting; 'replicator' = the
    update is divided by the probability of the action taken (importance weighting)."""
    p0 = rng.random(N) if p0 is None else np.asarray(p0, float)
    Q = np.stack([np.log(1 - p0), np.log(p0)], 1) / beta       # columns: D, C; initial policy = p0
    idx = np.arange(N)
    for _ in range(T):
        z = beta * Q; z -= z.max(1, keepdims=True); x = np.exp(z); x /= x.sum(1, keepdims=True)
        a = (rng.random(N) < x[:, 1]).astype(int)              # 1 = contribute
        pay = e - c * a + r * c * a.sum() / N
        w = 1.0 if alg == "plain" else 1 / np.maximum(x[idx, a], 1e-3)
        Q[idx, a] += alpha * pay * w
    z = beta * Q; z -= z.max(1, keepdims=True); x = np.exp(z); x /= x.sum(1, keepdims=True)
    return x[:, 1], p0
