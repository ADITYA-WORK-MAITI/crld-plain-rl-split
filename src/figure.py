"""Figure: contributor share vs endowment, simulation and formula. Writes figures/split_vs_endowment.png
and results/figure_data.csv."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dynamics import run, predicted_share

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(11)
E = [1, 1.5, 2, 3, 5, 10, 30]; RS = [(1.5, "#2a78d6", "o"), (3.0, "#eb6834", "s"), (6.0, "#1baf7a", "D")]
rows = []
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=200); fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
eg = np.geomspace(1, 30, 200)
for r, col, mk in RS:
    ax.plot(eg, predicted_share(eg, r), color=col, lw=2, label=f"r = {r}: formula")
    m, s = [], []
    for e in E:
        f = [(run(200, r, e, "self", rng)[0] > 0.5).mean() for _ in range(12)]
        m.append(np.mean(f)); s.append(np.std(f)); rows.append(dict(r=r, e=e, sim_mean=m[-1], sim_sd=s[-1], formula=predicted_share(e, r)))
    ax.errorbar(E, m, yerr=s, fmt=mk, color=col, ms=7, mec="#fcfcfb", mew=1.2, elinewidth=1.2, label=f"r = {r}: simulation")
ax.axhline(0, color="#52514e", lw=2, ls="--"); ax.text(1.05, 0.015, "importance-weighted (replicator) learners: everyone defects", fontsize=9, color="#52514e")
ax.axhline(0.5, color="#b9b8b2", lw=1, ls=":"); ax.text(13, 0.507, "limit: half and half", fontsize=9, color="#52514e")
ax.set_xscale("log"); ax.set_xticks(E); ax.set_xticklabels([str(x) for x in E])
ax.set_ylim(-0.02, 0.56); ax.set_xlabel("Flat endowment e paid to every agent (cost of contributing = 1)", color="#52514e")
ax.set_ylabel("Share of agents locked into contributing (N = 200)", color="#52514e")
ax.grid(color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: ax.spines[sp].set_color("#b9b8b2")
ax.tick_params(colors="#52514e"); ax.legend(frameon=False, fontsize=8.5, loc="center right")
ax.set_title("Identical learners split into two permanent classes.\nA flat payment, which changes no incentive, sets the split.",
             loc="left", fontsize=13, fontweight="bold", color="#0b0b0b")
fig.text(0.01, 0.01, "N-player public goods game, self-weighted dynamics (plain softmax RL, no importance weighting). Points: mean ± sd\n"
         "of 12 runs from uniform random initial policies. Formula: f = (e - 1 + r/2) / (2e - 1 + r).", fontsize=8, color="#52514e")
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(ROOT / "figures" / "split_vs_endowment.png", facecolor=fig.get_facecolor())
pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / "figure_data.csv", index=False)
print("wrote figure")
