# Audit notes

## Equations checked by hand

- Expected rewards: with m expected other contributors, R_C = e − c + r c (1 + m)/N and R_D = e + r c m/N. Linear in the others' probabilities, so iterating on probabilities is exact.
- Plain (self-weighted) update in log-weights is log w(a) += α x(a) R(a). For two actions this gives l += α (p R_C − (1 − p) R_D). The code matches.
- Replicator update log w(a) += α R(a) gives l += α (R_C − R_D). The code matches.
- Sampled plain learner: Q(a) += α · reward for the action taken. Its expected update is α x(a) R(a), the plain dynamics. The importance-weighted variant divides by x(a), whose expectation is α R(a), the replicator. The code matches.
- Formula: derived in the README from the threshold p > R_D / (R_C + R_D) with the field frozen at its initial mean. The code's `predicted_share` matches the derivation.

## A real bug found and fixed

The corner-stability test originally checked only where the state was after a fixed number of steps. Under the replicator dynamics the state can leave a corner, travel round a cycle of corners, and be back near the starting corner when the clock stops. This produced 1 to 3 false "attracting" verdicts out of 580 and was traced to one specific corner by stepping through the trajectory. The test now requires that the state never leaves the corner's neighbourhood (every agent keeps at least 0.9 on its corner action throughout). After the fix both rules agree in all corners (531/531 and 580/580), and the test runs ten times faster because it stops as soon as the state leaves.

This bug could not have produced a false positive for the plain dynamics result, because the plain rule already scored 531/531 before the fix. It is recorded because it affected the control.

## A claim corrected

The earlier summary said that for e ≥ 1 and N ≥ 20 every run splits. The saved sweep shows one setting below 100%: e = 1, r = 1.5, N = 20 splits in 97% of runs. The README now states the claim for N ≥ 50, with the N = 20 exception named.

## Checks that passed

- Every result in `results/` was regenerated from the refactored code and matches the earlier run digit for digit (same seeds).
- No run in the sweep ended unsettled.
- In every run with a locked-in contributor, that contributor's reward was positive, as the stability rule requires.
- In all 420 formula-check runs the contributors are exactly the initially most willing agents.
- A script named `inspect.py` was used briefly during the audit and shadowed Python's `inspect` module. It was renamed. No shipped script shadows a standard module.

## Second pass

- Re-read every script against the equations above; no changes.
- `corner_rule.py` draws the perturbation directions from the same generator as the games, so the two step-size runs use different perturbations. Both agree in every corner, which is the stronger check.
- The early-stop rule in `dynamics.run` (all log-odds beyond ±25) implies probabilities within 1e-11 of 0 or 1, consistent with the 1e-6 "settled" test in the sweep.

## Known weaknesses that remain

- The formula is a large-N, frozen-field approximation; its worst error in the tested range is 0.032.
- Results are for uniform random initial policies. Other initial distributions change the share (the derivation shows how).
- Fixed step sizes throughout; no adaptive integration.
