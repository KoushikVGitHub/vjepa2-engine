"""CPU test: the pk baseline (ps_baseline.py) is scored on EXACTLY the probe's held-out test sims.

S1 fix guard. The headline "conv R^2 vs pk R^2" is only fair if both report the same held-out set.
run_probe.py splits with sim_split(len(ds)) (defaults maps_per_sim=15, seed=0); ps_baseline.py now
splits with sim_split(N, maps_per_sim=15, seed=0). This proves those return identical indices, and
that the split is sim-level disjoint. Pure numpy + the real sim_split -- no data, no GPU.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from probe import sim_split

N, MPS = 15000, 15


def main():
    probe_tr, probe_va, probe_te = sim_split(N)                       # how run_probe.py calls it
    pk_tr, pk_va, pk_te = sim_split(N, maps_per_sim=MPS, seed=0)      # how ps_baseline.py calls it

    assert np.array_equal(probe_tr, pk_tr) and np.array_equal(probe_va, pk_va) \
        and np.array_equal(probe_te, pk_te), "pk baseline split != probe split -- comparison unfair!"
    print(f"[ok] pk test set == probe test set exactly: {len(pk_te)} maps ({len(pk_te)//MPS} sims)")

    tr_s = {int(i) // MPS for i in pk_tr}
    va_s = {int(i) // MPS for i in pk_va}
    te_s = {int(i) // MPS for i in pk_te}
    assert tr_s.isdisjoint(va_s) and tr_s.isdisjoint(te_s) and va_s.isdisjoint(te_s), \
        "a sim spans two splits -- leakage!"
    assert len(tr_s) + len(va_s) + len(te_s) == N // MPS == 1000
    print(f"[ok] sim-level disjoint: {len(tr_s)} train / {len(va_s)} val / {len(te_s)} test sims")

    print("\nALL GREEN -- pk baseline reports on precisely the probe's held-out test sims.")


if __name__ == "__main__":
    main()
