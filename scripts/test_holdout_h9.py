"""CPU test for the H9 control (--holdout-test-sims): the maps excluded from PRETRAINING must be
exactly the PROBE's test sims, across every pooled field. Pure index arithmetic -- no GPU/data.

Guards the one way this control can silently lie: if the held-out set and the probe's test split
disagree, the encoder would still see (some of) the test sims and the "no-leakage" claim is false.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from probe import sim_split

MPS = 15          # maps per sim (CAMELS-LH)
NF = 12           # pooled fields (train_fsdp default)
PER = 15000       # maps per field (keep-all: 1000 sims x 15)
N_SIMS = PER // MPS


def holdout_indices(per_field, nf, seed=0):
    """Re-implements the block in train_fsdp.build_loader so the test pins the real arithmetic."""
    _, _, te_idx = sim_split(per_field, maps_per_sim=MPS, seed=seed)
    te_sims = sorted({int(i) // MPS for i in te_idx})
    excluded = set()
    for fi in range(nf):
        base = fi * per_field
        for s in te_sims:
            excluded.update(range(base + s * MPS, base + s * MPS + MPS))
    kept = [i for i in range(nf * per_field) if i not in excluded]
    return te_idx, te_sims, excluded, kept


def main():
    tr_idx, va_idx, te_idx = sim_split(PER, maps_per_sim=MPS, seed=0)
    tr_sims = {int(i) // MPS for i in tr_idx}
    va_sims = {int(i) // MPS for i in va_idx}
    te_idx2, te_sims, excluded, kept = holdout_indices(PER, NF, seed=0)

    # 1. split shape: 80/10/10 at sim level, disjoint
    assert len(te_sims) == int(0.1 * N_SIMS) == 100, len(te_sims)
    assert tr_sims.isdisjoint(te_sims) and va_sims.isdisjoint(te_sims), "probe split leaks sims!"
    print(f"[ok] probe split: {len(tr_sims)} train / {len(va_sims)} val / {len(te_sims)} test sims (disjoint)")

    # 2. excluded/kept counts exact
    assert len(excluded) == len(te_sims) * MPS * NF == 18000, len(excluded)
    assert len(kept) == NF * PER - 18000 == 162000, len(kept)
    print(f"[ok] holdout counts: excluded {len(excluded)} maps (100 sims x 15 x 12 fields), kept {len(kept)}")

    # 3. EVERY probe-test map (field 0) is excluded from pretraining -> zero unlabeled exposure
    assert all(i in excluded for i in te_idx), "a probe test map survived into pretraining!"
    print(f"[ok] all {len(te_idx)} probe-test maps (Mgas) are held out of pretraining")

    # 4. no kept map belongs to a test sim, in ANY field
    kept_set = set(kept)
    for i in kept:
        sim = (i % PER) // MPS
        assert sim not in te_sims, f"kept map {i} is test sim {sim}"
    # 5. and every train+val map of every field IS kept (we only dropped test sims)
    for fi in range(NF):
        base = fi * PER
        assert all((base + int(i)) in kept_set for i in tr_idx), f"field {fi} dropped a train map"
        assert all((base + int(i)) in kept_set for i in va_idx), f"field {fi} dropped a val map"
    print("[ok] kept set == exactly all train+val sims across all 12 fields (nothing else dropped)")

    print("\nALL GREEN -- H9 holdout excludes precisely the probe's test sims, all fields.")


if __name__ == "__main__":
    main()
