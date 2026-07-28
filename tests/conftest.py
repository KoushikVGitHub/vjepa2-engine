"""Shared test fixtures.

Every test here is CPU-only and CI-sized: the point is to gate the ENGINE's invariants
(distribution math, anti-collapse geometry, tokenizer shape-compatibility, mask structure)
on every push, not to reproduce a training run. Transformer DEPTH is kept at <=2 layers
throughout -- torch's CPU build is unstable on the real 24-layer attention, and depth is
irrelevant to every property under test.
"""
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed every test. These assertions are statistical; an unseeded run would flake."""
    torch.manual_seed(0)
    yield


@pytest.fixture
def gen():
    """A seeded generator, matching how the training loop feeds SIGReg (base + step)."""
    return torch.Generator().manual_seed(1234)
