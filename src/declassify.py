"""
declassify.py -- adversarial suite-invariance (de-classification) for CAMELS-JEPA.

The L21/L22 diagnosis: the encoder learns suite-specific feedback physics, so cross-suite
transfer fails. The fix with the largest published evidence (Miest 2502.13239: +17% Omega_m
/ +38% sigma8 on unseen suites) is DANN-style de-classification -- an adversary tries to
predict which suite a latent came from, and a Gradient Reversal Layer makes the encoder
train to DEFEAT it. The encoder is pushed toward a representation from which the suite is
not decodable, i.e. suite/feedback-invariant cosmology.

This module is self-contained and testable with no CAMELS data (run `python declassify.py`).
Integration points into src/train_fsdp.py are in DECLASSIFY_INTEGRATION.md.

Contract:
  - forward the FULL-image encoder pass (the one lejepa/visreg already computes), mean-pool
    to (B, d), pass through declassify_loss with per-sample suite ids.
  - total_loss = jepa_loss + declassify_lambda * declassify_loss.
    The GRL carries the sign: minimizing this term MAXIMIZES the encoder's suite-confusion
    while the adversary's own params still learn to classify (standard DANN).
  - log suite_accuracy() every N steps -- it is the mechanistic gate (pre-registered claim
    astrid-suite-declassified: must fall to chance ~0.5, else the loss did not de-classify).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    """Identity forward; scaled-negated gradient backward (DANN). lambda_ is stashed on ctx."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return _GradReverse.apply(x, lambda_)


def dann_lambda(step, total_steps, gamma=10.0, max_lambda=1.0):
    """Standard DANN schedule: ramp the reversal strength 0 -> max_lambda over training so the
    adversary is not fighting a random encoder early (which destabilizes both). p in [0,1]."""
    p = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return max_lambda * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


class SuiteAdversary(nn.Module):
    """Small MLP that predicts the SUITE id from a pooled latent. Trained normally on its own
    params; the encoder sees the reversed gradient through the GRL. Deliberately shallow -- a
    too-strong adversary makes an unwinnable game and collapses the encoder."""

    def __init__(self, d, n_suites, hidden=None, p_drop=0.1):
        super().__init__()
        hidden = hidden or max(2 * d, 64)
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, n_suites),
        )

    def forward(self, pooled):
        return self.net(pooled)


def declassify_loss(pooled, suite_ids, adversary, lambda_=1.0):
    """Cross-entropy of the adversary on GRL-reversed pooled features.

    pooled     (B, d)   mean-pooled full-image encoder tokens (float)
    suite_ids  (B,)     long, 0..n_suites-1
    adversary  SuiteAdversary
    lambda_    reversal strength (use dann_lambda(step, total_steps))

    Returns (loss, logits). Add `loss` to the JEPA loss; the GRL flips the sign for the encoder.
    """
    logits = adversary(grad_reverse(pooled, lambda_))
    loss = F.cross_entropy(logits, suite_ids)
    return loss, logits


@torch.no_grad()
def suite_accuracy(logits, suite_ids):
    """The mechanistic metric. Target after successful de-classification: ~1/n_suites (chance)."""
    return (logits.argmax(dim=1) == suite_ids).float().mean().item()


# ---------------------------------------------------------------------------
# Feedback-parameter invariance (the CAMELS-fit variant). The suite structure is
# 2-similar (ITNG~Astrid, separability 0.512) + 1-outlier (SIMBA), so suite-de-classification
# cannot hold out SIMBA with a distinguishable training pair. Instead shed the REAL confound:
# the feedback parameters (A_SN1, A_AGN1, A_SN2, A_AGN2), which are known per-map and vary
# WITHIN ITNG. Train on ITNG, regress feedback out of the latent, test truly-unseen SIMBA.
# Cosmology (Omega_m, sigma8) is varied independently of feedback in the LH set, so this should
# not touch the signal. This is transfer-lab lever 3 (feedback marginalisation).
# ---------------------------------------------------------------------------
class FeedbackAdversary(nn.Module):
    """Regresses the continuous feedback params from a pooled latent. Same shallow shape as the
    suite adversary; MSE head instead of a classifier."""

    def __init__(self, d, n_params=4, hidden=None, p_drop=0.1):
        super().__init__()
        hidden = hidden or max(2 * d, 64)
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, n_params),
        )

    def forward(self, pooled):
        return self.net(pooled)


def feedback_loss(pooled, targets, adversary, lambda_=1.0):
    """MSE of the adversary on GRL-reversed pooled features. `targets` = STANDARDIZED feedback
    params (B, n_params). Adding this MINIMIZES feedback-decodability in the encoder while the
    adversary's own params learn to regress it. Returns (loss, preds)."""
    preds = adversary(grad_reverse(pooled, lambda_))
    loss = F.mse_loss(preds, targets)
    return loss, preds


@torch.no_grad()
def feedback_r2(preds, targets):
    """Mechanistic metric for feedback invariance: the adversary's R^2 predicting feedback.
    Target after success: ~0 (feedback no longer decodable from the latent). Mean over params."""
    p, t = preds.float(), targets.float()
    ss_res = ((t - p) ** 2).sum(0)
    ss_tot = ((t - t.mean(0)) ** 2).sum(0).clamp_min(1e-8)
    return float((1.0 - ss_res / ss_tot).mean().item())


# ---------------------------------------------------------------------------
# self-test: no CAMELS data. Verifies GRL sign, that the adversary CAN learn suite when the
# encoder is frozen, and that under GRL the encoder is pushed to erase the suite signal.
# ---------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    B, d, n_suites = 256, 32, 2

    # 1. GRL flips gradient sign and scales by lambda.
    x = torch.randn(4, 3, requires_grad=True)
    grad_reverse(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.5)), "GRL sign/scale wrong"
    print("  [ok] GRL reverses and scales gradient")

    # 2. Build a latent where the suite is linearly decodable (a shift along dim 0).
    def make_batch():
        suite = torch.randint(0, n_suites, (B,))
        z = torch.randn(B, d)
        z[:, 0] += 3.0 * suite.float()          # suite signature
        return z, suite

    # adversary alone (no reversal) must classify the suite well.
    adv = SuiteAdversary(d, n_suites)
    opt = torch.optim.Adam(adv.parameters(), lr=1e-2)
    for _ in range(300):
        z, s = make_batch()
        opt.zero_grad()
        loss, logit = declassify_loss(z, s, adv, lambda_=0.0)   # lambda 0 => adversary trains, no reversal
        loss.backward(); opt.step()
    z, s = make_batch()
    acc = suite_accuracy(adv(z), s)
    assert acc > 0.9, f"adversary failed to learn suite: acc={acc:.2f}"
    print(f"  [ok] adversary decodes suite when present (acc={acc:.2f})")

    # 3. A trainable 'encoder' (linear) under GRL should learn to ERASE the suite signal:
    #    the adversary's accuracy on the encoder output should fall toward chance.
    enc = nn.Linear(d, d)
    adv2 = SuiteAdversary(d, n_suites)
    opt = torch.optim.Adam(list(enc.parameters()) + list(adv2.parameters()), lr=5e-3)
    for step in range(1500):
        z, s = make_batch()
        opt.zero_grad()
        h = enc(z)
        lam = dann_lambda(step, 1500, max_lambda=1.0)
        loss, _ = declassify_loss(h, s, adv2, lambda_=lam)
        loss.backward(); opt.step()
    z, s = make_batch()
    acc_after = suite_accuracy(adv2(enc(z)), s)
    print(f"  [ok] under GRL the encoder erased the suite (adversary acc {acc_after:.2f} -> ~{1/n_suites:.2f})")
    assert acc_after < 0.75, f"GRL did not de-classify: acc={acc_after:.2f}"

    # 4. lambda schedule monotone 0 -> max
    assert dann_lambda(0, 100) < 1e-6 and dann_lambda(100, 100) > 0.99
    print("  [ok] dann_lambda ramps 0 -> 1")

    # 5. Feedback (continuous) invariance: a latent carrying a feedback signal in some dims;
    #    the adversary regresses it, and under GRL a trainable encoder erases it (feedback_r2 -> 0).
    def make_fb():
        fb = torch.randn(B, 4)                              # standardized feedback targets
        z = torch.randn(B, d)
        z[:, :4] += 2.0 * fb                                # feedback signature in the first 4 dims
        return z, fb
    fbadv = FeedbackAdversary(d, 4)
    opt = torch.optim.Adam(fbadv.parameters(), lr=1e-2)
    for _ in range(400):
        z, fb = make_fb()
        opt.zero_grad(); l, p = feedback_loss(z, fb, fbadv, lambda_=0.0); l.backward(); opt.step()
    z, fb = make_fb()
    r2_present = feedback_r2(fbadv(z), fb)
    assert r2_present > 0.7, f"feedback adversary failed to regress: r2={r2_present:.2f}"
    print(f"  [ok] feedback adversary regresses feedback when present (r2={r2_present:.2f})")

    # GRL vs a lambda=0 control isolates the reversal's effect (an invertible toy encoder can't
    # fully destroy info a strong adversary recovers; the real ViT >> the shallow adversary).
    def _train_enc(lam_max, steps=2000):
        enc = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        adv = FeedbackAdversary(d, 4)
        opt = torch.optim.Adam(list(enc.parameters()) + list(adv.parameters()), lr=3e-3)
        for step in range(steps):
            z, fb = make_fb()
            opt.zero_grad()
            l, _ = feedback_loss(enc(z), fb, adv, lambda_=dann_lambda(step, steps, max_lambda=lam_max))
            l.backward(); opt.step()
        z, fb = make_fb()
        return feedback_r2(adv(enc(z)), fb)
    r2_ctrl, r2_grl = _train_enc(0.0), _train_enc(1.0)
    print(f"  [ok] GRL reduces feedback decodability: control r2 {r2_ctrl:.2f} -> GRL {r2_grl:.2f}")
    assert r2_grl < r2_ctrl - 0.15, f"GRL did not reduce feedback (ctrl {r2_ctrl:.2f}, grl {r2_grl:.2f})"
    print("SELF-TEST PASS")


if __name__ == "__main__":
    _selftest()
