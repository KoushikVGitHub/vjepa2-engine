"""Day 5 - latent-rollout failure-mode analysis (the safety finding).

Roll the predictor forward N steps in latent space and measure how prediction
error grows vs ground-truth latents. Compounding error => shrinking reliable
planning horizon => a *safety* limit for any planner built on this world model.
"""
import numpy as np
import matplotlib.pyplot as plt


def rollout_error(model, clip, horizon: int):
    """Return per-step latent prediction error over `horizon` steps."""
    errs = []
    # TODO(Day5):
    #   z_t = encode(clip[:t0])
    #   for h in range(horizon):
    #       z_pred = predictor(z_t)            # predicted next latent
    #       z_true = encode(clip[:t0+h+1])     # actual latent
    #       errs.append(mse(z_pred, z_true))
    #       z_t = z_pred                        # autoregress in latent space
    raise NotImplementedError
    return np.array(errs)


def main():
    # TODO(Day5): average rollout_error over many clips; plot error vs step.
    horizon = 16
    mean_err = np.zeros(horizon)  # placeholder
    plt.plot(range(1, horizon + 1), mean_err, marker="o")
    plt.xlabel("rollout step"); plt.ylabel("latent prediction error")
    plt.title("Compounding error in V-JEPA 2 latent rollout")
    plt.savefig("analysis/rollout_error.png", dpi=130, bbox_inches="tight")
    print("saved analysis/rollout_error.png")
    # TODO: write the 1-paragraph safety claim this supports -> writeups/safety_post.md


if __name__ == "__main__":
    main()
