"""Natural-image SSL corpus — a VALIDATION dataset for the anti-collapse implementation.

Purpose: confirm VISReg / SIGReg behave correctly on the data type they were DESIGNED for (rich
natural images), which isolates the cosmology dimensional-collapse as a SMOOTH-FIELD property of
CAMELS rather than a bug in the loss. On smooth maps a masked block is largely interpolatable, so a
low-rank shortcut solves the pretext; natural images have sharp, high-frequency structure, so the
same masked-prediction task genuinely demands richer features. If pure VISReg holds rank here but
collapsed on CAMELS, the implementation is correct and the smoothness hypothesis holds.

Grayscale + resize so it drops into the existing 1-channel ViTEncoder with NO model change; SSL
only (labels discarded). torchvision auto-downloads STL-10 (100k unlabeled 96x96, built for SSL) or
CIFAR-10. `import torchvision` is lazy so CAMELS-only runs never need it installed.
"""
import torch
from torch.utils.data import Dataset

# Grayscale natural-image standardization ([0,1] scale). Pins the input scale the way the per-field
# CAMELS stats do, so the encoder sees a ~unit-variance input in both regimes.
_MEAN, _STD = 0.45, 0.25


class GrayImageDataset(Dataset):
    """torchvision image dataset -> standardized grayscale (1, img, img) tensors for SSL pretraining."""

    def __init__(self, root, name="stl10", img=96, augment=True, download=True):
        import torchvision
        import torchvision.transforms as T
        self._pipe = T.Compose([T.Grayscale(1), T.Resize((img, img)), T.ToTensor()])  # -> (1,H,W) in [0,1]
        self.augment = augment
        if name == "stl10":
            self.base = torchvision.datasets.STL10(root, split="unlabeled", download=download)
        elif name == "cifar10":
            self.base = torchvision.datasets.CIFAR10(root, train=True, download=download)
        else:
            raise ValueError(f"unknown image dataset {name!r} (expected 'stl10' | 'cifar10').")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _ = self.base[i]                        # (PIL image, label) -> drop the label (SSL)
        x = self._pipe(img)                          # (1, img, img) in [0,1]
        x = (x - _MEAN) / _STD
        # Natural images are NOT periodic, so the CAMELS roll augmentation would wrap edges wrongly.
        # A horizontal flip is the one exact symmetry that holds; the masking supplies the SSL signal.
        if self.augment and torch.rand(()) < 0.5:
            x = torch.flip(x, dims=(-1,))
        return x
