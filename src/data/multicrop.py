"""Multi-crop augmentation-invariance dataset — VISReg/LeJEPA's NATIVE training paradigm.

The reference VISReg does NOT do masked prediction. It makes several augmented crops of each image
(global + local views), encodes them all, and makes their embeddings AGREE (invariance), regularized
by VISReg. That augmentation invariance is a harder-to-shortcut objective than masked interpolation,
which is likely why VISReg holds rank there but collapsed in our masked-prediction setup (see
study/notes). This dataset produces the crops; the model's forward_multicrop consumes them.

V1 simplifications (documented, flagged for later):
  - ALL crops at the SAME resolution (varying scale/content via RandomResizedCrop), because our
    ViTEncoder has a fixed positional embedding; true multi-resolution (DINO's smaller local crops)
    would need pos-embedding interpolation.
  - Grayscale, so it drops into the 1-channel encoder unchanged (color jitter -> brightness/contrast).
  - NATURAL IMAGES ONLY (stl10/cifar10). Multi-crop invariance assumes the augmentation preserves the
    label; for cosmology, RandomResizedCrop changes the physical field-of-view and large-scale power,
    so it is NOT obviously Omega_m/sigma_8-preserving -- a separate design question.
"""
import torch
from torch.utils.data import Dataset

from data.images import _MEAN, _STD          # shared grayscale standardization


class MultiCropDataset(Dataset):
    """torchvision image dataset -> (n_global + n_local) standardized grayscale crops per image.

    __getitem__ returns a (C, 1, img, img) tensor, C = n_global + n_local; the GLOBAL crops come
    first (forward_multicrop pulls the others toward their mean). Default collate stacks to
    (B, C, 1, img, img).
    """

    def __init__(self, root, name="stl10", img=96, n_global=2, n_local=6, download=True):
        import torchvision
        import torchvision.transforms as T
        self.n_global, self.n_local = n_global, n_local

        self.gray = T.Grayscale(1)
        # global = large scale (most of the image); local = small scale (zoomed-in patches).
        # brightness/contrast jitter stands in for color jitter on a 1-channel image.
        jit = T.ColorJitter(brightness=0.4, contrast=0.4)
        self.global_t = T.Compose([
            T.RandomResizedCrop(img, scale=(0.4, 1.0)), T.RandomHorizontalFlip(),
            T.RandomApply([jit], p=0.8), T.ToTensor()])
        self.local_t = T.Compose([
            T.RandomResizedCrop(img, scale=(0.05, 0.4)), T.RandomHorizontalFlip(),
            T.RandomApply([jit], p=0.8), T.ToTensor()])

        if name == "stl10":
            self.base = torchvision.datasets.STL10(root, split="unlabeled", download=download)
        elif name == "cifar10":
            self.base = torchvision.datasets.CIFAR10(root, train=True, download=download)
        else:
            raise ValueError(f"multicrop needs an image dataset (stl10|cifar10), got {name!r}.")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _ = self.base[i]                       # PIL RGB, drop label (SSL)
        g = self.gray(img)                          # grayscale PIL
        views = ([self.global_t(g) for _ in range(self.n_global)]
                 + [self.local_t(g) for _ in range(self.n_local)])   # each (1, img, img) in [0,1]
        x = torch.stack(views)                      # (C, 1, img, img)
        return (x - _MEAN) / _STD
