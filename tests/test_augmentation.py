import numpy as np
import pytest

from util.augmentation import (
    RandomBrightness,
    RandomCrop,
    RandomCropOut,
    RandomFlip,
    RandomNoise,
)


def _sample():
    rgb = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    tdisp = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    tdisp = np.ascontiguousarray(tdisp[::-1])
    label = (np.arange(48 * 64) % 2).astype(np.uint8).reshape(48, 64)
    return rgb, tdisp, label


@pytest.mark.parametrize("aug", [
    RandomFlip(prob=1.0),
    RandomCrop(crop_rate=0.1, prob=1.0),
    RandomCropOut(crop_rate=0.2, prob=1.0),
    RandomBrightness(bright_range=0.15, prob=1.0),
    RandomNoise(noise_range=5, prob=1.0),
])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_augmentation_shapes_aligned(aug, seed):
    np.random.seed(seed)
    rgb, tdisp, label = _sample()
    rgb, tdisp, label = aug(rgb.copy(), tdisp.copy(), label.copy())
    assert rgb.shape == tdisp.shape
    assert rgb.shape[0] == label.shape[0]
    assert rgb.shape[1] == label.shape[1]
    assert rgb.shape[0] > 0 and rgb.shape[1] > 0
