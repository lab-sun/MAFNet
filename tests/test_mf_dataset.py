import numpy as np
import pytest
import torch
from PIL import Image

from util.MF_dataset import MF_dataset


def test_mf_dataset_rejects_unknown_split(tmp_path):
    with pytest.raises(AssertionError, match=r'split must be "train"\|"validation"\|"test"'):
        MF_dataset(str(tmp_path), "nope")


def test_mf_dataset_synthetic(tmp_path):
    split = "train"
    name = "0001"
    for folder in ("rgb", "tdisp", "label"):
        (tmp_path / split / folder).mkdir(parents=True)

    rgb = np.full((20, 30, 3), 255, dtype=np.uint8)
    tdisp = np.zeros((20, 30, 3), dtype=np.uint8)
    tdisp[..., 0] = 128
    tdisp[..., 1] = 64
    tdisp[..., 2] = 32
    label = np.zeros((20, 30), dtype=np.uint8)
    label[5:15, 8:22] = 1

    Image.fromarray(rgb).save(tmp_path / split / "rgb" / (name + ".png"))
    Image.fromarray(tdisp).save(tmp_path / split / "tdisp" / (name + ".png"))
    Image.fromarray(label).save(tmp_path / split / "label" / (name + ".png"))
    (tmp_path / (split + ".txt")).write_text(name + "\n")

    dataset = MF_dataset(str(tmp_path), split, input_h=16, input_w=24)
    assert len(dataset) == 1
    rgb_t, tdisp_t, label_t, out_name = dataset[0]

    assert out_name == name
    assert rgb_t.shape == (3, 16, 24)
    assert tdisp_t.shape == (3, 16, 24)
    assert label_t.shape == (16, 24)
    assert rgb_t.dtype == torch.float32
    assert tdisp_t.dtype == torch.float32
    assert label_t.dtype == torch.int64
    assert torch.all(rgb_t == 1)
    assert torch.allclose(tdisp_t[0], torch.full((16, 24), 128 / 255))
    assert torch.allclose(tdisp_t[1], torch.full((16, 24), 64 / 255))
    assert torch.allclose(tdisp_t[2], torch.full((16, 24), 32 / 255))
    assert set(label_t.unique().tolist()) <= {0, 1}
    assert int(label_t.max()) == 1
