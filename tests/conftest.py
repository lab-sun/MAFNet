import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# One thread keeps the CPU forward/backward test from oversubscribing.
import torch

torch.set_num_threads(1)
