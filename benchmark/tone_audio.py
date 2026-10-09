"""Общий фильтр 16 → 8 кГц для обоих вариантов теста T-one."""
from __future__ import annotations

import numpy as np


def _lowpass():
    n = np.arange(127) - 63
    cutoff = 3800 / 16000
    kernel = 2 * cutoff * np.sinc(2 * cutoff * n) * np.blackman(127)
    return (kernel / kernel.sum()).astype(np.float32)


LOWPASS_8K = _lowpass()


def to_8k(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float32)
    filtered = np.convolve(samples, LOWPASS_8K, mode="full")[63:63 + len(samples)]
    return np.ascontiguousarray(filtered[::2], dtype=np.float32)
