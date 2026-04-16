import numpy as np
import os


def read_complex64(path: str) -> np.ndarray:
    """Read a raw binary IQ file as complex64."""
    path = os.path.normpath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.fromfile(path, dtype=np.complex64)
