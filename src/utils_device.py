# utils_device.py
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    else:
        return "cpu"
