"""CLIP embeddings for matching — reuses EdgeAI's exact model (ViT-B-32/openai)
so user text/photos land in the SAME vector space as the detected-item crops.

CLIP puts text and images in one space, so a typed description
("black wallet") can be compared directly to a detected crop's image vector —
matching works whether or not the user uploads a photo.

The model is loaded lazily on first use so the API boots instantly and auth /
report endpoints don't pay the CLIP load cost.
"""
from __future__ import annotations

import sys
import threading

import numpy as np

from .config import EDGEAI_DIR

# make `edge_cctv` importable so we stay in lockstep with Process 1's model
if str(EDGEAI_DIR) not in sys.path:
    sys.path.insert(0, str(EDGEAI_DIR))

_lock = threading.Lock()
_model = None
_tokenizer = None
_preprocess = None
_device = None


def _load() -> None:
    global _model, _tokenizer, _preprocess, _device
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        import open_clip
        import torch
        from edge_cctv import config as ec  # same model name/pretrained as EdgeAI

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            ec.CLIP_MODEL_NAME, pretrained=ec.CLIP_PRETRAINED
        )
        model.eval().to(device)
        _model = model
        _tokenizer = open_clip.get_tokenizer(ec.CLIP_MODEL_NAME)
        _preprocess = preprocess
        _device = device


def embed_text(text: str) -> np.ndarray:
    _load()
    import torch

    toks = _tokenizer([text]).to(_device)
    with torch.no_grad():
        feats = _model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy().astype("float32")


def embed_image(path: str) -> np.ndarray:
    _load()
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB")
    tensor = _preprocess(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        feats = _model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy().astype("float32")


def clip_version() -> str:
    """The CLIP half of the model_version contract (detector-independent)."""
    from edge_cctv import config as ec

    return ec.CLIP_MODEL_NAME
