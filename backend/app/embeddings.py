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

# ViT-B-32/openai is trained on English only. Measured against real crops, a
# correct Thai phrase ("กระเป๋าสตางค์สีดำ") scores 0.19-0.22 — the same band as
# keyboard mash — while its English equivalent reaches 0.28. No threshold can
# separate Thai from noise, so map the vocabulary this app actually sees (the
# things people lose, plus colours) into English before embedding. Thai has no
# word spacing, hence substring matching rather than a token split.
_TH_EN = {
    # bags — the most-reported category, and COCO detects several of them
    "กระเป๋าสตางค์": "wallet", "สตางค์": "wallet",
    "กระเป๋าเป้": "backpack", "เป้": "backpack",
    "กระเป๋าถือ": "handbag", "กระเป๋าสะพาย": "shoulder bag",
    "กระเป๋าเดินทาง": "suitcase", "กระเป๋า": "bag",
    # electronics
    "โน้ตบุ๊ก": "laptop", "โน๊ตบุ๊ค": "laptop", "แล็ปท็อป": "laptop",
    "แลปท็อป": "laptop", "คอมพิวเตอร์": "laptop computer", "คอม": "laptop computer",
    "แท็บเล็ต": "tablet", "ไอแพด": "tablet",
    "โทรศัพท์": "cell phone", "มือถือ": "cell phone", "ไอโฟน": "iphone cell phone",
    "หูฟัง": "headphones", "เมาส์": "computer mouse", "คีย์บอร์ด": "keyboard",
    "สายชาร์จ": "charger cable", "ที่ชาร์จ": "charger",
    "พาวเวอร์แบงค์": "power bank", "แบตสำรอง": "power bank",
    # everyday objects
    "แว่นตา": "glasses", "แว่น": "glasses", "นาฬิกา": "watch", "ร่ม": "umbrella",
    "ขวดน้ำ": "water bottle", "ขวด": "bottle", "แก้ว": "cup",
    "หนังสือ": "book", "สมุด": "notebook", "ปากกา": "pen", "ดินสอ": "pencil",
    "กุญแจ": "keys", "บัตรนักศึกษา": "student id card", "บัตร": "card",
    "หมวก": "hat", "รองเท้า": "shoes", "เสื้อ": "shirt", "ตุ๊กตา": "teddy bear",
    # colours
    "สีดำ": "black", "ดำ": "black", "สีขาว": "white", "ขาว": "white",
    "สีแดง": "red", "แดง": "red", "สีน้ำเงิน": "blue", "น้ำเงิน": "blue",
    "สีฟ้า": "light blue", "ฟ้า": "light blue", "สีเขียว": "green", "เขียว": "green",
    "สีเหลือง": "yellow", "เหลือง": "yellow", "สีชมพู": "pink", "ชมพู": "pink",
    "สีม่วง": "purple", "ม่วง": "purple", "สีส้ม": "orange", "ส้ม": "orange",
    "สีน้ำตาล": "brown", "น้ำตาล": "brown", "สีเทา": "grey", "เทา": "grey",
    "สีทอง": "gold", "ทอง": "gold", "สีเงิน": "silver", "เงิน": "silver",
}
# longest first so "กระเป๋าสตางค์" wins over "กระเป๋า", "สีน้ำเงิน" over "เงิน"
_TH_KEYS = sorted(_TH_EN, key=len, reverse=True)


def _to_english_offline(text: str) -> str:
    """Dictionary pass. Unknown words are dropped unless nothing matched, in which
    case the original is returned so the caller still gets a (weak) vector rather
    than an empty one."""
    rest = text
    hits = []
    for th in _TH_KEYS:
        if th in rest:
            hits.append(_TH_EN[th])
            rest = rest.replace(th, " ")
    leftover = " ".join(w for w in rest.split() if w.isascii())
    return " ".join(x for x in (*hits, leftover) if x).strip() or text


def to_english(text: str) -> str:
    """Rewrite a Thai query into the English CLIP understands.

    Gemini handles the open vocabulary (the dictionary only knows the words we
    thought of); the dictionary covers it when the key is unset or the call
    fails, so search still works with the booth's wifi down.
    """
    from . import llm

    return llm.translate_to_english(text) or _to_english_offline(text)


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

    toks = _tokenizer([to_english(text)]).to(_device)
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
