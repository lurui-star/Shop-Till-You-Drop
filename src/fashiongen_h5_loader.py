# fashiongen_h5_loader.py
from pathlib import Path
import zipfile, re, json, random
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict, Counter

import numpy as np
import h5py
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# ==========================
# A) ZIP → ensure H5 present
# ==========================
def ensure_h5_ready(
    src_path: str,
    out_dir: Optional[str] = None,
    need_unzip: Optional[bool] = None,
    force_reextract: bool = False,
    h5_glob: str = "**/*.h5",  # recursive search
) -> List[Path]:
    """
    Ensure FashionGen .h5 files are available.

    - If `src_path` is a .zip, it extracts (only if needed) into `out_dir`
      (default: alongside the zip, folder=<zip_stem>).
    - If `src_path` is a directory, it just looks for .h5 files inside.
    - Returns a list of .h5 Paths.
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Path not found: {src}")

    # Determine if unzip is needed
    if need_unzip is None:
        need_unzip = src.suffix.lower() == ".zip"

    if need_unzip:
        if src.suffix.lower() != ".zip":
            raise ValueError(f"Asked to unzip, but not a .zip: {src}")
        extract_dir = Path(out_dir) if out_dir else src.with_suffix("")
        extract_dir.mkdir(parents=True, exist_ok=True)

        existing_h5 = sorted(extract_dir.rglob(h5_glob))
        if existing_h5 and not force_reextract:
            return existing_h5

        with zipfile.ZipFile(src, "r") as zf:
            for m in zf.infolist():
                if m.filename.startswith("__MACOSX/"):
                    continue
                zf.extract(m, extract_dir)

        h5_files = sorted(extract_dir.rglob(h5_glob))
        if not h5_files:
            raise FileNotFoundError(f"No {h5_glob} found after extraction in {extract_dir}")
        return h5_files
    else:
        if src.is_dir():
            h5_files = sorted(src.rglob(h5_glob))
            if not h5_files:
                raise FileNotFoundError(f"No {h5_glob} found in {src}")
            return h5_files
        if src.suffix.lower() == ".h5":
            return [src]
        raise ValueError(f"Unsupported src_path: {src}")

# ==========================
# B) String utils & canonicalizers
# ==========================
def _decode_bytes(x):
    """
    Robustly decode HDF5 bytes-like values.
    - Handles np.void (fixed-len strings), bytes, bytearray, np.bytes_
    - Strips trailing NULs
    - Tries utf-8, then latin-1, then utf-8 with 'ignore'
    """
    import numpy as _np
    if isinstance(x, _np.void):
        b = bytes(x)
    elif isinstance(x, (bytes, bytearray, _np.bytes_)):
        b = bytes(x)
    else:
        return str(x)
    b = b.split(b"\x00", 1)[0]
    for enc in ("utf-8", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", errors="ignore")

def _as_scalar(x):
    """Turn (1,) arrays or 0-d arrays into a Python scalar for clean decoding."""
    import numpy as _np
    if isinstance(x, _np.ndarray):
        return x.squeeze().item() if x.ndim > 0 else x.item()
    return x

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def canonicalize_category(cat: str) -> str:
    s = str(cat).replace("\\", "/").strip().lower()
    # If categories are hierarchical, keep the most specific tail
    for sep in ["/", ">", "|", ",", ";"]:
        if sep in s:
            s = s.split(sep)[-1]
    s = re.sub(r"\s+", " ", s)
    return s

_GENDER_MAP = {
    "women": "women", "woman": "women", "female": "women",
    "men": "men", "man": "men", "male": "men",
    "unisex": "unisex", "kids": "kids", "kid": "kids",
}
def canonicalize_gender(g: str) -> str:
    s = _norm_text(g)
    return _GENDER_MAP.get(s, s)

# ==========================
# C) Build label vocabulary (category or gender)
# ==========================
def build_vocab_from_h5(
    h5_path: str,
    label_key: str = "category",  # e.g., "input_category" or "input_gender"
    add_other: bool = True,
    min_count: int = 1,
    out_json: str = "label_vocab.json",
) -> Dict[str, int]:
    """
    Build a string-label → id mapping from an H5 dataset.
    Uses canonicalizers for consistent mapping.
    """
    cnt = Counter()
    with h5py.File(h5_path, "r") as f:
        if label_key not in f:
            raise KeyError(f"'{label_key}' not found in {list(f.keys())}")
        d = f[label_key]
        N = d.shape[0]
        is_int = np.issubdtype(d.dtype, np.integer)
        for i in range(N):
            raw = _as_scalar(d[i])
            if is_int:
                x = str(int(raw))
            else:
                x = _decode_bytes(raw)
            x = canonicalize_gender(x) if label_key.lower().endswith("gender") else canonicalize_category(x)
            cnt[x] += 1

    vocab: Dict[str, int] = {}
    idx = 0
    if add_other and not label_key.lower().endswith("gender"):
        vocab["__other__"] = idx
        idx += 1

    # keep all >= min_count
    for k, n in sorted(cnt.items()):
        if n >= min_count and k not in vocab:
            vocab[k] = idx
            idx += 1

    with open(out_json, "w") as f:
        json.dump(vocab, f, indent=2, sort_keys=True)
    print(f"[vocab:{label_key}] Saved {len(vocab)} labels to {out_json}")
    return vocab

# ==========================
# D) Worker-safe H5 Dataset
# ==========================
class FashionGenH5(Dataset):
    """
    Returns dict:
      image: Tensor [3,H,W]
      label: int (category id) or string (if no vocab provided)
      label_raw: raw category string
      gender: int (gender id) or string (if no vocab provided)
      gender_raw: raw gender string
      caption: str
      index: int (raw H5 row index)
    """
    def __init__(
        self,
        h5_path: str,
        image_key: str = "input_image",
        caption_key_candidates: Tuple[str, ...] = ("input_description","input_concat_description","captions","descriptions","caption","description"),
        label_key_candidates: Tuple[str, ...] = ("input_category","category","articleType","class","label"),
        use_random_caption: bool = True,
        image_size: int = 256,
        normalize: str = "imagenet",
        vocab_label_json: Optional[str] = None,   # category vocab
        vocab_gender_json: Optional[str] = None,  # gender vocab
        drop_unknown: bool = False,               # (kept for API symmetry; not filtering here)
        train_transforms: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.image_key = image_key
        self.caption_key_candidates = caption_key_candidates
        self.label_key_candidates = label_key_candidates
        self.use_random_caption = use_random_caption

        # transforms
        if train_transforms is not None:
            self.transform = train_transforms
        else:
            t = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
            if normalize == "imagenet":
                t += [transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
            elif normalize == "gan":
                t += [transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
            self.transform = transforms.Compose(t)

        # load vocabs
        self.label_vocab = None
        self.gender_vocab = None
        self.other_id = None
        if vocab_label_json and Path(vocab_label_json).exists():
            with open(vocab_label_json) as f:
                self.label_vocab = json.load(f)
            self.other_id = self.label_vocab.get("__other__", None)
        if vocab_gender_json and Path(vocab_gender_json).exists():
            with open(vocab_gender_json) as g:
                self.gender_vocab = json.load(g)

        # inspect file & select keys
        with h5py.File(self.h5_path, "r") as f:
            if self.image_key not in f:
                raise KeyError(f"'{self.image_key}' not found in {list(f.keys())}")
            self._length = f[self.image_key].shape[0]
            self.caption_key = next((k for k in self.caption_key_candidates if k in f), None)
            self.label_key   = next((k for k in self.label_key_candidates   if k in f), None)
            self.gender_key  = next((k for k in ("input_gender","gender")   if k in f), None)
            self.keep_idx = np.arange(self._length, dtype=np.int64)

        self._h5 = None

    def __len__(self):
        return len(self.keep_idx)

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _pick_caption(self, cap_entry):
        if isinstance(cap_entry, (bytes, np.bytes_)):
            return _decode_bytes(cap_entry)
        if isinstance(cap_entry, (np.ndarray, list)):
            if len(cap_entry) == 0:
                return ""
            j = random.randrange(len(cap_entry)) if self.use_random_caption else 0
            return _decode_bytes(cap_entry[j])
        return str(cap_entry)

    def _label_to_id(self, raw_label):
        # map raw category string -> id using category vocab
        if isinstance(raw_label, (int, np.integer)):
            return int(raw_label)
        s = canonicalize_category(raw_label)
        if self.label_vocab is None:
            return s
        if s in self.label_vocab:
            return int(self.label_vocab[s])
        if self.other_id is not None:
            return int(self.other_id)
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_i = self.keep_idx[idx]
        f = self._get_h5()

        # image
        img_np = f[self.image_key][real_i]
        im = Image.fromarray(img_np, mode="RGB")
        img = self.transform(im)

        # caption
        caption = ""
        if self.caption_key is not None:
            cap_val = _as_scalar(f[self.caption_key][real_i])
            caption = _decode_bytes(cap_val)

        # category label
        label_raw, label_id = None, None
        if self.label_key is not None:
            d = f[self.label_key]
            is_int = np.issubdtype(d.dtype, np.integer)
            raw = _as_scalar(d[real_i])
            if is_int:
                raw = int(raw)
            else:
                raw = _decode_bytes(raw)
            label_raw = raw
            label_id = self._label_to_id(raw)

        # gender
        gender_id, gender_raw = None, ""
        if self.gender_key is not None:
            dg = f[self.gender_key]
            is_int_g = np.issubdtype(dg.dtype, np.integer)
            rawg = _as_scalar(dg[real_i])
            gender_raw = int(rawg) if is_int_g else _decode_bytes(rawg)
            g = canonicalize_gender(gender_raw)
            if self.gender_vocab is not None and g in self.gender_vocab:
                gender_id = int(self.gender_vocab[g])
            else:
                gender_id = g  # string fallback

        return {
            "image": img,
            "label": label_id,            # category id
            "label_raw": label_raw,       # category string
            "gender": gender_id,          # gender id (or string if no vocab)
            "gender_raw": gender_raw,     # gender string
            "caption": caption,
            "index": int(real_i),
        }

# ==========================
# E) Transforms & builders
# ==========================
def build_transforms(image_size: int = 256, train: bool = False, norm: str = "imagenet"):
    if train:
        t = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
        ]
    else:
        t = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if norm == "imagenet":
        t += [transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
    elif norm == "gan":
        t += [transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
    return transforms.Compose(t)

def make_datasets_from_h5(
    train_h5: str,
    val_h5: Optional[str] = None,
    test_h5: Optional[str] = None,
    *,
    vocab_label_json: Optional[str] = None,
    vocab_gender_json: Optional[str] = None,
    image_key: str = "input_image",
    label_key_candidates: Tuple[str, ...] = ("input_category","category","articleType","class","label"),
    caption_key_candidates: Tuple[str, ...] = ("input_description","input_concat_description","captions","descriptions","caption","description"),
    image_size: int = 256,
    normalize: str = "imagenet",
    drop_unknown: bool = False,
) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    train_tf = build_transforms(image_size, train=True, norm=normalize)
    eval_tf  = build_transforms(image_size, train=False, norm=normalize)

    train_ds = FashionGenH5(
        train_h5, image_key, caption_key_candidates, label_key_candidates,
        True, image_size, normalize,
        vocab_label_json=vocab_label_json,
        vocab_gender_json=vocab_gender_json,
        drop_unknown=drop_unknown, train_transforms=train_tf
    )
    val_ds = FashionGenH5(
        val_h5, image_key, caption_key_candidates, label_key_candidates,
        False, image_size, normalize,
        vocab_label_json=vocab_label_json,
        vocab_gender_json=vocab_gender_json,
        drop_unknown=drop_unknown, train_transforms=eval_tf
    ) if val_h5 else None
    test_ds = FashionGenH5(
        test_h5, image_key, caption_key_candidates, label_key_candidates,
        False, image_size, normalize,
        vocab_label_json=vocab_label_json,
        vocab_gender_json=vocab_gender_json,
        drop_unknown=drop_unknown, train_transforms=eval_tf
    ) if test_h5 else None
    return train_ds, val_ds, test_ds

def make_loaders(
    train_ds: Dataset,
    val_ds: Optional[Dataset] = None,
    test_ds: Optional[Dataset] = None,
    *,
    batch_size: int = 64,
    num_workers: int = 8,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    if use_weighted_sampler:
        labels = np.array([train_ds[i]["label"] for i in range(len(train_ds))])
        max_id = int(np.max(labels))
        counts = np.bincount(labels, minlength=max_id+1)
        weights = 1.0 / np.maximum(counts, 1)
        sample_weights = weights[labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True) if test_ds else None
    return train_loader, val_loader, test_loader

# ==========================
# F) Splitting helpers (for test-from-train)
# ==========================
def read_label_ids_from_h5(
    h5_path: str,
    label_key: str = "input_category",
    vocab_json: Optional[str] = None
) -> np.ndarray:
    other_id = 0
    vocab = None
    if vocab_json:
        with open(vocab_json) as f:
            vocab = json.load(f)
        other_id = vocab.get("__other__", 0)

    with h5py.File(h5_path, "r") as f:
        if label_key not in f:
            raise KeyError(f"'{label_key}' not in {list(f.keys())}")
        d = f[label_key]
        N = d.shape[0]
        is_int = np.issubdtype(d.dtype, np.integer)
        labels = np.empty(N, dtype=np.int64)
        if is_int:
            for i in range(N):
                labels[i] = int(d[i])
        else:
            for i in range(N):
                raw = _as_scalar(d[i])
                s = _decode_bytes(raw)
                s = canonicalize_gender(s) if label_key.lower().endswith("gender") else canonicalize_category(s)
                labels[i] = vocab.get(s, other_id) if vocab else s  # type: ignore
    return labels

def stratified_train_test_indices(
    labels: np.ndarray, train_ratio: float = 0.9, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[int(y)].append(i)
    train_idx, test_idx = [], []
    for _, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_tr = int(n * train_ratio)
        train_idx += idxs[:n_tr]
        test_idx  += idxs[n_tr:]
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)

def make_loaders_with_existing_val(
    *,
    train_h5: str,
    val_h5: str,
    vocab_label_json: str,
    vocab_gender_json: Optional[str] = None,
    image_key: str = "input_image",
    label_key: str = "input_category",
    caption_key_candidates: Tuple[str, ...] = ("input_description","input_concat_description","caption","descriptions"),
    label_key_candidates: Tuple[str, ...] = ("input_category","category","class","label"),
    image_size: int = 256,
    normalize: str = "imagenet",
    train_ratio: float = 0.9,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 8,
    use_weighted_sampler: bool = False,
):
    labels = read_label_ids_from_h5(train_h5, label_key=label_key, vocab_json=vocab_label_json)
    train_idx, test_idx = stratified_train_test_indices(labels, train_ratio, seed)

    train_tf = build_transforms(image_size, True, normalize)
    eval_tf  = build_transforms(image_size, False, normalize)

    train_ds = FashionGenH5(
        train_h5, image_key, caption_key_candidates, label_key_candidates,
        True, image_size, normalize,
        vocab_label_json=vocab_label_json, vocab_gender_json=vocab_gender_json,
        drop_unknown=False, train_transforms=train_tf
    )
    train_ds.keep_idx = train_idx

    test_ds = FashionGenH5(
        train_h5, image_key, caption_key_candidates, label_key_candidates,
        False, image_size, normalize,
        vocab_label_json=vocab_label_json, vocab_gender_json=vocab_gender_json,
        drop_unknown=False, train_transforms=eval_tf
    )
    test_ds.keep_idx = test_idx

    val_ds = FashionGenH5(
        val_h5, image_key, caption_key_candidates, label_key_candidates,
        False, image_size, normalize,
        vocab_label_json=vocab_label_json, vocab_gender_json=vocab_gender_json,
        drop_unknown=False, train_transforms=eval_tf
    )

    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds,
        batch_size=batch_size, num_workers=num_workers, use_weighted_sampler=use_weighted_sampler
    )
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds