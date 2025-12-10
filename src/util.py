import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from tqdm.auto import tqdm

def _safe_idx2name(idx2name):
    return (lambda i: idx2name.get(int(i), str(i))) if isinstance(idx2name, dict) else (lambda i: str(i))


def load_report(path):
    path = Path(path)
    if not path.exists():
        print(f"[warn] missing: {path}")
        return []
    with open(path) as f:
        return json.load(f)

def _micro_macro_acc(report):
    if not report:
        return 0.0, 0.0, 0
    total = sum(r["support"] for r in report)
    correct = sum(int(r["support"] * r["accuracy"]) for r in report)
    micro = (correct / total) if total else 0.0
    macro = sum(r["accuracy"] for r in report) / len(report)
    return micro, macro, total

def print_report(report, title="Report", topn=15, sort_by="support", reverse=True):
    """
    sort_by: 'support' or 'accuracy'
    reverse: True -> descending
    """
    if not report:
        print(f"{title}: (empty)")
        return
    key = (lambda r: r["support"]) if sort_by == "support" else (lambda r: r["accuracy"])
    rows = sorted(report, key=key, reverse=reverse)
    micro, macro, total = _micro_macro_acc(report)
    print(f"\n=== {title} ===")
    print(f"classes={len(report)} | total={total} | micro@1={micro:.4f} | macro@1={macro:.4f}")
    print(f"{'id':>3}  {'name':<24} {'support':>8}  {'acc':>6}")
    print("-"*48)
    for r in rows[:topn]:
        print(f"{r['id']:>3}  {r['name']:<24} {r['support']:>8}  {r['accuracy']:.3f}")

def print_worst(report, title="Worst classes", k=15, min_support=10):
    if not report:
        return print(f"{title}: (empty)")
    rows = [r for r in report if r["support"] >= min_support]
    rows.sort(key=lambda r: r["accuracy"])  # ascending = worst first
    print(f"\n--- {title} (k={k}, min_support={min_support}) ---")
    print(f"{'id':>3}  {'name':<24} {'support':>8}  {'acc':>6}")
    print("-"*48)
    for r in rows[:k]:
        print(f"{r['id']:>3}  {r['name']:<24} {r['support']:>8}  {r['accuracy']:.3f}")

def print_confmat(cm, idx2name, title="Confusion Matrix", max_classes=20):
    """
    Pretty print (truncated) confusion matrix.
    Rows = true, Cols = pred. Truncates to top-left max_classes x max_classes.
    """
    if cm is None or cm.size == 0:
        print(f"{title}: (empty)")
        return
    n = cm.shape[0]
    k = min(max_classes, n)
    name = _safe_idx2name(idx2name)
    # header
    col_names = [name(j)[:8] for j in range(k)]
    print(f"\n=== {title} (showing {k} of {n} classes) ===")
    print("true\\pred | " + " ".join(f"{c:>8}" for c in col_names))
    print("-" * (11 + 9 * k))
    for i in range(k):
        row = " ".join(f"{cm[i, j]:>8d}" for j in range(k))
        print(f"{name(i)[:8]:>8} | {row}")

def print_top_confusions(cm, idx2name, title="Top Confusions", k=20, min_support=10):
    """
    List largest off-diagonal (true!=pred) counts, optionally filtering rows with small support.
    """
    if cm is None or cm.size == 0:
        print(f"{title}: (empty)")
        return
    name = _safe_idx2name(idx2name)
    support = cm.sum(axis=1)
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        if support[i] < min_support:  # skip rare true classes
            continue
        for j in range(n):
            if i == j: 
                continue
            c = int(cm[i, j])
            if c > 0:
                pairs.append((c, i, j))
    pairs.sort(reverse=True)
    print(f"\n--- {title} (k={k}, min_support={min_support}) ---")
    print(f"{'count':>6}  {'true':<20} -> {'pred':<20}  {'row_err%':>8}")
    print("-" * 64)
    for c, i, j in pairs[:k]:
        row_err = (c / max(1, support[i])) * 100.0
        print(f"{c:>6}  {name(i):<20} -> {name(j):<20}  {row_err:8.2f}")

def print_normalized_per_row(cm, idx2name, title="Row-Normalized CM (top-left)", max_classes=20, precision=3):
    """Row-normalize and print a compact corner; useful to see error distribution by class."""
    if cm is None or cm.size == 0:
        print(f"{title}: (empty)")
        return
    n = cm.shape[0]
    k = min(max_classes, n)
    rn = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    name = _safe_idx2name(idx2name)
    col_names = [name(j)[:8] for j in range(k)]
    print(f"\n=== {title} (showing {k} of {n} classes) ===")
    print("true\\pred | " + " ".join(f"{c:>8}" for c in col_names))
    print("-" * (11 + 9 * k))
    fmt = f"{{:>8.{precision}f}}"
    for i in range(k):
        row = " ".join(fmt.format(rn[i, j]) for j in range(k))
        print(f"{name(i)[:8]:>8} | {row}")




def plot_confmat(cm, idx2name=None, title="Confusion Matrix", normalize=False, max_classes=None, annotate=False):
    """
    cm: numpy array [C,C]
    idx2name: dict[int->str] or list of class names
    normalize: row-normalize to show per-class error distribution
    max_classes: if set, show top-left KxK block for readability
    annotate: write values in the cells (can be heavy for large C)
    """
    if cm is None or cm.size == 0:
        print(f"{title}: (empty)"); return

    cm = cm.copy().astype(float)
    C = cm.shape[0]
    K = min(C, max_classes) if max_classes else C

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm / np.clip(row_sums, 1, None)

    fig = plt.figure(figsize=(max(5, K*0.5), max(4, K*0.5)))
    ax = plt.gca()
    im = ax.imshow(cm[:K, :K])  # default colormap

    # tick labels
    if idx2name is None:
        labels = [str(i) for i in range(C)]
    elif isinstance(idx2name, dict):
        labels = [idx2name.get(i, str(i)) for i in range(C)]
    else:  # list/sequence
        labels = [str(x) for x in idx2name]

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([labels[j][:20] for j in range(K)], rotation=90)
    ax.set_yticklabels([labels[i][:20] for i in range(K)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # optional annotations
    if annotate:
        fmt = "{:.2f}" if normalize else "{:d}"
        for i in range(K):
            for j in range(K):
                ax.text(j, i, fmt.format(cm[i, j]),
                        ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.show()


def plot_top_confusions_bar(cm, idx2name=None, k=20, min_support=10, title="Top Confusions"):
    """
    Show a bar chart of the largest off-diagonal counts.
    """
    if cm is None or cm.size == 0:
        print(f"{title}: (empty)"); return
    C = cm.shape[0]
    support = cm.sum(axis=1)
    pairs = []
    for i in range(C):
        if support[i] < min_support: 
            continue
        for j in range(C):
            if i == j: 
                continue
            c = int(cm[i, j])
            if c > 0:
                pairs.append((c, i, j))
    if not pairs:
        print("No confusions above thresholds."); return

    pairs.sort(reverse=True)
    top = pairs[:k]
    # names
    if idx2name is None:
        name = lambda x: str(x)
    elif isinstance(idx2name, dict):
        name = lambda x: idx2name.get(int(x), str(x))
    else:
        name = lambda x: str(idx2name[int(x)])
    labels = [f"{name(i)} → {name(j)}"[:40] for _, i, j in top]
    counts = [c for c, _, _ in top]

    fig = plt.figure(figsize=(8, max(4, len(top)*0.35)))
    ax = plt.gca()
    y = np.arange(len(top))
    ax.barh(y, counts)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()



@torch.no_grad()
def embed_loader(model, loader, device):
    model.eval()
    ids, vecs, metas = [], [], []
    for batch in tqdm(loader, desc="Embedding catalog"):
        imgs = batch["images"].to(device)
        out  = model(imgs)                 # expects FashionMultiTaskModel forward
        z    = out["img_feat"]             # [B, D]
        z    = torch.nn.functional.normalize(z, dim=-1)  # cosine-friendly
        vecs.append(z.cpu().numpy())

        # collect ids + metadata for later filtering
        idxs = batch.get("idxs")
        ids  += [int(i) for i in (idxs.tolist() if torch.is_tensor(idxs) else idxs)]
        metas += [{
            "category": (int(batch["y_cat"][i]) if batch.get("y_cat") is not None else None),
            "gender":   (int(batch["y_gender"][i]) if batch.get("y_gender") is not None else None),
            "material": (int(batch["y_material"][i]) if batch.get("y_material") is not None else None),
            # add more: price, brand, color, product_id, etc. if available in batch["meta"]
        } for i in range(len(imgs))]
    return ids, np.vstack(vecs), metas
