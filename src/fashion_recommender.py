# =========================== #
# Recommender (image + text)  #
# =========================== #
import os, re, pickle, json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import faiss
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import textwrap

# ---------- Basic utils ----------
def _ensure_unit(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x)
    return x / n if n > 0 else x

def _denorm_imagenet(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3,1,1)
    x = t * std + mean
    x = (x.clamp(0,1) * 255.0).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    return x

def _name_from_id(v: Optional[int], id2name: Optional[dict]):
    if v is None or id2name is None: return None
    return id2name.get(int(v), str(v))

def _title_for_item(item, id2label, id2gen, id2mat, score=None, prefix:str=""):
    cat = _name_from_id(item.get("label"), id2label)
    gen = _name_from_id(item.get("gender"), id2gen)
    mat = _name_from_id(item.get("material"), id2mat)
    bits = []
    if prefix: bits.append(prefix)          # e.g., "id=12345 | pos=678"
    if cat is not None: bits.append(f"cat: {cat}")
    if gen is not None: bits.append(f"gen: {gen}")
    if mat is not None: bits.append(f"mat: {mat}")
    if score is not None: bits.append(f"s={score:.3f}")
    return " | ".join(bits) if bits else ""

# ---------- Tiny TF-IDF for input_name ----------
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
def _tok(s: str): return _WORD_RE.findall(s.lower()) if s else []

def _tfidf_fit(texts: List[str]):
    vocab, df, docs = {}, {}, []
    for s in texts:
        toks = _tok(s)
        docs.append(toks)
        seen = set()
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)
    N = max(1, len(texts))
    idf = np.zeros(len(vocab), dtype=np.float32)
    for t, j in vocab.items():
        idf[j] = np.log((1 + N) / (1 + df.get(t, 0))) + 1.0
    return vocab, idf, docs

def _tfidf_vec(toks: List[str], vocab: Dict[str,int], idf: np.ndarray):
    v = np.zeros(len(vocab), dtype=np.float32)
    for t in toks:
        j = vocab.get(t)
        if j is not None: v[j] += 1.0
    if v.sum() > 0: v /= np.linalg.norm(v)
    return v * idf

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else 0.0

# ---------- Embedding helpers ----------
@torch.no_grad()
def embed_loader(model, loader, device):
    model.eval()
    ids, vecs, metas, names = [], [], [], []
    for batch in tqdm(loader, desc="Embedding catalog", leave=False):
        imgs = batch["images"].to(device)
        out  = model(imgs)
        z    = torch.nn.functional.normalize(out["img_feat"], dim=-1)
        vecs.append(z.cpu().numpy())

        idxs = batch.get("idxs")
        ids  += [int(i) for i in (idxs.tolist() if torch.is_tensor(idxs) else idxs)]
        B = imgs.size(0)
        metas += [{
            "category": (int(batch["y_cat"][i]) if batch.get("y_cat") is not None else None),
            "gender":   (int(batch["y_gender"][i]) if batch.get("y_gender") is not None else None),
            "material": (int(batch["y_material"][i]) if batch.get("y_material") is not None else None),
        } for i in range(B)]
        # best-effort input_name / caption text if present
        if batch.get("caption_raw") is not None:
            names += [str(batch["caption_raw"][i]) for i in range(B)]
        else:
            names += [""] * B
    return ids, np.vstack(vecs), metas, names

@torch.no_grad()
def embed_images(model, images, device):
    model.eval()
    out = model(images.to(device))
    z   = torch.nn.functional.normalize(out["img_feat"], dim=-1)
    return z.cpu().numpy()

# ---------- Text-aware score (optional name|image) ----------
@torch.no_grad()
def caption_logprob_per_token(model, image_tensor, tok, text: str):
    ids = torch.tensor(tok.encode(text), dtype=torch.long, device=image_tensor.device)
    if ids.numel() < 2:  # need at least 2 for teacher-forcing
        return -1e9
    cap_in  = ids[:-1].unsqueeze(0)
    cap_tgt = ids[1:].unsqueeze(0)
    out = model(image_tensor.unsqueeze(0), cap_in=cap_in)
    logits = out["logits_caption"]          # [1, T-1, V]
    logp = torch.log_softmax(logits, dim=-1)
    tgt_lp = logp.gather(-1, cap_tgt.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    return float(tgt_lp.mean().item())

# ---------- Recommender ----------
class TextAwareRecommender:
    def __init__(self, *, device, model=None, tok=None, dataset=None):
        self.device  = device
        self.model   = model
        self.tok     = tok
        self.dataset = dataset

        self.index = None
        self.side  = None   # {'ids': [], 'meta': [], 'names': [], 'vecs': np.ndarray}
        self.id2pos_ds = None

        # TF-IDF cache
        self._vocab = None
        self._idf   = None
        self._tfidf = None  # matrix [N,V]

    # ----- build/save/load index -----
    def build_from_loader(self, model, loader, out_dir='runs/index', version ="v1"):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ids, vecs, meta, names = embed_loader(model, loader, self.device)

        index = faiss.IndexFlatIP(vecs.shape[1])   # cosine via IP on unit vecs
        index.add(vecs.astype(np.float32))

        side = {"ids": ids, "meta": meta, "names": names, "vecs": vecs}
        with open(Path(out_dir)/f"catalog_side_{version}.pkl", "wb") as f:
            pickle.dump(side, f, protocol=pickle.HIGHEST_PROTOCOL)
        faiss.write_index(index, str(Path(out_dir)/f"faiss_ip_{version}.index"))

        # build TF-IDF once and save
        vocab, idf, docs = _tfidf_fit(names)
        tfidf = np.vstack([_tfidf_vec(d, vocab, idf) for d in docs]) if len(names) else np.zeros((len(names),0), np.float32)
        with open(Path(out_dir)/f"tfidf_{version}.bin", "wb") as f:
            pickle.dump({"vocab":vocab, "idf":idf, "tfidf":tfidf}, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.index, self.side = index, side
        self._vocab, self._idf, self._tfidf = vocab, idf, tfidf
        return side

    def load(self, in_dir="runs/index"):
        with open(Path(in_dir)/"catalog_side.pkl", "rb") as f:
            self.side = pickle.load(f)
        self.index = faiss.read_index(str(Path(in_dir)/"faiss_ip.index"))
        tfidf_path = Path(in_dir)/"tfidf.bin"
        if tfidf_path.exists():
            with open(tfidf_path, "rb") as f:
                blob = pickle.load(f)
            self._vocab, self._idf, self._tfidf = blob["vocab"], blob["idf"], blob["tfidf"]
        else:
            names = self.side.get("names", [""] * len(self.side["ids"]))
            self._vocab, self._idf, docs = _tfidf_fit(names)
            self._tfidf = np.vstack([_tfidf_vec(d, self._vocab, self._idf) for d in docs])

    def attach_dataset(self, ds):
        self.dataset = ds
        rows = np.asarray(ds.keep_idx).astype(int)
        self.id2pos_ds = {int(r): i for i, r in enumerate(rows)}

    def attach_model(self, model, tok, device):
        self.model, self.tok, self.device = model, tok, device

    # ----- core retrieval -----
    def _neighbors(self, q_vec: np.ndarray, topk: int = 200):
        D, I = self.index.search(q_vec[None,:].astype(np.float32), topk)
        return D[0], I[0]

    def _filter(self, idx: int, req_cat, req_gen, req_mat):
        meta = self.side["meta"][idx]
        if req_cat is not None and meta.get("category") != req_cat: return False
        if req_gen is not None and meta.get("gender")   != req_gen: return False
        if req_mat is not None and meta.get("material") != req_mat: return False
        return True

    def _resolve_query(self, query_pos: Optional[int], query_row_id: Optional[int]):
        if (query_pos is None) == (query_row_id is None):
            raise ValueError("Provide exactly one of query_pos or query_row_id.")
        if query_pos is None:
            # map row_id to side position
            if "id2pos" not in self.side:
                self.side["id2pos"] = {int(r): i for i, r in enumerate(self.side["ids"])}
            query_pos = self.side["id2pos"][int(query_row_id)]
        query_pos = int(query_pos)
        query_row_id = int(self.side["ids"][query_pos])
        q_vec = _ensure_unit(self.side["vecs"][query_pos].astype(np.float32))
        q_meta = self.side["meta"][query_pos]
        return query_pos, query_row_id, q_vec, q_meta

    def recommend(
        self,
        *,
        query_pos: Optional[int] = None,
        query_row_id: Optional[int] = None,
        k: int = 12,
        topk_search: int = 300,
        # text-aware toggle + weights
        use_name: bool = False,
        w_img: float = 1.0,
        w_name_name: float = 0.5,
        w_name_given_img: float = 0.0,
        # filters
        must_category: bool = True,
        must_gender: bool = True,
        must_material: bool = False,
    ) -> List[Tuple[int, float]]:
        assert self.index is not None and self.side is not None, "Load or build index first."
        qp, qrid, q_vec, q_meta = self._resolve_query(query_pos, query_row_id)

        req_cat = q_meta.get("category") if must_category else None
        req_gen = q_meta.get("gender")   if must_gender else None
        req_mat = q_meta.get("material") if must_material else None

        scores, idxs = self._neighbors(q_vec, topk=topk_search)

        out = []
        if not use_name:
            # image-only
            for idx, s in zip(idxs, scores):
                if idx < 0: continue
                if int(self.side["ids"][idx]) == qrid: continue
                if not self._filter(idx, req_cat, req_gen, req_mat): continue
                out.append((int(self.side["ids"][idx]), float(s)))
                if len(out) >= k: break
            return out

        # text-aware blend
        assert self._tfidf is not None and self._vocab is not None and self._idf is not None, "TF-IDF not built."
        q_name = self.side.get("names", [""]*len(self.side["ids"]))[qp]
        q_vec_tfidf = _tfidf_vec(_tok(q_name), self._vocab, self._idf)

        for idx, s_ip in zip(idxs, scores):
            if idx < 0: continue
            rid = int(self.side["ids"][idx])
            if rid == qrid: continue
            if not self._filter(idx, req_cat, req_gen, req_mat): continue

            # name↔name similarity
            name_name = _cos(q_vec_tfidf, self._tfidf[idx])

            # optional P(name|image)
            name_given_img = 0.0
            if w_name_given_img > 0 and self.model is not None and self.tok is not None and self.dataset is not None and self.id2pos_ds is not None:
                pos_ds = self.id2pos_ds.get(rid)
                if pos_ds is not None:
                    img = self.dataset[pos_ds]["image"].to(self.device)
                    lp  = caption_logprob_per_token(self.model, img, self.tok, q_name)
                    name_given_img = 1.0 / (1.0 + np.exp(-lp))

            blended = (w_img * float(s_ip)
                       + w_name_name * float(name_name)
                       + w_name_given_img * float(name_given_img))
            out.append((rid, blended))

        out.sort(key=lambda x: -x[1])
        return out[:k]

    def recommend_both(self, **kwargs):
        # image-only
        kw_img = dict(kwargs)
        kw_img.update(dict(use_name=False, w_name_name=0.0, w_name_given_img=0.0))
        img_only = self.recommend(**kw_img)
        # text-aware
        kw_txt = dict(kwargs)
        kw_txt.update(dict(use_name=True))
        txt_aware = self.recommend(**kw_txt)
        return {"image_only": img_only, "text_aware": txt_aware}

    # ----- plotting -----
    @staticmethod
    def _denorm_imagenet(t: torch.Tensor) -> np.ndarray:
        mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3,1,1)
        x = t * std + mean
        return (x.clamp(0,1) * 255.0).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

    @staticmethod
    def _safe_filename(s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in s)

    @staticmethod
    def _fmt_title(item, id2label, id2gen, id2mat, score=None, wrap=28):
        def name(mapper, v):
            if v is None or mapper is None:
                return None
            return mapper.get(int(v), str(v))

        bits = []
        cat = name(id2label, item.get("label"))
        gen = name(id2gen,   item.get("gender"))
        mat = name(id2mat,   item.get("material"))
        if cat: bits.append(f"cat: {cat}")
        if gen: bits.append(f"gen: {gen}")
        if mat: bits.append(f"mat: {mat}")
        if score is not None: bits.append(f"s={score:.3f}")
        txt = " | ".join(bits) if bits else ""
        return "\n".join(textwrap.wrap(txt, width=wrap)) if wrap else txt
    
    def plot_grid(
        self,                         
        pairs,                        
        query_row_id: int,
        *,
        cols: int = 6,
        id2label=None, id2gen=None, id2mat=None,
        title: str | None = None,
        title_size: int = 16,
        tile_size: int = 8,
        wrap: int = 28,               # wrap per-tile titles to avoid overlap
        show_ids: bool = True,
        show_side_pos: bool = True,
        dpi: int = 150,
        save_path: str | None = None  # if None -> auto "runs/viz/query{query_row_id}_recs.png"
    ):
        # --- require dataset attached (or adapt to your signature)
        assert getattr(self, "dataset", None) is not None, "Attach dataset first."
        ds = self.dataset
        if self.id2pos_ds is None:
            rows = np.asarray(ds.keep_idx).astype(int)
            self.id2pos_ds = {int(r): i for i, r in enumerate(rows)}
        id2pos = self.id2pos_ds

        # normalize input
        norm_pairs = []
        for r in pairs:
            if isinstance(r, dict):
                rid = int(r.get("product_id", r.get("row_id", r.get("id"))))
                sc  = float(r.get("score", 0.0))
            else:
                rid, sc = r
                rid, sc = int(rid), float(sc)
            norm_pairs.append((rid, sc))

        # figure layout
        n_recs  = len(norm_pairs)
        n_total = 1 + n_recs
        rows    = int(np.ceil(n_total / cols))
        fig_w   = max(3.0, cols * 2.6)
        fig_h   = max(2.8, rows * 3.0)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

        # ---- query tile ----
        q_pos  = id2pos[query_row_id]
        q_item = ds[q_pos]
        ax = plt.subplot(rows, cols, 1)
        ax.imshow(_denorm_imagenet(q_item["image"]))
        ax.axis("off")
        q_title = "[QUERY]"
        if show_ids:
            sid = f"id={query_row_id}"
            if show_side_pos: sid += f" | pos={q_pos}"
            q_title = sid + "\n" + q_title
        q_title += "\n" + self._fmt_title(q_item, id2label, id2gen, id2mat, wrap=wrap)
        ax.set_title(q_title, fontsize=tile_size)

        # ---- recommendation tiles ----
        for i, (rid, score) in enumerate(norm_pairs, start=2):
            ax = plt.subplot(rows, cols, i)
            try:
                pos  = id2pos[rid]
                item = ds[pos]
                ax.imshow(_denorm_imagenet(item["image"]))
                ax.axis("off")
                t = self._fmt_title(item, id2label, id2gen, id2mat, score=score, wrap=wrap)
                if show_ids:
                    head = f"id={rid}"
                    if show_side_pos: head += f" | pos={pos}"
                    t = head + "\n" + t
                ax.set_title(t, fontsize=tile_size)
            except Exception:
                ax.axis("off")
                ax.set_title(f"[load err] id={rid}", fontsize=tile_size)

        # figure title placed high, leaving room for tiles
        if title is None:
            title = "Recommendations"
        fig.suptitle(title, fontsize=title_size, y=0.995)
        # more breathing room at top; also help avoid overlaps
        plt.subplots_adjust(top=0.93, hspace=0.35, wspace=0.05)

        # auto filename if not provided
        if save_path is None:
            Path("runs/viz").mkdir(parents=True, exist_ok=True)
            save_path = f"runs/viz/query{self._safe_filename(str(query_row_id))}_recs.png"

        plt.savefig(save_path, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
        plt.show()