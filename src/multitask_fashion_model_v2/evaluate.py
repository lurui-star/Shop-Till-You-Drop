import math, torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# --- tiny helpers for caption metrics ---
from collections import Counter

def _ngrams(seq, n): 
    return [tuple(seq[i:i+n]) for i in range(max(0, len(seq)-n+1))]

def _bleu_k(ref_tokens, hyp_tokens, k):
    r = Counter(_ngrams(ref_tokens, k)); h = Counter(_ngrams(hyp_tokens, k))
    overlap = sum(min(h[g], r[g]) for g in h)
    total = max(1, sum(h.values()))
    return overlap / total

def _brevity_penalty(r_len, h_len):
    if h_len == 0: return 0.0
    if h_len > r_len: return 1.0
    return math.exp(1 - r_len / h_len)

def _bleu_1_2_4(ref, hyp):
    r = ref.lower().split(); h = hyp.lower().split()
    b1 = _bleu_k(r, h, 1)
    b2 = (_bleu_k(r, h, 1) * _bleu_k(r, h, 2)) ** 0.5
    b4_geo = 1.0
    for n in (1,2,3,4):
        b4_geo *= max(1e-9, _bleu_k(r, h, n))
    b4 = b4_geo ** 0.25
    bp = _brevity_penalty(len(r), len(h))
    return bp*b1, bp*b2, bp*b4

def _rouge_l(ref, hyp):
    r = ref.lower().split(); h = hyp.lower().split()
    m, n = len(r), len(h)
    if m == 0 or n == 0: return 0.0, 0.0, 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            dp[i+1][j+1] = dp[i][j]+1 if r[i]==h[j] else max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    P = lcs / max(1, n); R = lcs / max(1, m)
    F1 = 0.0 if (P+R)==0 else 2*P*R/(P+R)
    return P, R, F1

@torch.no_grad()
def evaluate(model, loader, tokenizer, device, *, decode_mode="beam", max_len=16):
    """
    Eval with tqdm progress bar, head-specific denominators, caption NLL/ppl,
    and decoded caption metrics (BLEU-1/2/4, ROUGE-L).
    """
    model.eval()
    cat_ok = gen_ok = mat_ok = 0
    cat_n  = gen_n  = mat_n  = 0

    cap_nll_sum = 0.0
    cap_tok_cnt = 0

    # caption text metrics
    bleu1 = bleu2 = bleu4 = rlP = rlR = rlF = 0.0
    cap_pairs = 0  # number of caption pairs compared

    pbar = tqdm(loader, desc="Eval", leave=False)
    for batch in pbar:
        imgs = batch["images"].to(device)
        y_cat = batch["y_cat"]
        y_gen = batch["y_gender"]
        y_mat = batch.get("y_material", None)

        cap_in  = batch.get("cap_in")
        cap_tgt = batch.get("cap_tgt")

        cap_in  = cap_in.to(device)   if cap_in  is not None else None
        cap_tgt = cap_tgt.to(device)  if cap_tgt is not None else None

        out = model(imgs, cap_in=cap_in)

        # --- top-1s with per-head denominators ---
        if y_cat is not None and "logits_category" in out:
            pred = out["logits_category"].argmax(dim=-1).cpu()
            cat_ok += (pred == y_cat).sum().item()
            cat_n  += y_cat.numel()

        if y_gen is not None and "logits_gender" in out:
            pred = out["logits_gender"].argmax(dim=-1).cpu()
            gen_ok += (pred == y_gen).sum().item()
            gen_n  += y_gen.numel()

        if y_mat is not None and "logits_material" in out:
            pred = out["logits_material"].argmax(dim=-1).cpu()
            mat_ok += (pred == y_mat).sum().item()
            mat_n  += y_mat.numel()

        # --- caption token NLL/ppl ---
        if cap_tgt is not None and "logits_caption" in out:
            B, T, V = out["logits_caption"].shape
            loss_sum = F.cross_entropy(
                out["logits_caption"].reshape(B*T, V),
                cap_tgt.reshape(B*T),
                ignore_index=tokenizer.pad_id,
                reduction="sum",
            )
            cap_nll_sum += loss_sum.item()
            cap_tok_cnt += (cap_tgt != tokenizer.pad_id).sum().item()

        # --- decoded caption metrics (optional but useful) ---
        if cap_tgt is not None:
            # re-use backbone features to decode; choose mode
            z = model.backbone(imgs)
            if decode_mode == "beam":
                from multitask_fashion_model_v2 import decoding as dec
                Y = dec.beam_search(model.caption_head, z, tokenizer.bos_id, tokenizer.eos_id,
                                    max_len=max_len, beam_size=5)
            elif decode_mode == "nucleus":
                from multitask_fashion_model_v2 import decoding as dec
                Y = dec.nucleus(model.caption_head, z, tokenizer.bos_id, tokenizer.eos_id,
                                max_len=max_len, top_p=0.9, temperature=0.9)
            else:
                from multitask_fashion_model_v2 import decoding as dec
                Y = dec.greedy(model.caption_head, z, tokenizer.bos_id, tokenizer.eos_id,
                               max_len=max_len)

            hyp_txts = [tokenizer.decode(seq.tolist()) for seq in Y]
            # turn cap_tgt back into text to compare apples-to-apples
            ref_txts = []
            for seq in batch["cap_tgt"]:
                ref_txts.append(tokenizer.decode(seq.tolist()))

            for ref, hyp in zip(ref_txts, hyp_txts):
                b1, b2, b4 = _bleu_1_2_4(ref, hyp)
                p, r, f = _rouge_l(ref, hyp)
                bleu1 += b1; bleu2 += b2; bleu4 += b4
                rlP += p; rlR += r; rlF += f
                cap_pairs += 1

    results = {
        "category_top1": (cat_ok / cat_n) if cat_n else 0.0,
        "gender_top1":   (gen_ok / gen_n) if gen_n else 0.0,
        "material_top1": (mat_ok / mat_n) if mat_n else 0.0,
    }
    if cap_tok_cnt > 0:
        nll_tok = cap_nll_sum / cap_tok_cnt
        results["caption_nll_per_token"] = nll_tok
        results["caption_ppl"] = math.exp(nll_tok)
    if cap_pairs > 0:
        results.update({
            "BLEU-1": bleu1 / cap_pairs,
            "BLEU-2": bleu2 / cap_pairs,
            "BLEU-4": bleu4 / cap_pairs,
            "ROUGE-L_P": rlP / cap_pairs,
            "ROUGE-L_R": rlR / cap_pairs,
            "ROUGE-L_F1": rlF / cap_pairs,
        })
    return results
