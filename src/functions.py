import torch, re



SPECIAL = {"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
VOCAB_SIZE = 50000  # simple hash vocab to start

def basic_tokenize(s: str, max_len=40):
    toks = re.sub(r"\s+"," ", s.lower().strip()).split()
    ids = [SPECIAL["<bos>"]]
    for t in toks[:max_len-2]:
        ids.append(hash(t) % VOCAB_SIZE)
    ids.append(SPECIAL["<eos>"])
    return ids

def collate(batch):
    imgs   = torch.stack([b["image"] for b in batch], 0)
    y_cat  = torch.tensor([int(b["label"])  for b in batch], dtype=torch.long)
    y_gen  = torch.tensor([int(b["gender"]) for b in batch], dtype=torch.long)
    seqs   = [torch.tensor(basic_tokenize(b.get("caption","")), dtype=torch.long) for b in batch]
    maxL   = max(x.numel() for x in seqs)
    pad_id = SPECIAL["<pad>"]
    cap    = torch.full((len(batch), maxL), pad_id, dtype=torch.long)
    mask   = torch.zeros_like(cap)
    for i,x in enumerate(seqs):
        cap[i,:x.numel()] = x
        mask[i,:x.numel()] = 1
    return {"image": imgs, "y_cat": y_cat, "y_gen": y_gen, "cap_ids": cap, "cap_mask": mask}