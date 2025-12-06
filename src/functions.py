import re

# captures "98% cotton", "2 % Elastane", etc.
_PCT_NAME_RE = re.compile(r'(\d{1,3})\s*%?\s*([A-Za-z][A-Za-z\- ]+)', re.I)

def canon_mat_name(s: str) -> str:
    # very light canonicalization
    s = s.strip().lower()
    s = s.replace('.', '').replace(',', '')
    s = re.sub(r'\s+', ' ', s)
    return s

def parse_composition_to_dict(text: str) -> dict[str, float]:
    """
    Parse composition like '98% cotton, 2% elastane.' -> {'cotton':0.98, 'elastane':0.02}
    No predefined vocab; unknown names kept as-is (canonicalized).
    If no percents found, returns {}.
    """
    t = str(text)
    pairs = _PCT_NAME_RE.findall(t)
    counts: dict[str, float] = {}
    total = 0.0
    for pct, name in pairs:
        name_c = canon_mat_name(name)
        v = float(pct)
        counts[name_c] = counts.get(name_c, 0.0) + v
        total += v
    if total > 0:
        for k in counts:
            counts[k] /= total
    return counts  