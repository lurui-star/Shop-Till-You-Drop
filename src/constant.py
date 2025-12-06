# 1) Canonical base set (expand if needed)
BASE_MATS = {
    "cotton","polyester","nylon","polyamide","viscose","rayon","elastane","spandex","lycra",
    "wool","cashmere","alpaca","mohair","silk","linen","hemp","ramie",
    "leather","sheepskin","lambskin","goatskin","calfskin","suede","fur","down","feather",
    "acrylic","acetate","triacetate","modal","lyocell","tencel","cupro","bemberg",
    "polypropylene","polyethylene","pbt","pet",
    "polyurethane","pu","pvc","rubber","latex",
    # optional keepers for accessories/shoes/bags; drop if clothing-only:
    "brass","steel","stainless steel","silver","gold","titanium","aluminum","ceramic","glass","crystal","wood","paper","straw","raffia"
}

# 2) Alias map (typos, synonyms → canonical)
ALIASES = {
    # nylons
    "polyamide": "nylon", "poliamide": "nylon", "polyamid": "nylon", "polymide": "nylon",
    "poylamide": "nylon", "ny": "nylon", "mylon": "nylon",
    # elastics
    "spandex": "elastane", "lycra": "elastane", "lastol": "elastane",
    # rayons
    "rayon": "viscose", "visc": "viscose", "rayon-viscose": "viscose",
    # lyocell/tencel
    "tencell": "tencel", "lyocel": "lyocell",
    # cupro
    "bemberg": "cupro", "cupra": "cupro",
    # leathers grouped (optional: collapse all to 'leather')
    "sheepskin": "leather", "lambskin": "leather", "goatskin": "leather", "calfskin": "leather",
    "kidskin": "leather", "cowhide": "leather", "cow leather": "leather", "nappa leather": "leather",
    "suede": "leather",
    # furs (collapse)
    "rabbit fur": "fur", "fox fur": "fur", "coyote fur": "fur", "mink fur": "fur",
    # polyurethanes
    "polyurethane": "pu", "polyeurethane": "pu", "polyuretHane": "pu",
    # pvc
    "polyvinyl chloride": "pvc", "polyvinylchloride": "pvc",
    # typos (examples from your list)
    "coton": "cotton", "cototn": "cotton", "cotto": "cotton", "curpo": "cupro",
    "wooll": "wool", "wol": "wool", "woo": "wool",
}
