#!/usr/bin/env python3
"""Validate and report coverage for the Titan Register dataset (JSONL files).

Usage: python validate_titan.py <dir-or-files...> [--strict] [--quiet]
No network, no models: pure-Python checks (schema, dedup, style, fact preservation,
repetition, coverage against the plan's quotas).
"""
import sys, re, json, glob, os, difflib
from collections import Counter, defaultdict

DOMAIN_MAP = {
    "retail": "retail/shopping", "transport": "transport/commute", "food": "food/cooking",
    "weather": "weather/nature/outdoors", "finance": "finance/money", "school": "school/education",
    "tech": "tech/devices/software", "sports": "sports/games", "holidays": "holidays/celebrations",
}
DOMAIN_QUOTA = {
    "workplace": 200, "home/chores": 100, "home/kids": 100, "family/relationships": 100,
    "social/dating": 100, "customer service": 100, "retail/shopping": 100, "travel": 100,
    "transport/commute": 100, "health/fitness": 100, "food/cooking": 100,
    "weather/nature/outdoors": 100, "finance/money": 100, "school/education": 100,
    "tech/devices/software": 100, "sports/games": 80, "pets": 80, "holidays/celebrations": 80,
    "entertainment": 80, "civic/admin/housing": 80,
}
INTENT_QUOTA = {
    "request": 220, "announce": 200, "question": 200, "small talk": 180, "complaint": 160,
    "instruction": 160, "apology": 150, "compliment": 150, "good news": 150, "bad news": 150,
    "warning": 140, "refusal": 140,
}
MOVES = ["ledger", "inevitability", "correction", "wisdom", "threat", "mission", "mercy",
         "limits", "judgment", "patience"]
LENGTH_SHARE = {"short": 0.25, "medium": 0.45, "long": 0.30}
REGISTER_SHARE = {"casual": 0.30, "neutral": 0.45, "formal": 0.25}
TOTAL = 2000
MAX_SENTENCES = 5
MAX_WORDS = 60

# real contractions only; possessive 's is allowed
CONTRACTION_RE = re.compile(
    r"\b(it|he|she|that|what|there|here|who|let|where|how|this)'s\b|\b\w+'(re|ve|ll|m)\b|n't\b"
    r"|\b(i|he|she|we|they|you|it|that|who)'d\b", re.I)
COSMIC = ["universe", "stars", "fate", "destiny", "cosmos", "galaxy"]
# Fragments of well-known lines the register must not reproduce (filter only).
BLOCKLIST = ["inevitable", "perfectly balanced", "as all things should be", "dread it", "run from it",
             "destiny arrives", "little one", "strongest wills", "fun isn't", "reduced to atoms",
             "half of all", "gone for the head", "the work is done", "i hope they remember"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORDINAL = {1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth", 9: "ninth", 12: "twelfth"}


def cardinal(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return TENS[t] + ("" if o == 0 else "-" + ONES[o])
    if n < 1000:
        h, r = divmod(n, 100)
        return ONES[h] + " hundred" + ("" if r == 0 else " " + cardinal(r))
    if n < 1000000:
        th, r = divmod(n, 1000)
        return cardinal(th) + " thousand" + ("" if r == 0 else " " + cardinal(r))
    return str(n)


def ordinal(n):
    if n in ORDINAL:
        return ORDINAL[n]
    if n < 20:
        return cardinal(n) + "th"
    if n >= 100:
        head, tail = n - n % 100, n % 100
        return cardinal(head) + ("th" if tail == 0 else " " + ordinal(tail))
    if n % 10 == 0:
        return cardinal(n)[:-1] + "ieth"
    t, o = divmod(n, 10)
    return TENS[t] + "-" + ordinal(o)


def num_words(n):
    """Spelled-out forms a target might use for integer n."""
    c, o = cardinal(n), ordinal(n)
    forms = {c, c.replace("-", " "), c.replace(" hundred ", " hundred and "), o, o.replace("-", " ")}
    if 1000 <= n < 10000 and n % 100 == 0:
        forms.add(cardinal(n // 100) + " hundred")          # eighteen hundred
    if n == 15:
        forms.update({"quarter past", "a quarter"})
    if n == 30:
        forms.update({"half past", "half an hour", "half"})
    if n == 45:
        forms.add("quarter to")
    return forms


def has_any(low, forms):
    return any(f in low for f in forms)


def time_ok(h, mm, low):
    H, M = int(h), int(mm)
    nxt = H % 12 + 1
    if f"{H}:{mm}" in low:
        return True
    if M == 0:
        return has_any(low, num_words(H))
    if has_any(low, num_words(H)) and has_any(low, num_words(M)):
        return True
    if M == 15 and "quarter past" in low and has_any(low, num_words(H)):
        return True
    if M == 30 and "half past" in low and has_any(low, num_words(H)):
        return True
    if M == 45 and "quarter to" in low and has_any(low, num_words(nxt)):
        return True
    if M > 30 and f"{cardinal(60 - M)} to" in low and has_any(low, num_words(nxt)):
        return True
    return False


def source_length_bucket(src):
    n = len(src.split())
    return "short" if n <= 6 else ("medium" if n <= 15 else "long")


def register_guess(src):
    s = src.strip()
    if re.search(r"\b(pls|u|thx|tmrw|lol|ugh|gonna|wanna|omg|idk|rn|btw)\b", s, re.I) or (s and s[0].islower()):
        return "casual"
    if re.search(r"\b(please be advised|dear|hereby|required|must|effective|advisory|notice|policy|kindly|we regret)\b", s, re.I) or len(s.split()) > 22:
        return "formal"
    return "neutral"


def sentences(t):
    return [s for s in re.split(r"(?<=[.?!])\s+", t.strip()) if s]


def load(paths):
    rows, errors = [], []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"{p}:{ln} JSON error: {e}")
                    continue
                for k in ("id", "domain", "intent", "move", "source", "target"):
                    if k not in r:
                        errors.append(f"{p}:{ln} missing field {k}")
                r["domain"] = DOMAIN_MAP.get(r.get("domain"), r.get("domain"))
                r.setdefault("length", source_length_bucket(r.get("source", "")))
                r.setdefault("register", register_guess(r.get("source", "")))
                r["_file"] = os.path.basename(p)
                rows.append(r)
    return rows, errors


def check_row(r):
    issues = []
    src, tgt = r["source"], r["target"]
    low = tgt.lower().replace("-", " ")
    if r["domain"] not in DOMAIN_QUOTA:
        issues.append(f"unknown domain '{r['domain']}'")
    if r["intent"] not in INTENT_QUOTA:
        issues.append(f"unknown intent '{r['intent']}'")
    if r["move"] not in MOVES:
        issues.append(f"unknown move '{r['move']}'")
    if r.get("length") not in LENGTH_SHARE or r.get("register") not in REGISTER_SHARE:
        issues.append("bad length/register tag")
    elif r["length"] != source_length_bucket(src):
        issues.append(f"length tag '{r['length']}' but source is {source_length_bucket(src)}")
    if CONTRACTION_RE.search(tgt):
        issues.append("contraction in target")
    if "!" in tgt:
        issues.append("exclamation mark in target")
    ns = len(sentences(tgt))
    if ns > MAX_SENTENCES:
        issues.append(f"{ns} sentences (>{MAX_SENTENCES})")
    nw = len(tgt.split())
    if nw > MAX_WORDS:
        issues.append(f"{nw} words (>{MAX_WORDS})")
    if sum(low.count(c) for c in COSMIC) > 1:
        issues.append("more than one cosmic noun")
    for b in BLOCKLIST:
        if b in low:
            issues.append(f"blocklisted fragment '{b}'")
    # fact preservation: clock times first, then numbers, weekdays, months, mid-sentence names
    rest = src
    for m in re.finditer(r"\b(\d{1,2}):(\d{2})\b", src):
        if not time_ok(m.group(1), m.group(2), low):
            issues.append(f"time '{m.group(0)}' not preserved")
        rest = rest.replace(m.group(0), " ")
    for m in re.finditer(r"\$?\d[\d,]*(?:\.\d+)?", rest):
        tok = m.group(0).replace("$", "").replace(",", "")
        if "." in tok:
            whole, frac = tok.split(".")
            ok = tok in low or (tok == "0.25" and "quarter" in low) or (tok == "0.5" and "half" in low) or \
                 (has_any(low, num_words(int(whole))) and has_any(low, num_words(int(frac))))
        else:
            ok = tok in low or has_any(low, num_words(int(tok)))
        if not ok:
            issues.append(f"number '{m.group(0)}' not preserved")
    for d in WEEKDAYS + MONTHS:
        if d == "may":   # modal verb unless used as a month: "May 3", "3 May", "in/by/until May"
            present = re.search(r"(\d\w*\s+May\b|\bMay\s+\d|\b(in|by|until|of|before|after|this|next|last)\s+May\b)", src)
        else:
            present = re.search(rf"\b{d}\b", src, re.I)
        if present and d not in low:
            issues.append(f"'{d}' not preserved")
    for m in re.finditer(r"(?<![.?!]\s)(?<!^)\b([A-Z][a-z]{2,})\b", src):
        name = m.group(1)
        if name.lower() in WEEKDAYS + MONTHS:
            continue
        if name.lower() not in low:
            issues.append(f"name '{name}' not preserved")
    return issues


STOP = set("the a an and or of to in is it i you that will not be for this with as on at by do have has "
           "are was were your my we they them its their one once from than then there here when what which "
           "who if so no yes into out up down but can may would should could all any each every some more most "
           "very only also just now me us him her his".split())


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strict, quiet = "--strict" in sys.argv, "--quiet" in sys.argv
    paths = []
    for a in args or ["."]:
        paths += sorted(glob.glob(os.path.join(a, "titan_pairs_*.jsonl"))) if os.path.isdir(a) else [a]
    rows, errors = load(paths)
    print(f"files: {len(paths)}  rows: {len(rows)}")
    for e in errors:
        print("ERROR", e)

    ids = [r["id"] for r in rows]
    dup_ids = [i for i, c in Counter(ids).items() if c > 1]
    if dup_ids:
        print("DUPLICATE IDS:", dup_ids)

    flagged = 0
    for r in rows:
        iss = check_row(r)
        if iss:
            flagged += 1
            print(f"[{r['id']}] " + "; ".join(iss))
    print(f"rows flagged: {flagged}")

    for field in ("source", "target"):
        seen = {}
        normed = [(r["id"], norm(r[field])) for r in rows]
        for id1, s1 in normed:
            if s1 in seen:
                print(f"EXACT DUP {field}: {id1} == {seen[s1]}")
            seen[s1] = id1
        # candidate pairs share >= 60% of the shorter row's word set (inverted index), then difflib
        near = 0
        toksets = [set(s.split()) for _, s in normed]
        index = defaultdict(set)
        for i, ts in enumerate(toksets):
            for w in ts:
                if w not in STOP:
                    index[w].add(i)
        for i in range(len(normed)):
            shared = Counter()
            for w in toksets[i]:
                if w not in STOP:
                    for j in index[w]:
                        if j > i:
                            shared[j] += 1
            for j, c in shared.items():
                small = min(len(toksets[i] - STOP), len(toksets[j] - STOP)) or 1
                if c / small < 0.6:
                    continue
                a, b = normed[i][1], normed[j][1]
                if abs(len(a) - len(b)) > 0.3 * max(len(a), len(b)):
                    continue
                if difflib.SequenceMatcher(None, a, b).ratio() > 0.85:
                    print(f"NEAR DUP {field}: {normed[i][0]} ~ {normed[j][0]}")
                    near += 1
        print(f"near-duplicate {field}s: {near}")

    grams, words = Counter(), Counter()
    for r in rows:
        toks = norm(r["target"]).split()
        for w in toks:
            if w not in STOP and len(w) > 2:
                words[w] += 1
        for g in set(zip(toks, toks[1:], toks[2:], toks[3:])):
            grams[g] += 1
    rep = [(" ".join(g), c) for g, c in grams.items() if c > 4]
    print(f"4-grams used in >4 targets: {len(rep)}")
    for g, c in sorted(rep, key=lambda x: -x[1])[:20]:
        print(f"   {c:3d}  {g}")
    print("top content words in targets:")
    print("   " + ", ".join(f"{w}:{c}" for w, c in words.most_common(25)))

    if quiet:
        return
    n = len(rows)
    scale = n / TOTAL

    def table(title, counter, quota):
        print(f"\n{title:<26}{'count':>6}{'quota@n':>9}{'dev':>7}")
        for k in quota:
            q = quota[k] * scale
            c = counter.get(k, 0)
            dev = (c - q) / q * 100 if q else 0
            flag = "  <--" if abs(dev) > 8 else ""
            print(f"  {k:<24}{c:>6}{q:>9.1f}{dev:>+6.0f}%{flag}")
        extra = [k for k in counter if k not in quota]
        if extra:
            print("  (unmapped:", extra, ")")

    table("domain", Counter(r["domain"] for r in rows), DOMAIN_QUOTA)
    table("intent", Counter(r["intent"] for r in rows), INTENT_QUOTA)
    table("move", Counter(r["move"] for r in rows), {m: TOTAL // len(MOVES) for m in MOVES})
    table("source length", Counter(r["length"] for r in rows), {k: v * TOTAL for k, v in LENGTH_SHARE.items()})
    table("source register", Counter(r["register"] for r in rows), {k: v * TOTAL for k, v in REGISTER_SHARE.items()})
    tl = Counter(min(len(sentences(r["target"])), 5) for r in rows)
    print("\ntarget sentences: " + ", ".join(f"{k}:{tl[k]}" for k in sorted(tl)))

    spread = defaultdict(lambda: [set(), set()])
    for r in rows:
        spread[r["domain"]][0].add(r["intent"])
        spread[r["domain"]][1].add(r["move"])
    thin = {d: (len(a), len(b)) for d, (a, b) in spread.items() if len(a) < 6 or len(b) < 6}
    print("domains with <6 intents or <6 moves:", thin if thin else "none")

    if strict and (errors or flagged or dup_ids):
        sys.exit(1)


if __name__ == "__main__":
    main()
