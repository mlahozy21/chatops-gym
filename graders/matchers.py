"""Typed answer matching.

Raw substring matching is hackable ("definitely not $48,500" contains
"48,500"). Answers are declared as typed values and matched with normalization
plus a negation guard. This is deliberately shallow NLP over free text — the
residual limitation is documented in the README — but it kills the cheap
false-positive classes that substring matching allows.

Supported types:
  {"type": "money", "value": 48500}          -> $48,500 / 48500 / 48,500.00 / 48.5k
  {"type": "text",  "value": "roadmap doc"}  -> normalized containment
  {"type": "number","value": 17}             -> standalone number token
"""
from __future__ import annotations

import re

_NEGATORS = {"not", "isn't", "isnt", "no", "never", "wasn't", "wasnt", "don't",
             "dont", "neither", "nor", "without", "except"}
_NEG_WINDOW = 4  # words before the value in which a negator invalidates it


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w$.,']+", text.lower())


def _money_values(text: str) -> list[tuple[float, int]]:
    """Extract (value, token_index) for money-looking quantities."""
    out = []
    tokens = _tokenize(text)
    for i, tok in enumerate(tokens):
        m = re.fullmatch(r"\$?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(k)?", tok.rstrip(".,"))
        if not m:
            continue
        val = float(m.group(1).replace(",", ""))
        if m.group(2):
            val += float(f"0.{m.group(2)}")
        if m.group(3):
            val *= 1000
        out.append((val, i))
    return out


def _negated(tokens: list[str], idx: int) -> bool:
    window = tokens[max(0, idx - _NEG_WINDOW):idx]
    return any(w.strip("$.,'") in _NEGATORS for w in window)


def match(answer: dict, text: str) -> tuple[bool, str]:
    """Return (matched, reason)."""
    kind = answer["type"]
    if kind == "money":
        target = float(answer["value"])
        tokens = _tokenize(text)
        found = _money_values(text)
        hits = [(v, i) for v, i in found if abs(v - target) < 0.01]
        if not hits:
            return False, f"value {target:g} not found in reply"
        if all(_negated(tokens, i) for _, i in hits):
            return False, f"value {target:g} only appears negated"
        # A different, non-negated money value alongside the right one is
        # ambiguous -> reject (the model must commit to one answer).
        others = [v for v, i in found if abs(v - target) >= 0.01 and not _negated(tokens, i)
                  and v not in (float(x) for x in answer.get("allow_extra", []))]
        if others:
            return False, f"ambiguous reply: also asserts {others}"
        return True, "ok"

    if kind == "text":
        norm = re.sub(r"\s+", " ", text.lower()).strip()
        needle = re.sub(r"\s+", " ", str(answer["value"]).lower()).strip()
        if needle not in norm:
            return False, f"required text {needle!r} not present"
        tokens = _tokenize(text)
        first = needle.split()[0]
        idxs = [i for i, t in enumerate(tokens) if t.strip("$.,'") == first.strip("$.,'")]
        if idxs and all(_negated(tokens, i) for i in idxs):
            return False, "required text only appears negated"
        return True, "ok"

    if kind == "number":
        target = float(answer["value"])
        tokens = _tokenize(text)
        hits = []
        for i, tok in enumerate(tokens):
            try:
                if abs(float(tok.strip("$.,'")) - target) < 1e-9:
                    hits.append(i)
            except ValueError:
                continue
        if not hits:
            return False, f"number {target:g} not found"
        if all(_negated(tokens, i) for i in hits):
            return False, f"number {target:g} only appears negated"
        return True, "ok"

    raise ValueError(f"unknown answer type: {kind}")
