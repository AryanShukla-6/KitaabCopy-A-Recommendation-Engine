"""Text normalization shared by indexing and querying (so 'C++' means the same thing in both)."""
from __future__ import annotations

import json
import math
import re

from .config import GENERIC_REGEX

_GENERIC_RE = re.compile(GENERIC_REGEX)


def norm_text(s: str | None) -> str:
    """Lowercase, make programming-language names tokenizable, strip punctuation."""
    s = (s or "").lower()
    s = (
        s.replace("c++", " cpp ")
        .replace("c#", " csharp ")
        .replace("f#", " fsharp ")
        .replace(".net", " dotnet ")
        .replace("objective-c", " objectivec ")
    )
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"\bc plus plus\b", " cpp ", s)
    s = re.sub(r"\bc sharp\b", " csharp ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_generic_shelf(name: str) -> bool:
    return bool(_GENERIC_RE.search(name))


def parse_shelves(shelves_json: str | None) -> list[tuple[str, int]]:
    """shelves_json is a JSON list of [name, count] pairs sorted by count desc."""
    if not shelves_json:
        return []
    try:
        return [(n, int(c)) for n, c in json.loads(shelves_json)]
    except (ValueError, TypeError):
        return []


def build_doc(title: str, description: str, shelves: list[tuple[str, int]]) -> str:
    """Document text for TF-IDF: title (x3) + count-weighted informative shelves + description."""
    t = norm_text(title)
    parts = [t, t, t]
    for name, cnt in shelves[:30]:
        if cnt < 2 or is_generic_shelf(name):
            continue
        reps = min(4, 1 + int(math.log2(1 + cnt)))
        parts.extend([norm_text(name)] * reps)
    parts.append(norm_text((description or "")[:1500]))
    return " ".join(parts)
