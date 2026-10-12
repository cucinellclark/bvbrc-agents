"""Remove links from an answer that no tool returned this turn.

Models invent BV-BRC URLs despite the prompt rule (response_format_skill
rule 3): ``docs.bv-brc.org/website/personnel.html`` (the domain does not
resolve), ``bv-brc.org/contact`` (404), ``/view/SimilarGenomeList/``,
``/job/…`` pages that do not exist.  A link is kept when its URL appeared in
a tool result this turn, or matches a page pattern ``url_utils`` builds
(viewer, workspace, service app, site root).  A DOI link is kept when the DOI
appeared in a tool result.  Everything else is unwrapped to its text (or
``(link removed)`` when there is no text) and logged as ANSWER_LINK_REMOVED.

BV-BRC viewer and download URLs carry raw RQL parentheses
(``?eq(genus,Salmonella)``), so URLs are matched with balanced parentheses
and markdown links are scanned rather than matched with a regex.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from shared.tools.url_utils import COLLECTION_VIEWER_MAP, ENTITY_VIEWER_MAP

logger = logging.getLogger(__name__)

MAX_KNOWN_LINKS = 500
REMOVED = "(link removed)"

# Greedy on purpose (parentheses included); _trim_url balances them.
_URL_RE = re.compile(r"https?://[^\s<>\"'`\]]+")
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s<>\"'`\]]+")
_TRAILING = ".,;:!?'\"*_"

_VIEWERS = (
    set(COLLECTION_VIEWER_MAP.values())
    | set(ENTITY_VIEWER_MAP.values())
    | {"GenomeList", "Antibiotic"}
)
_BUILT_PATTERNS = [
    re.compile(r"^https://www\.bv-brc\.org/?$"),
    re.compile(r"^https://www\.bv-brc\.org/view/(" + "|".join(sorted(_VIEWERS)) + r")(/|\?|#|$)"),
    re.compile(r"^https://www\.bv-brc\.org/workspace/"),
    re.compile(r"^https://www\.bv-brc\.org/app/[A-Za-z0-9_]+/?$"),
]
_DOI_URL_RE = re.compile(r"^https?://(dx\.)?doi\.org/(10\..+)$", re.IGNORECASE)


def _trim_url(url: str) -> str:
    """Drop trailing punctuation and closing parens that are not the URL's own."""
    while url:
        if url[-1] in _TRAILING:
            url = url[:-1]
        elif url[-1] == ")" and url.count(")") > url.count("("):
            url = url[:-1]
        else:
            break
    return url


def _norm(url: str) -> str:
    return _trim_url(url.strip()).rstrip("/")


def _walk_strings(value: Any, out: list[str], budget: list[int]) -> None:
    if budget[0] <= 0:
        return
    if isinstance(value, str):
        budget[0] -= 1
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _walk_strings(v, out, budget)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _walk_strings(v, out, budget)


def collect_known_links(tool_results: Iterable[Any]) -> list[str]:
    """URLs (and ``doi:`` entries) that appear anywhere in this turn's tool results."""
    strings: list[str] = []
    budget = [20000]
    for result in tool_results:
        _walk_strings(result, strings, budget)
    known: dict[str, None] = {}
    for s in strings:
        # A field whose whole value is a URL (viewer_url, download_tsv_url,
        # ...) is recorded as-is: url_utils leaves raw spaces in RQL values
        # (eq(species,Escherichia coli)), which the pattern below would cut.
        if s.startswith(("http://", "https://")) and "\n" not in s and len(s) < 4000:
            known.setdefault(_norm(s), None)
        if "http" in s:
            for m in _URL_RE.findall(s):
                known.setdefault(_norm(m), None)
        if "10." in s:
            for m in _DOI_RE.findall(s):
                known.setdefault("doi:" + _norm(m).lower(), None)
        if len(known) >= MAX_KNOWN_LINKS:
            break
    return list(known)[:MAX_KNOWN_LINKS]


def _allowed(url: str, known: set[str]) -> bool:
    url = _norm(url)
    if url in known:
        return True
    doi = _DOI_URL_RE.match(url)
    if doi:
        return "doi:" + doi.group(2).lower() in known
    return any(p.match(url) for p in _BUILT_PATTERNS)


def _scan_md_link(text: str, start: int) -> tuple[int, str, str] | None:
    """Parse ``[label](url)`` / ``[label](<url>)`` at *start* (the ``[``).

    Returns ``(end_index, label, url)`` or None.  Parentheses inside a bare
    URL are balanced, so RQL links survive.
    """
    close = text.find("]", start + 1)
    if close < 0 or close + 1 >= len(text) or text[close + 1] != "(":
        return None
    label = text[start + 1:close]
    i = close + 2
    if i < len(text) and text[i] == "<":
        gt = text.find(">", i + 1)
        if gt < 0 or gt + 1 >= len(text) or text[gt + 1] != ")":
            return None
        return gt + 2, label, text[i + 1:gt]
    depth = 0
    j = i
    while j < len(text):
        c = text[j]
        if c.isspace():
            return None
        if c == "(":
            depth += 1
        elif c == ")":
            if depth == 0:
                return j + 1, label, text[i:j]
            depth -= 1
        j += 1
    return None


def filter_answer(answer: str, known_links: Iterable[str]) -> tuple[str, list[str]]:
    """Return ``(answer_without_unknown_links, removed_urls)``."""
    if not answer or "http" not in answer:
        return answer, []
    known = set(known_links or ())
    removed: list[str] = []
    out: list[str] = []
    i = 0
    n = len(answer)
    while i < n:
        c = answer[i]
        # Markdown link (or image): [label](url)
        if c == "[":
            parsed = _scan_md_link(answer, i)
            if parsed:
                end, label, url = parsed
                bang = bool(out) and out[-1] == "!"
                if not url.lower().startswith(("http://", "https://")) or _allowed(url, known):
                    out.append(answer[i:end])
                else:
                    removed.append(url)
                    if bang:
                        out.pop()
                    label = label.strip()
                    out.append(REMOVED if not label or _URL_RE.match(label) else label)
                i = end
                continue
        # Bare URL
        if c == "h" and answer.startswith(("http://", "https://"), i):
            m = _URL_RE.match(answer, i)
            if m:
                raw = m.group(0)
                url = _trim_url(raw)
                if _allowed(url, known):
                    out.append(url)
                else:
                    removed.append(url)
                    out.append(REMOVED)
                i += len(url)
                continue
        out.append(c)
        i += 1
    if removed:
        logger.warning("ANSWER_LINK_REMOVED n=%d urls=%s", len(removed), removed[:10])
    return "".join(out), removed
