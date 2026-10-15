"""``list_samples``: the samples in a reads folder or an SRA list, with no LLM.

D4 Phase 4 (``current_plans/multi_step_requests.md`` §6).  The model used to
pair R1/R2 files by eye from a ``workspace_browse`` listing, and invented
paths doing it (F16).  Pairing is a naming rule, so code does it:

- read files are ``.fastq`` / ``.fq`` (optionally ``.gz``) in ONE folder
  level (subfolders are not searched);
- a mate token is ``R1``/``R2`` or ``1``/``2`` between separators
  (``S1_L001_R1_001.fastq.gz``, ``A_1.fq``, ``B.R2.fastq``); two files whose
  names differ only in that token are a pair;
- a file with a mate token and no partner is reported (``unpaired``) and
  never treated as a single-end sample;
- a file with no mate token is a single-end sample.

The result feeds ``prepare_batch``: ``paired[]`` / ``single[]`` / ``sra[]``
entries are its ``samples``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from shared.tools.workspace import workspace_browse

logger = logging.getLogger(__name__)

MAX_FOLDER_ITEMS = 500

_READ_EXT = re.compile(r"\.(fastq|fq)(\.gz)?$", re.IGNORECASE)
# A mate token: R1/R2 or 1/2, after a separator and before a separator or
# the end of the stem.
_MATE = re.compile(r"(?<=[._-])(R?)([12])(?=[._-]|$)", re.IGNORECASE)
_SRR = re.compile(r"^[SED]RR\d{5,}$")
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _stem(name: str) -> str:
    return _READ_EXT.sub("", name)


def _mate_keys(stem: str) -> List[tuple[str, int, str]]:
    """``(pair_key, mate, sample_prefix)`` for every mate token in *stem*.

    The key replaces the token with ``#`` so both mates share it.  The
    prefix is the text before the token, without its separator.
    """
    found = list(_MATE.finditer(stem))
    # An R1/R2 token outranks bare digits: in sample_1_R1 the "1" is part of
    # the sample name, and sample_2_R1 must not become its mate.
    if any(m.group(1) for m in found):
        found = [m for m in found if m.group(1)]
    keys = []
    for m in found:
        key = f"{stem[:m.start()]}{m.group(1).upper()}#{stem[m.end():]}"
        keys.append((key, int(m.group(2)), stem[: m.start() - 1]))
    # The mate written twice (Trim Galore: x_1_val_1 / x_2_val_2): replace
    # every token carrying the same digit at once.
    for mate in (1, 2):
        same = [m for m in found if int(m.group(2)) == mate]
        if len(same) > 1:
            key, last = "", 0
            for m in same:
                key += f"{stem[last:m.start()]}{m.group(1).upper()}#"
                last = m.end()
            keys.insert(0, (key + stem[last:], mate, stem[: same[0].start() - 1]))
    return keys


def _sample_id(raw: str, used: set[str]) -> str:
    base = _UNSAFE.sub("_", raw).strip("._-") or "sample"
    sid, n = base, 2
    while sid in used:
        sid = f"{base}_{n}"
        n += 1
    used.add(sid)
    return sid


def pair_read_files(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pair read files (``{name, path}``) into samples.  Pure; see the module doc."""
    reads = sorted(
        (f for f in files if isinstance(f, dict) and _READ_EXT.search(f.get("name") or "")),
        key=lambda f: f["name"],
    )
    by_key: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for f in reads:
        for key, mate, _ in _mate_keys(_stem(f["name"])):
            by_key.setdefault(key, {}).setdefault(mate, f)

    used_files: set[str] = set()
    used_ids: set[str] = set()
    paired: List[Dict[str, Any]] = []
    unpaired: List[Dict[str, Any]] = []
    single: List[Dict[str, Any]] = []
    for f in reads:
        if f["path"] in used_files:
            continue
        stem = _stem(f["name"])
        keys = _mate_keys(stem)
        if not keys:
            used_files.add(f["path"])
            single.append({"sample_id": _sample_id(stem, used_ids), "read": f["path"]})
            continue
        # A doubled-token key first, then the last token that has a partner
        # (S1_L001_R1_001: the R1, not a "1" earlier in the name).
        match = None
        doubled = [k for k in keys if k[0].count("#") > 1]
        for key, mate, prefix in doubled + [k for k in reversed(keys) if k not in doubled]:
            mates = by_key.get(key, {})
            if mate == 1 and 2 in mates and mates[2]["path"] not in used_files:
                match = (f, mates[2], prefix)
                break
            if mate == 2 and 1 in mates and mates[1]["path"] not in used_files:
                match = (mates[1], f, prefix)
                break
        if match:
            r1, r2, prefix = match
            used_files.update({r1["path"], r2["path"]})
            paired.append({
                "sample_id": _sample_id(prefix, used_ids),
                "read1": r1["path"],
                "read2": r2["path"],
            })
        else:
            used_files.add(f["path"])
            unpaired.append({"read": f["path"], "mate": keys[-1][1]})
    return {"paired": paired, "single": single, "unpaired": unpaired}


def _sra_samples(sra_ids: List[Any]) -> tuple[List[Dict[str, str]], List[str]]:
    sra, invalid, seen = [], [], set()
    for raw in sra_ids or []:
        acc = str(raw).strip().upper()
        if not acc or acc in seen:
            continue
        seen.add(acc)
        if _SRR.match(acc):
            sra.append({"sample_id": acc, "srr_accession": acc})
        else:
            invalid.append(str(raw))
    return sra, invalid


async def list_samples(
    path: Optional[str] = None,
    sra_ids: Optional[List[str]] = None,
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Samples in a reads folder (one level) and/or a list of SRA run accessions.

    Returns ``{"paired": [{sample_id, read1, read2}], "single": [{sample_id,
    read}], "sra": [{sample_id, srr_accession}], "unpaired": [{read, mate}],
    "invalid_sra": [...], "sample_count", "note"}`` or an error dict.
    """
    if not path and not sra_ids:
        return {
            "error": "Give a folder path with read files, or sra_ids.",
            "errorType": "INVALID_PARAMETERS",
        }
    result: Dict[str, Any] = {"paired": [], "single": [], "sra": [], "unpaired": []}
    notes: List[str] = []

    if path:
        listing = await workspace_browse(
            path=path, num_results=MAX_FOLDER_ITEMS + 1, config=config, headers=headers,
        )
        if not isinstance(listing, dict) or listing.get("error"):
            return listing if isinstance(listing, dict) else {"error": "Workspace listing failed."}
        inner = listing.get("result") or {}
        items = [i for i in inner.get("items") or [] if isinstance(i, dict)]
        folder = inner.get("path") or path
        result["folder"] = folder
        if len(items) > MAX_FOLDER_ITEMS:
            notes.append(
                f"The folder has more than {MAX_FOLDER_ITEMS} items; only the first "
                f"{MAX_FOLDER_ITEMS} were read."
            )
            items = items[:MAX_FOLDER_ITEMS]
        files = [i for i in items if i.get("type") != "folder" and i.get("name") and i.get("path")]
        result.update(pair_read_files(files))
        subfolders = sum(1 for i in items if i.get("type") == "folder")
        if subfolders and not (result["paired"] or result["single"]):
            notes.append(
                f"No read files directly in {folder}; it has {subfolders} subfolder"
                f"{'s' if subfolders != 1 else ''}, which are not searched. Ask which "
                f"subfolder holds the reads."
            )

    if sra_ids:
        result["sra"], invalid = _sra_samples(sra_ids)
        if invalid:
            result["invalid_sra"] = invalid
            notes.append(
                f"Not SRA run accessions (SRR/ERR/DRR + digits), left out: {', '.join(invalid)}."
            )

    count = len(result["paired"]) + len(result["single"]) + len(result["sra"])
    result["sample_count"] = count
    if result["unpaired"]:
        notes.append(
            f"{len(result['unpaired'])} read file(s) look like one mate of a pair but have no "
            f"partner; they are NOT samples. Tell the user."
        )
    if result["paired"] and result["single"]:
        notes.append(
            "The folder mixes paired and single-end files; check with the user that the "
            "single-end files are real samples."
        )
    if count:
        notes.insert(0, f"{count} sample{'s' if count != 1 else ''}, paired by file name.")
    elif not notes:
        notes.append("No read files (.fastq/.fq, optionally .gz) and no SRA accessions found.")
    result["note"] = " ".join(notes)
    logger.info(
        "list_samples: %d paired, %d single, %d sra, %d unpaired (path=%s)",
        len(result["paired"]), len(result["single"]), len(result["sra"]),
        len(result["unpaired"]), path,
    )
    return result
