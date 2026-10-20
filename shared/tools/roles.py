"""Role inputs: values whose meaning is "which item plays which part".

Some workflows say which genome is the reference, or which samples form a
condition, through an input the model sets and the user never sees in plain
words.  The question then shows a bare number and the user approves the wrong
job: cmao's two Proteome Comparison runs (Sep 17 and Oct 16) asked for an AMR
feature group as the reference but sent ``reference_genome_index: 1``, which
is the first of six genome ids (the group was 7).  An RNA-Seq run (Oct 8)
sent ``contrasts: "Cd30 vs Cd0"``, which the app read as eleven one-letter
contrasts.

``check_roles(workflow_name, inputs)`` returns ``(error, lines)``: ``error``
when the inputs cannot be what the user meant (the call is refused, nothing
is submitted), else plain-language ``lines`` for the confirmation question.
It never changes the inputs, so a replay runs what was shown.  It runs in
``submit_gowe_job``, at the plan-mode gate, and in ``prepare_batch`` /
``rerun_jobs``.  Add a workflow by adding a function to ``_CHECKS``.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

RoleResult = Tuple[Optional[str], List[str]]

_GENOME_ID_RE = re.compile(r"^\d+\.\d+$")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name or "").lower())


def _as_list(value: Any) -> list:
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def _basename(path: Any) -> str:
    return str(path).rstrip("/").rsplit("/", 1)[-1]


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


# ---------------------------------------------------------------------------
# Proteome Comparison (app GenomeComparison)
# ---------------------------------------------------------------------------
# reference_genome_index is 1-based and counts genome_ids, then user_genomes,
# then user_feature_groups (website SeqComparison.js, refIndex).

def _proteome_comparison(inputs: Dict[str, Any]) -> RoleResult:
    items = (
        [f"genome {g}" for g in _as_list(inputs.get("genome_ids"))]
        + [f"FASTA file `{_basename(p)}`" for p in _as_list(inputs.get("user_genomes"))]
        + [f"feature group `{_basename(p)}`" for p in _as_list(inputs.get("user_feature_groups"))]
    )
    if not items:
        return None, []
    raw = inputs.get("reference_genome_index", 1)
    index = _as_int(raw)
    if index is None or not 1 <= index <= len(items):
        listing = "; ".join(f"{i} = {item}" for i, item in enumerate(items, 1))
        return (
            f"reference_genome_index {raw!r} does not point at an input. It is 1-based and "
            f"counts genome_ids first, then user_genomes, then user_feature_groups: {listing}. "
            f"Set it to the position of the reference the user named."
        ), []
    return None, [f"reference: {items[index - 1]} (reference_genome_index {index})"]


# ---------------------------------------------------------------------------
# RNA-Seq
# ---------------------------------------------------------------------------
# The website sends labels: each library's condition is one of
# experimental_conditions, and contrasts are [[label, label], ...]; the app
# rejects anything else ("Contrast N must be a list of 2 condition names").
# The p3 CLI sends 1-based ints instead; labels_for_numbered_conditions turns
# a recorded job's ints into labels before a rerun.

_RNASEQ_LIBS = ("paired_end_libs", "single_end_libs", "srr_libs")
_SHAPE = (
    'Use labels, as the BV-BRC website does: experimental_conditions: ["<condition A>", '
    '"<condition B>"], each library\'s condition: "<condition A>" (one of those labels), '
    'contrasts: [["<condition A>", "<condition B>"]].'
)


def _library_name(lib: Dict[str, Any]) -> str:
    for key in ("sample_id", "srr_accession", "read1", "read"):
        if lib.get(key):
            return _basename(lib[key])
    return "a library"


def _rnaseq(inputs: Dict[str, Any]) -> RoleResult:
    conditions = [str(c) for c in _as_list(inputs.get("experimental_conditions"))]
    contrasts = inputs.get("contrasts")
    libs: List[Dict[str, Any]] = []
    for key in _RNASEQ_LIBS:
        libs += [lib for lib in _as_list(inputs.get(key)) if isinstance(lib, dict)]

    groups: Dict[str, List[str]] = {c: [] for c in conditions}
    unassigned: List[str] = []
    for lib in libs:
        name = _library_name(lib)
        cond = lib.get("condition")
        if cond is None or cond == "":
            unassigned.append(name)
            continue
        if _as_int(cond) is not None and str(cond).strip() not in conditions:
            return f"{name} has condition {cond!r}, a number. {_SHAPE}", []
        if str(cond) not in groups:
            return (
                f"{name} has condition {cond!r}, which is not in experimental_conditions "
                f"{conditions}. {_SHAPE}"
            ), []
        groups[str(cond)].append(name)

    if not conditions and not any(lib.get("condition") not in (None, "") for lib in libs):
        if contrasts not in (None, "", []):
            return f"contrasts are set but there are no experimental_conditions. {_SHAPE}", []
        return None, []
    if unassigned:
        return f"{', '.join(unassigned)} has no condition; give every library one. {_SHAPE}", []
    empty = [c for c, members in groups.items() if not members]
    if empty:
        return f"No library is in condition(s) {empty}; drop them or assign libraries. {_SHAPE}", []
    if inputs.get("recipe") == "HTSeq-DESeq":
        thin = [c for c, members in groups.items() if len(members) < 2]
        if thin:
            return (
                f"HTSeq-DESeq needs at least 2 replicates per condition; {thin} "
                f"{'has' if len(thin) == 1 else 'have'} fewer. Ask the user."
            ), []

    pairs: List[Tuple[str, str]] = []
    if contrasts not in (None, "", []):
        if not isinstance(contrasts, list):
            return f"contrasts {contrasts!r} must be a list of pairs. {_SHAPE}", []
        for pair in contrasts:
            if not (isinstance(pair, list) and len(pair) == 2):
                return f"contrast {pair!r} must be a pair [condition, condition]. {_SHAPE}", []
            names = []
            for part in pair:
                if _as_int(part) is not None and str(part).strip() not in conditions:
                    return f"contrast {pair!r} uses a number. {_SHAPE}", []
                if str(part) not in groups:
                    return f"contrast {pair!r} names {part!r}, which is not a condition. {_SHAPE}", []
                names.append(str(part))
            if names[0] == names[1]:
                return f"contrast {pair!r} compares a condition with itself.", []
            pairs.append((names[0], names[1]))

    lines = ["conditions: " + "; ".join(f"{c} = {', '.join(m)}" for c, m in groups.items())]
    if pairs:
        lines.append("contrasts: " + "; ".join(f"{a} vs {b}" for a, b in pairs))
    return None, lines


# ---------------------------------------------------------------------------
# MSA SNP
# ---------------------------------------------------------------------------
# ref_type "first" is the first sequence of the merged input, which only the
# app knows; a named reference goes in ref_string with ref_type feature_id /
# genome_id / string.

def _msa(inputs: Dict[str, Any]) -> RoleResult:
    ref_type = str(inputs.get("ref_type") or "none")
    ref = str(inputs.get("ref_string") or "").strip()
    if ref_type == "none":
        return None, []
    if ref_type == "first":
        return None, ["reference: the first input sequence (BV-BRC decides the order)"]
    if ref_type in ("feature_id", "genome_id", "string") and not ref:
        return f"ref_type {ref_type!r} needs ref_string (the reference {ref_type.replace('_', ' ')}).", []
    if ref_type == "genome_id" and not _GENOME_ID_RE.match(ref):
        return f"ref_string {ref!r} is not a BV-BRC genome id like 83332.12.", []
    if ref_type == "string":
        return None, [f"reference: the pasted sequence ({len(ref)} characters)"]
    if ref_type in ("feature_id", "genome_id"):
        return None, [f"reference: {ref_type.replace('_', ' ')} {ref}"]
    return f"ref_type {ref_type!r} is not one of none, first, string, feature_id, genome_id.", []


_CHECKS: Dict[str, Callable[[Dict[str, Any]], RoleResult]] = {
    "proteomecomparison": _proteome_comparison,
    "genomecomparison": _proteome_comparison,
    "rnaseq": _rnaseq,
    "msasnp": _msa,
}


def check_roles(workflow_name: str, inputs: Any) -> RoleResult:
    """``(error, lines)`` for *workflow_name*'s role inputs; ``(None, [])`` if it has none."""
    check = _CHECKS.get(_norm(workflow_name))
    if check is None or not isinstance(inputs, dict):
        return None, []
    return check(inputs)


def labels_for_numbered_conditions(workflow_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """A recorded RNA-Seq job's 1-based condition numbers (p3 CLI) as labels.

    For ``rerun_jobs``: the replay goes through check_roles, which takes
    labels only.  Numbers that do not point at a condition are left as they
    are, so the check reports them.  Other workflows come back unchanged.
    """
    if _norm(workflow_name) != "rnaseq":
        return inputs
    conditions = [str(c) for c in _as_list(inputs.get("experimental_conditions"))]

    def label(value: Any) -> Any:
        number = _as_int(value)
        if number is None or str(value).strip() in conditions or not 1 <= number <= len(conditions):
            return value
        return conditions[number - 1]

    out = dict(inputs)
    for key in _RNASEQ_LIBS:
        if isinstance(inputs.get(key), list):
            out[key] = [
                {**lib, "condition": label(lib["condition"])}
                if isinstance(lib, dict) and "condition" in lib else lib
                for lib in inputs[key]
            ]
    if isinstance(inputs.get("contrasts"), list):
        out["contrasts"] = [
            [label(p) for p in pair] if isinstance(pair, list) else pair
            for pair in inputs["contrasts"]
        ]
    return out


def with_roles(summary: str, lines: List[str]) -> str:
    """The question's one-line summary with the role lines appended."""
    return f"{summary} ({'; '.join(lines)})" if lines else summary
