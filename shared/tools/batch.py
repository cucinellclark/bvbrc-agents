"""``prepare_batch``: one workflow over many samples, prepared with no LLM per job.

D4 Phase 4 (``current_plans/multi_step_requests.md`` §6).  Before this the
model built one ``submit_gowe_job`` call per sample by hand -- one LLM call
(~10 s) each, with read paths copied out of a listing.  Here the model
populates the workflow's inputs ONCE (``template_inputs``, from
``get_workflow_inputs``) and this tool fills in each sample's reads:

- the sample goes into the input the workflow declares for its kind
  (``read1``/``read2`` record → paired, ``read`` record → single,
  ``srr_accession`` record or an ``srr_ids``/``srr_id`` string → SRA), as a
  list or a single record as the input's type says;
- the record gets only fields the workflow declares (``sample_id`` where
  declared; ``platform`` and the like copied from the template's record);
- every other library input is removed from the template, so one sample's
  reads never leak into another's job;
- ``output_file`` is ``<template output_file>_<sample_id>``; all jobs share
  the template's output folder;
- ``one_job=true`` puts every sample into one job instead (RNA-Seq,
  Taxonomic Classification of several samples together).

Each job is checked (identifiers, GoWe dry run) and returned as
``prepared_actions``: nothing is submitted.  The user answers one Yes / No /
Other question and Yes replays the jobs verbatim, like ``rerun_jobs``.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
from typing import Any, Dict, List, Optional

from shared.tools import gowe
from shared.tools.gowe import _check_identifiers, _workflow_status, place_output_path
from shared.tools.roles import check_roles, with_roles

logger = logging.getLogger(__name__)

# Same limit as a prepared set and as replay (MAX_PREPARED_JOBS, gateway
# MAX_REPLAY_ACTIONS): a bigger set could be prepared but never confirmed.
MAX_BATCH_JOBS = 10

_SRA_STRING_INPUTS = ("srr_ids", "srr_id")
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _is_list(input_type: Any) -> bool:
    return "[]" in str(input_type or "")


def _fields(spec: Dict[str, Any]) -> List[str]:
    return [f.get("name") if isinstance(f, dict) else str(f) for f in spec.get("fields") or []]


def library_inputs(specs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{"paired"|"single"|"sra_record"|"sra_string": {"id", "list", "fields"}}``."""
    found: Dict[str, Dict[str, Any]] = {}
    for spec in specs or []:
        if not isinstance(spec, dict) or not spec.get("id"):
            continue
        fields = _fields(spec)
        entry = {"id": spec["id"], "list": _is_list(spec.get("type")), "fields": fields}
        if "read1" in fields and "read2" in fields:
            found.setdefault("paired", entry)
        elif "read" in fields:
            found.setdefault("single", entry)
        elif "srr_accession" in fields:
            found.setdefault("sra_record", entry)
        elif spec["id"] in _SRA_STRING_INPUTS:
            found.setdefault("sra_string", entry)
    return found


def _kind(sample: Dict[str, Any]) -> Optional[str]:
    if sample.get("read1") and sample.get("read2"):
        return "paired"
    if sample.get("read"):
        return "single"
    if sample.get("srr_accession"):
        return "sra"
    return None


def _label(sample: Dict[str, Any]) -> str:
    def name(p: Any) -> str:
        return str(p).rstrip("/").rsplit("/", 1)[-1]
    kind = _kind(sample)
    if kind == "paired":
        return f"{name(sample['read1'])} + {name(sample['read2'])}"
    if kind == "single":
        return name(sample["read"])
    return str(sample.get("srr_accession"))


def _record(sample: Dict[str, Any], kind: str, lib: Dict[str, Any], template_rec: Dict[str, Any]) -> Dict[str, Any]:
    fields = lib["fields"]
    rec = {k: v for k, v in template_rec.items() if k in fields}
    values = {
        "paired": {"read1": sample.get("read1"), "read2": sample.get("read2")},
        "single": {"read": sample.get("read")},
        "sra": {"srr_accession": sample.get("srr_accession")},
    }[kind]
    rec.update(values)
    if "sample_id" in fields:
        rec["sample_id"] = sample["sample_id"]
    return rec


def _template_record(template: Dict[str, Any], input_id: str) -> Dict[str, Any]:
    value = template.get(input_id)
    if isinstance(value, list):
        value = value[0] if value else None
    rec = dict(value) if isinstance(value, dict) else {}
    for key in ("read", "read1", "read2", "srr_accession", "sample_id"):
        rec.pop(key, None)
    return rec


def _place(sample: Dict[str, Any], kind: str, libs: Dict[str, Dict[str, Any]],
           template: Dict[str, Any], inputs: Dict[str, Any]) -> Optional[str]:
    """Add *sample* to *inputs*; return why it cannot be, or None."""
    if kind == "sra":
        lib = libs.get("sra_record")
        if lib:
            rec = _record(sample, "sra", lib, _template_record(template, lib["id"]))
        elif libs.get("sra_string"):
            lib = libs["sra_string"]
            rec = sample["srr_accession"]
        else:
            return "the workflow takes no SRA accession"
    else:
        lib = libs.get(kind)
        if not lib:
            return f"the workflow takes no {'paired-end' if kind == 'paired' else 'single-end'} reads"
        rec = _record(sample, kind, lib, _template_record(template, lib["id"]))
    if lib["list"]:
        inputs.setdefault(lib["id"], []).append(rec)
    elif lib["id"] in inputs:
        return f"{lib['id']} takes one library per job, so these samples cannot share a job"
    else:
        inputs[lib["id"]] = rec
    return None


def _clean_samples(samples: Any) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Valid samples with a unique ``sample_id``, and the rejected ones."""
    good: List[Dict[str, Any]] = []
    bad: List[Dict[str, Any]] = []
    used: set[str] = set()
    seen: set[tuple] = set()
    for raw in samples if isinstance(samples, list) else []:
        if not isinstance(raw, dict) or not _kind(raw):
            bad.append({"sample": raw, "reason": "needs read1+read2, read, or srr_accession"})
            continue
        sample = {k: raw[k] for k in ("read1", "read2", "read", "srr_accession") if raw.get(k)}
        key = tuple(sorted(sample.items()))
        if key in seen:
            continue
        seen.add(key)
        base = _UNSAFE.sub("_", str(raw.get("sample_id") or _label(sample).split(" ")[0])).strip("._-") or "sample"
        base = re.sub(r"\.(fastq|fq)(\.gz)?$", "", base, flags=re.IGNORECASE)
        sid, n = base, 2
        while sid in used:
            sid, n = f"{base}_{n}", n + 1
        used.add(sid)
        sample["sample_id"] = sid
        good.append(sample)
    return good, bad


async def prepare_batch(
    workflow_id: str = "",
    template_inputs: Optional[Dict[str, Any]] = None,
    samples: Optional[List[Dict[str, Any]]] = None,
    one_job: bool = False,
    config: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Prepare one job per sample (or one job for all, ``one_job``).

    Returns ``{"prepared_actions": [...], "skipped": [...], "note"}`` or an
    error dict (``TOO_MANY`` above 10 jobs: nothing is prepared then).
    """
    if not workflow_id:
        return {"error": "workflow_id is required (the workflow NAME from list_gowe_workflows).",
                "errorType": "INVALID_PARAMETERS"}
    template = template_inputs if isinstance(template_inputs, dict) else {}
    good, bad = _clean_samples(samples)
    if not good:
        return {"error": "No usable samples. Pass the entries of list_samples (paired / single / sra).",
                "errorType": "INVALID_PARAMETERS", "rejected": bad}

    resolved, ref_error = await gowe._resolve_workflow_ref(workflow_id, config)
    if ref_error:
        return {"error": ref_error, "errorType": "INVALID_PARAMETERS"}
    name = gowe._workflow_names.get(resolved) or gowe._workflow_name(workflow_id, resolved)
    status, reason = _workflow_status(name)
    if status == "experimental":
        return {"error": f"{name} is marked experimental: {reason}. Warn the user and ask before preparing it.",
                "errorType": "EXPERIMENTAL", "workflow": name}
    try:
        specs = await gowe._get_client(config).get_workflow_inputs(resolved, auth_token=gowe._get_auth(config))
    except Exception as e:
        return {"error": f"Could not read the inputs of {name}: {e}", "errorType": "API_ERROR"}
    # Same check submit_gowe_job makes; asked once for the whole batch.
    organism = gowe._check_organism(name, template)
    if organism:
        return {"error": organism[1], "needs_user_input": True, "errorType": "MISSING_INPUT", "workflow": name}
    libs = library_inputs(specs or [])
    if not libs:
        return {"error": f"{name} takes no read or SRA inputs, so it cannot run per sample.",
                "errorType": "INVALID_PARAMETERS", "workflow": name}
    declared = {s["id"] for s in specs or [] if isinstance(s, dict) and s.get("id")}

    lib_ids = {lib["id"] for lib in libs.values()} | set(_SRA_STRING_INPUTS) | {
        "paired_end_libs", "single_end_libs", "srr_libs", "paired_end_lib", "single_end_lib"}
    # Only inputs the workflow declares; others are reported, never sent.
    base_inputs = {k: copy.deepcopy(v) for k, v in template.items()
                   if k in declared and k not in lib_ids and k not in ("output_path", "output_file")}
    dropped = sorted(k for k in template if k not in declared and k not in lib_ids)
    out_base = _UNSAFE.sub("_", str(template.get("output_file") or name)).strip("._-") or "batch"
    session_id = getattr(config, "session_id", None)
    workspace_path = getattr(config, "workspace_path", None)
    if session_id and workspace_path:
        out_path, _ = place_output_path(template.get("output_path") or f"{out_base}_batch",
                                        f"{workspace_path}/chats/{session_id}", session_id, name)
    else:
        out_path = template.get("output_path") or f"{out_base}_batch"

    skipped: List[Dict[str, Any]] = [{"sample": b["sample"], "reason": b["reason"]} for b in bad]
    jobs: List[Dict[str, Any]] = []  # {"inputs", "samples": [...]}
    if one_job:
        inputs = copy.deepcopy(base_inputs)
        members = []
        for s in good:
            why = _place(s, _kind(s), libs, template, inputs)
            if why:
                skipped.append({"sample": s["sample_id"], "reason": why})
            else:
                members.append(s)
        if members:
            inputs.update({"output_path": out_path, "output_file": out_base})
            jobs.append({"inputs": inputs, "samples": members})
    else:
        for s in good:
            inputs = copy.deepcopy(base_inputs)
            why = _place(s, _kind(s), libs, template, inputs)
            if why:
                skipped.append({"sample": s["sample_id"], "reason": why})
                continue
            inputs.update({"output_path": out_path, "output_file": f"{out_base}_{s['sample_id']}"})
            jobs.append({"inputs": inputs, "samples": [s]})

    for job in list(jobs):
        id_error = _check_identifiers(job["inputs"]) or check_roles(name, job["inputs"])[0]
        if id_error:
            jobs.remove(job)
            skipped += [{"sample": s["sample_id"], "reason": id_error} for s in job["samples"]]

    if len(jobs) > MAX_BATCH_JOBS:
        return {
            "error": (
                f"{len(jobs)} jobs; one message can prepare at most {MAX_BATCH_JOBS}. Nothing "
                f"was prepared. Ask the user which samples to run first (or to split the "
                f"folder), or use one_job=true if {name} takes several samples in one job."
            ),
            "errorType": "TOO_MANY",
            "job_count": len(jobs),
            "samples": [s["sample_id"] for j in jobs for s in j["samples"]],
            "skipped": skipped,
        }

    refusals = await asyncio.gather(*(gowe.dry_run_refusal(resolved, j["inputs"], config) for j in jobs))
    prepared: List[Dict[str, Any]] = []
    for job, refusal in zip(jobs, refusals):
        ids = [s["sample_id"] for s in job["samples"]]
        if refusal:
            skipped += [{"sample": i, "reason": f"GoWe refuses the job: {refusal}"} for i in ids]
            continue
        label = (f"{len(ids)} samples ({', '.join(ids)})" if one_job
                 else f"{ids[0]} ({_label(job['samples'][0])})")
        prepared.append({
            "tool": "submit_gowe_job",
            "summary": with_roles(f"{name} ← {label}", check_roles(name, job["inputs"])[1]),
            "arguments": {"workflow_id": resolved, "inputs": job["inputs"]},
            "origin": "prepare_batch",
            "samples": ids,
        })

    notes = []
    if prepared:
        notes.append(
            f"Prepared {len(prepared)} {name} job{'s' if len(prepared) != 1 else ''}"
            f"{' with all samples together' if one_job else ', one per sample'}; nothing has been "
            f"submitted. The user confirms the set with one question (Yes / No / Other). "
            f"Results go to {out_path}."
        )
    else:
        notes.append("No job could be prepared; nothing has been submitted.")
    if skipped:
        notes.append("Tell the user each skipped sample and why.")
    if dropped:
        notes.append(f"{name} does not declare these template inputs, so they were left out: {', '.join(dropped)}.")
    logger.info("prepare_batch: %s, %d prepared, %d skipped (one_job=%s)", name, len(prepared), len(skipped), one_job)
    result = {"prepared_actions": prepared, "skipped": skipped, "output_path": out_path, "note": " ".join(notes)}
    if dropped:
        result["dropped_inputs"] = dropped
    return result
