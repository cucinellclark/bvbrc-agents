"""
Hybrid service-specific workflow planning functions.

Each function accepts a params dict from the LLM, validates/corrects it
deterministically, applies defaults, builds a single-step workflow manifest,
and persists it to the workflow engine. No LLM calls are made here.

Supported services:
  - GenomeAssembly2  (plan_genome_assembly_fn)
  - GenomeAnnotation (plan_genome_annotation_fn)
  - ComparativeSystems (plan_comparative_systems_fn)
"""

import json
import os
import sys
import time
import uuid
from typing import Dict, List, Any, Optional

from common.workflow_engine_client import WorkflowEngineClient, WorkflowEngineError
from common.json_rpc import JsonRpcCaller
from functions.group_functions import resolve_group_path

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_config_file(filename: str) -> Dict:
    """Load a JSON config file from the config directory."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, 'config', filename)
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _build_manifest(
    workflow_name: str,
    step_name: str,
    app_api_name: str,
    params: Dict[str, Any],
    outputs: Dict[str, str],
    user_id: str,
) -> Dict[str, Any]:
    """Build a single-step workflow manifest."""
    return {
        "workflow_name": workflow_name,
        "version": "1.0",
        "base_context": {
            "base_url": "https://www.bv-brc.org",
            "workspace_output_folder": f"/{user_id}/home"
        },
        "steps": [
            {
                "step_name": step_name,
                "app": app_api_name,
                "params": params,
                "outputs": outputs,
                "depends_on": []
            }
        ],
        "workflow_outputs": list(outputs.values())
    }


def _default_output(user_id: str, app_api_name: str, output_path: Optional[str], output_file: Optional[str]):
    """Resolve output_path and output_file with sensible defaults."""
    if not output_file:
        output_file = f"{app_api_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    if not output_path:
        output_path = f"/{user_id}/home/CopilotWorkflows"
    else:
        # Ensure path is rooted to user workspace
        if not output_path.startswith('/'):
            output_path = f"/{user_id}/home/{output_path}"
    return output_path, output_file


def _output_patterns(app_api_name: str) -> Dict[str, str]:
    """Load output patterns from service_outputs.json for the given app."""
    try:
        all_outputs = _load_config_file('service_outputs.json')
    except Exception:
        return {"job_output_path": "${params.output_path}/.${params.output_file}"}

    patterns = dict(all_outputs.get(app_api_name, {}))
    # Always include job_output_path
    patterns["job_output_path"] = "${params.output_path}/.${params.output_file}"
    return patterns


def _fuzzy_match_enum(value: str, valid_values: set, aliases: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Attempt to match a value against valid enum values, case-insensitively.
    Also checks an optional alias map.

    Returns the canonical value if matched, or None if no match.
    """
    if value in valid_values:
        return value

    # Check aliases first
    if aliases:
        lower_val = value.lower().strip()
        if lower_val in aliases:
            return aliases[lower_val]

    # Case-insensitive match against valid values
    lower_map = {v.lower(): v for v in valid_values}
    lower_val = value.lower().strip()
    if lower_val in lower_map:
        return lower_map[lower_val]

    # Underscore/hyphen normalization
    normalized = lower_val.replace('_', '-')
    if normalized in lower_map:
        return lower_map[normalized]
    normalized = lower_val.replace('-', '_')
    if normalized in lower_map:
        return lower_map[normalized]

    return None


def _coerce_to_list(value: Any) -> List[Any]:
    """Coerce a value to a list if it isn't already."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _coerce_to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Coerce a value to an integer if possible."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return default
    return default


def _coerce_to_bool(value: Any) -> bool:
    """Coerce a value to a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower().strip() in ('true', '1', 'yes')
    if isinstance(value, (int, float)):
        return bool(value)
    return False


async def _persist_to_engine(manifest: Dict[str, Any], auth_token: str) -> Dict[str, Any]:
    """Persist a workflow manifest to the workflow engine. Returns workflow_id and status."""
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'config.json'
        )
        with open(config_path, 'r') as f:
            config = json.load(f)

        engine_config = config.get('workflow_engine', {})
        if not engine_config.get('enabled', False):
            local_id = f"wf_local_{uuid.uuid4().hex[:12]}"
            return {
                "workflow_id": local_id,
                "status": "planned",
                "persisted": False,
                "warning": "Workflow engine is disabled; assigned local ID"
            }

        engine_url = engine_config.get('api_url', 'http://localhost:8000/api/v1')
        engine_timeout = engine_config.get('timeout', 30)
        client = WorkflowEngineClient(base_url=engine_url, timeout=engine_timeout)

        result = await client.plan_workflow(manifest, auth_token)
        workflow_id = (
            result.get('workflow_id')
            or result.get('id')
            or (result.get('workflow') or {}).get('workflow_id')
        )
        if not workflow_id:
            # Fallback to register
            result = await client.register_workflow(manifest, auth_token)
            workflow_id = (
                result.get('workflow_id')
                or result.get('id')
                or (result.get('workflow') or {}).get('workflow_id')
            )

        return {
            "workflow_id": workflow_id or f"wf_local_{uuid.uuid4().hex[:12]}",
            "status": result.get('status', 'planned'),
            "persisted": workflow_id is not None,
        }

    except WorkflowEngineError as e:
        print(f"Workflow engine error during persist: {e}", file=sys.stderr)
        local_id = f"wf_local_{uuid.uuid4().hex[:12]}"
        return {
            "workflow_id": local_id,
            "status": "planned",
            "persisted": False,
            "warning": f"Could not persist to workflow engine: {str(e)}"
        }
    except Exception as e:
        print(f"Unexpected error persisting workflow: {e}", file=sys.stderr)
        local_id = f"wf_local_{uuid.uuid4().hex[:12]}"
        return {
            "workflow_id": local_id,
            "status": "planned",
            "persisted": False,
            "warning": f"Unexpected error: {str(e)}"
        }


def _build_success_result(
    persist_result: Dict[str, Any],
    workflow_name: str,
    app_api_name: str,
    params: Dict[str, Any],
    auto_corrections: List[str],
    tool_name: str,
) -> Dict[str, Any]:
    """Build the standardized success response from a plan function."""
    result = {
        "workflow_id": persist_result["workflow_id"],
        "status": persist_result["status"],
        "persisted": persist_result.get("persisted", False),
        "workflow_name": workflow_name,
        "app": app_api_name,
        "parameters": params,
        "message": f"Workflow planned successfully. Workflow ID: {persist_result['workflow_id']}",
        "call": {
            "tool": tool_name,
            "arguments_executed": {"params": params},
            "replayable": True
        },
        "source": "bvbrc-service"
    }
    if persist_result.get("warning"):
        result["warning"] = persist_result["warning"]
    if auto_corrections:
        result["auto_corrections"] = auto_corrections
    return result


# ---------------------------------------------------------------------------
# Assembly recipe constants
# ---------------------------------------------------------------------------

ASSEMBLY_VALID_RECIPES = {
    "auto", "unicycler", "flye", "meta-flye", "canu",
    "spades", "meta-spades", "plasmid-spades", "single-cell", "megahit"
}

ASSEMBLY_RECIPE_ALIASES = {
    "metaflye": "meta-flye",
    "meta_flye": "meta-flye",
    "metaspades": "meta-spades",
    "meta_spades": "meta-spades",
    "plasmidspades": "plasmid-spades",
    "plasmid_spades": "plasmid-spades",
    "single_cell": "single-cell",
    "singlecell": "single-cell",
}

ASSEMBLY_DEFAULTS = {
    "recipe": "auto",
    "racon_iter": 2,
    "pilon_iter": 2,
    "trim": False,
    "normalize": False,
    "min_contig_len": 300,
    "min_contig_cov": 5,
    "genome_size": 5000000,
    "debug": 0,
}


# ---------------------------------------------------------------------------
# Annotation constants
# ---------------------------------------------------------------------------

ANNOTATION_VALID_DOMAINS = {"Bacteria", "Archaea", "Viruses", "auto"}
ANNOTATION_DOMAIN_ALIASES = {
    "bacteria": "Bacteria",
    "bacterial": "Bacteria",
    "archaea": "Archaea",
    "archaeal": "Archaea",
    "virus": "Viruses",
    "viruses": "Viruses",
    "viral": "Viruses",
}

ANNOTATION_VALID_CODES = {0, 1, 4, 11, 25}

ANNOTATION_DEFAULTS = {
    "code": 0,
    "domain": "auto",
    "public": False,
    "queue_nowait": False,
    "skip_indexing": False,
    "skip_workspace_output": False,
}


# ---------------------------------------------------------------------------
# Genome Assembly
# ---------------------------------------------------------------------------

async def plan_genome_assembly_fn(
    user_id: str,
    auth_token: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate, correct, build, and persist a genome assembly workflow.

    Accepts a free-form params dict from the LLM, validates and coerces
    all fields deterministically, then persists a single-step manifest.
    """
    if not isinstance(params, dict):
        params = {}

    auto_corrections: List[str] = []

    # --- Extract and coerce read inputs ---
    paired_end_libs = params.get("paired_end_libs")
    single_end_libs = params.get("single_end_libs")
    srr_ids = params.get("srr_ids")

    if paired_end_libs is not None:
        paired_end_libs = _coerce_to_list(paired_end_libs)
    if single_end_libs is not None:
        single_end_libs = _coerce_to_list(single_end_libs)
    if srr_ids is not None:
        srr_ids = _coerce_to_list(srr_ids)

    has_paired = paired_end_libs and len(paired_end_libs) > 0
    has_single = single_end_libs and len(single_end_libs) > 0
    has_srr = srr_ids and len(srr_ids) > 0

    if not (has_paired or has_single or has_srr):
        return {
            "error": "At least one read input is required: paired_end_libs, single_end_libs, or srr_ids",
            "errorType": "MISSING_PARAMETERS",
            "missing": ["paired_end_libs | single_end_libs | srr_ids"],
            "hint": "Provide sequencing reads as paired-end files, single-end files, or SRA Run IDs (e.g., SRR12345678)",
            "source": "bvbrc-service"
        }

    # --- Validate and coerce recipe ---
    recipe = params.get("recipe", "auto")
    if isinstance(recipe, str):
        matched = _fuzzy_match_enum(recipe, ASSEMBLY_VALID_RECIPES, ASSEMBLY_RECIPE_ALIASES)
        if matched is None:
            return {
                "error": f"Invalid assembly recipe: '{recipe}'",
                "errorType": "INVALID_PARAMETER",
                "parameter": "recipe",
                "valid_values": sorted(ASSEMBLY_VALID_RECIPES),
                "hint": "Use 'auto' if unsure which assembly algorithm to use",
                "source": "bvbrc-service"
            }
        if matched != recipe:
            auto_corrections.append(f"recipe: '{recipe}' -> '{matched}'")
            recipe = matched

    # --- Coerce numeric and boolean fields ---
    trim = _coerce_to_bool(params.get("trim", ASSEMBLY_DEFAULTS["trim"]))
    normalize = _coerce_to_bool(params.get("normalize", ASSEMBLY_DEFAULTS["normalize"]))
    racon_iter = _coerce_to_int(params.get("racon_iter"), ASSEMBLY_DEFAULTS["racon_iter"])
    pilon_iter = _coerce_to_int(params.get("pilon_iter"), ASSEMBLY_DEFAULTS["pilon_iter"])
    min_contig_len = _coerce_to_int(params.get("min_contig_len"), ASSEMBLY_DEFAULTS["min_contig_len"])
    min_contig_cov = _coerce_to_int(params.get("min_contig_cov"), ASSEMBLY_DEFAULTS["min_contig_cov"])
    genome_size = params.get("genome_size", ASSEMBLY_DEFAULTS["genome_size"])
    debug = _coerce_to_int(params.get("debug"), ASSEMBLY_DEFAULTS["debug"])

    # --- Resolve output path/file ---
    output_path = params.get("output_path")
    output_file = params.get("output_file")
    output_path, output_file = _default_output(user_id, "GenomeAssembly2", output_path, output_file)

    # --- Build final params ---
    final_params = {
        "recipe": recipe,
        "trim": trim,
        "normalize": normalize,
        "racon_iter": racon_iter,
        "pilon_iter": pilon_iter,
        "min_contig_len": min_contig_len,
        "min_contig_cov": min_contig_cov,
        "genome_size": genome_size,
        "debug": debug,
        "output_path": output_path,
        "output_file": output_file,
    }
    if has_paired:
        final_params["paired_end_libs"] = paired_end_libs
    if has_single:
        final_params["single_end_libs"] = single_end_libs
    if has_srr:
        final_params["srr_ids"] = srr_ids

    # --- Build manifest and persist ---
    outputs = _output_patterns("GenomeAssembly2")
    workflow_name = f"genome-assembly-{time.strftime('%Y%m%d-%H%M%S')}"
    manifest = _build_manifest(workflow_name, "assemble_reads", "GenomeAssembly2", final_params, outputs, user_id)

    persist_result = await _persist_to_engine(manifest, auth_token)

    return _build_success_result(
        persist_result, workflow_name, "GenomeAssembly2",
        final_params, auto_corrections, "plan_genome_assembly"
    )


# ---------------------------------------------------------------------------
# Genome Annotation
# ---------------------------------------------------------------------------

async def plan_genome_annotation_fn(
    user_id: str,
    auth_token: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate, correct, build, and persist a genome annotation workflow.

    Accepts a free-form params dict from the LLM, validates and coerces
    all fields deterministically, then persists a single-step manifest.
    """
    if not isinstance(params, dict):
        params = {}

    auto_corrections: List[str] = []

    # --- Validate required params ---
    contigs = params.get("contigs")
    scientific_name = params.get("scientific_name")

    missing = []
    if not contigs or not isinstance(contigs, str) or not contigs.strip():
        missing.append("contigs")
    if not scientific_name or not isinstance(scientific_name, str) or not scientific_name.strip():
        missing.append("scientific_name")

    if missing:
        hints = {
            "contigs": "Provide the workspace path to a contigs or FASTA file (e.g., /username/home/mycontigs.fasta)",
            "scientific_name": "Provide the organism's scientific name (e.g., 'Escherichia coli')"
        }
        return {
            "error": f"Missing required parameter(s): {', '.join(missing)}",
            "errorType": "MISSING_PARAMETERS",
            "missing": missing,
            "hints": {m: hints[m] for m in missing},
            "source": "bvbrc-service"
        }

    # --- Validate and coerce domain ---
    domain = params.get("domain", ANNOTATION_DEFAULTS["domain"])
    if isinstance(domain, str):
        matched = _fuzzy_match_enum(domain, ANNOTATION_VALID_DOMAINS, ANNOTATION_DOMAIN_ALIASES)
        if matched is None:
            return {
                "error": f"Invalid domain: '{domain}'",
                "errorType": "INVALID_PARAMETER",
                "parameter": "domain",
                "valid_values": sorted(ANNOTATION_VALID_DOMAINS),
                "hint": "Use 'auto' if unsure",
                "source": "bvbrc-service"
            }
        if matched != domain:
            auto_corrections.append(f"domain: '{domain}' -> '{matched}'")
            domain = matched

    # --- Validate and coerce code ---
    code = params.get("code", ANNOTATION_DEFAULTS["code"])
    code = _coerce_to_int(code, ANNOTATION_DEFAULTS["code"])
    if code not in ANNOTATION_VALID_CODES:
        return {
            "error": f"Invalid genetic code: {code}",
            "errorType": "INVALID_PARAMETER",
            "parameter": "code",
            "valid_values": sorted(ANNOTATION_VALID_CODES),
            "hint": "Use 0 for standard/auto genetic code",
            "source": "bvbrc-service"
        }

    # --- Coerce boolean fields ---
    public = _coerce_to_bool(params.get("public", ANNOTATION_DEFAULTS["public"]))
    queue_nowait = _coerce_to_bool(params.get("queue_nowait", ANNOTATION_DEFAULTS["queue_nowait"]))
    skip_indexing = _coerce_to_bool(params.get("skip_indexing", ANNOTATION_DEFAULTS["skip_indexing"]))
    skip_workspace_output = _coerce_to_bool(params.get("skip_workspace_output", ANNOTATION_DEFAULTS["skip_workspace_output"]))

    # --- Resolve output path/file ---
    output_path = params.get("output_path")
    output_file = params.get("output_file")
    output_path, output_file = _default_output(user_id, "GenomeAnnotation", output_path, output_file)

    # --- Build final params ---
    final_params = {
        "contigs": contigs.strip(),
        "scientific_name": scientific_name.strip(),
        "code": code,
        "domain": domain,
        "public": public,
        "queue_nowait": queue_nowait,
        "skip_indexing": skip_indexing,
        "skip_workspace_output": skip_workspace_output,
        "output_path": output_path,
        "output_file": output_file,
    }

    # Optional params
    taxonomy_id = params.get("taxonomy_id")
    if taxonomy_id is not None:
        taxonomy_id = _coerce_to_int(taxonomy_id)
        if taxonomy_id is not None and taxonomy_id > 0:
            final_params["taxonomy_id"] = taxonomy_id

    recipe = params.get("recipe")
    if recipe and isinstance(recipe, str) and recipe.strip():
        final_params["recipe"] = recipe.strip()

    # --- Build manifest and persist ---
    outputs = _output_patterns("GenomeAnnotation")
    workflow_name = f"genome-annotation-{time.strftime('%Y%m%d-%H%M%S')}"
    manifest = _build_manifest(workflow_name, "annotate_genome", "GenomeAnnotation", final_params, outputs, user_id)

    persist_result = await _persist_to_engine(manifest, auth_token)

    return _build_success_result(
        persist_result, workflow_name, "GenomeAnnotation",
        final_params, auto_corrections, "plan_genome_annotation"
    )


# ---------------------------------------------------------------------------
# Comparative Systems
# ---------------------------------------------------------------------------

async def _resolve_genome_group_names(
    genome_groups: List[str],
    api: Optional[JsonRpcCaller],
    auth_token: str,
    auto_corrections: List[str],
) -> List[str]:
    """
    Resolve genome group values to full workspace paths.

    If a value already looks like a workspace path (starts with '/'),
    it is kept as-is.  Otherwise, it is treated as a bare group name
    and resolved via resolve_group_path().
    """
    if not api:
        # No workspace API available — return values as-is
        return genome_groups

    resolved: List[str] = []
    for group_val in genome_groups:
        if group_val.startswith("/"):
            # Already a full workspace path
            resolved.append(group_val)
        else:
            # Bare name — resolve to workspace path
            result = await resolve_group_path(
                api, group_val, "genome_group", auth_token
            )
            if "path" in result:
                resolved.append(result["path"])
                auto_corrections.append(
                    f"Resolved genome group name '{group_val}' to path '{result['path']}'"
                )
                logger.info(
                    f"Resolved genome group '{group_val}' -> '{result['path']}'"
                )
            else:
                # Resolution failed — keep original value and log warning
                resolved.append(group_val)
                logger.warning(
                    f"Could not resolve genome group name '{group_val}': "
                    f"{result.get('error', 'unknown error')}"
                )
    return resolved


async def plan_comparative_systems_fn(
    user_id: str,
    auth_token: str,
    params: Dict[str, Any],
    api: Optional[JsonRpcCaller] = None,
) -> Dict[str, Any]:
    """
    Validate, correct, build, and persist a comparative systems workflow.

    Accepts a free-form params dict from the LLM, validates and coerces
    all fields deterministically, then persists a single-step manifest.
    """
    if not isinstance(params, dict):
        params = {}

    auto_corrections: List[str] = []

    # --- Extract and coerce genome inputs ---
    genome_ids = params.get("genome_ids")
    genome_groups = params.get("genome_groups")

    if genome_ids is not None:
        genome_ids = _coerce_to_list(genome_ids)
        # Ensure all IDs are strings
        genome_ids = [str(gid) for gid in genome_ids if gid is not None]
    if genome_groups is not None:
        genome_groups = _coerce_to_list(genome_groups)
        genome_groups = [str(g) for g in genome_groups if g is not None]
        # Resolve bare group names to full workspace paths
        genome_groups = await _resolve_genome_group_names(
            genome_groups, api, auth_token, auto_corrections
        )

    has_ids = genome_ids and len(genome_ids) > 0
    has_groups = genome_groups and len(genome_groups) > 0

    if not (has_ids or has_groups):
        return {
            "error": "At least one genome input is required: genome_ids or genome_groups",
            "errorType": "MISSING_PARAMETERS",
            "missing": ["genome_ids | genome_groups"],
            "hint": "Provide BV-BRC genome IDs (e.g., ['83332.12']) or workspace paths to genome groups",
            "source": "bvbrc-service"
        }

    # --- Resolve output path/file ---
    output_path = params.get("output_path")
    output_file = params.get("output_file")
    output_path, output_file = _default_output(user_id, "ComparativeSystems", output_path, output_file)

    # --- Build final params ---
    final_params = {
        "output_path": output_path,
        "output_file": output_file,
    }
    if has_ids:
        final_params["genome_ids"] = genome_ids
    if has_groups:
        final_params["genome_groups"] = genome_groups

    # --- Build manifest and persist ---
    outputs = _output_patterns("ComparativeSystems")
    workflow_name = f"comparative-systems-{time.strftime('%Y%m%d-%H%M%S')}"
    manifest = _build_manifest(workflow_name, "compare_systems", "ComparativeSystems", final_params, outputs, user_id)

    persist_result = await _persist_to_engine(manifest, auth_token)

    return _build_success_result(
        persist_result, workflow_name, "ComparativeSystems",
        final_params, auto_corrections, "plan_comparative_systems"
    )
