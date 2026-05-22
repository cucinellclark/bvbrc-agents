"""
General-purpose service parameter validation for all BV-BRC services.

Validates parameters against service_required_params.json config,
applies defaults, fuzzy-matches enums, and returns structured results.
No LLM calls, no workflow engine calls — purely deterministic.

Used by the Service Agent's plan_service tool.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Shared helpers (canonical source — also imported by service_plan_functions)
# ---------------------------------------------------------------------------

def _load_config_file(filename: str) -> Dict:
    """Load a JSON config file from the config directory."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, 'config', filename)
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _fuzzy_match_enum(
    value: str,
    valid_values: set,
    aliases: Optional[Dict[str, str]] = None,
) -> Optional[str]:
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
    lower_map = {str(v).lower(): v for v in valid_values}
    lower_val = str(value).lower().strip()
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


def _default_output(
    user_id: str,
    app_api_name: str,
    output_path: Optional[str],
    output_file: Optional[str],
):
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


# ---------------------------------------------------------------------------
# Service catalog helpers
# ---------------------------------------------------------------------------

_SERVICE_MAPPING: Optional[Dict[str, str]] = None
_SERVICE_PARAMS: Optional[Dict[str, Dict]] = None


def _get_service_mapping() -> Dict[str, str]:
    """Load and cache service_mapping.json (friendly_name -> api_name)."""
    global _SERVICE_MAPPING
    if _SERVICE_MAPPING is None:
        data = _load_config_file('service_mapping.json')
        _SERVICE_MAPPING = data.get('friendly_to_api', data)
    return _SERVICE_MAPPING


def _get_service_params() -> Dict[str, Dict]:
    """Load and cache service_required_params.json."""
    global _SERVICE_PARAMS
    if _SERVICE_PARAMS is None:
        _SERVICE_PARAMS = _load_config_file('service_required_params.json')
    return _SERVICE_PARAMS


def _get_api_name(service_name: str) -> Optional[str]:
    """Get the API name for a friendly service name."""
    mapping = _get_service_mapping()
    return mapping.get(service_name)


# ---------------------------------------------------------------------------
# Service listing and schema retrieval
# ---------------------------------------------------------------------------

# Service categories for organization
SERVICE_CATEGORIES = {
    "Genomics": [
        "genome_assembly", "genome_annotation", "comprehensive_genome_analysis",
        "similar_genome_finder", "genome_alignment",
    ],
    "Phylogenomics": [
        "bacterial_genome_tree", "gene_tree", "core_genome_mlst",
        "whole_genome_snp",
    ],
    "Metagenomics": [
        "taxonomic_classification", "metagenomic_binning",
        "metagenomic_read_mapping", "metacats",
    ],
    "Transcriptomics": [
        "rnaseq", "expression_import",
    ],
    "Proteomics": [
        "proteome_comparison", "docking",
    ],
    "Variation Analysis": [
        "variation", "msa_snp_analysis",
    ],
    "Sequence Analysis": [
        "blast", "primer_design", "fastqutils",
    ],
    "Transposon Analysis": [
        "tnseq",
    ],
    "Viral Analysis": [
        "viral_assembly", "sars_genome_analysis", "sars_wastewater_analysis",
        "influenza_ha_subtype_conversion", "subspecies_classification",
    ],
    "Comparative Analysis": [
        "comparative_systems",
    ],
    "Submission": [
        "sequence_submission",
    ],
    "Utility": [
        "date",
    ],
}

# Reverse lookup: service_name -> category
_SERVICE_TO_CATEGORY: Dict[str, str] = {}
for cat, services in SERVICE_CATEGORIES.items():
    for svc in services:
        _SERVICE_TO_CATEGORY[svc] = cat


# Short descriptions for each service
SERVICE_DESCRIPTIONS = {
    "genome_assembly": "Assemble genomes from sequencing reads (Illumina, PacBio, Nanopore)",
    "genome_annotation": "Annotate assembled genomes with gene predictions and functional annotations",
    "comprehensive_genome_analysis": "Full pipeline: assembly, annotation, and quality analysis",
    "similar_genome_finder": "Find similar genomes in BV-BRC using Mash/MinHash",
    "genome_alignment": "Align multiple genomes using Mauve",
    "bacterial_genome_tree": "Build phylogenetic trees from bacterial genomes using codon trees",
    "gene_tree": "Build gene-level phylogenetic trees (RAxML, PhyML, FastTree)",
    "core_genome_mlst": "Core genome MLST analysis using chewBBACA",
    "whole_genome_snp": "Whole genome SNP analysis for phylogenomics",
    "taxonomic_classification": "Classify metagenomic reads taxonomically (Kraken2)",
    "metagenomic_binning": "Bin metagenomic assemblies into individual genomes",
    "metagenomic_read_mapping": "Map metagenomic reads to gene sets (VFDB, CARD)",
    "metacats": "Metagenomic comparative analysis (MetaCATS)",
    "rnaseq": "RNA-Seq analysis with differential expression",
    "expression_import": "Import gene expression datasets",
    "proteome_comparison": "Compare proteomes across genomes",
    "docking": "Molecular docking analysis",
    "variation": "SNP/variant calling from sequencing reads",
    "msa_snp_analysis": "Multiple sequence alignment and SNP analysis",
    "blast": "BLAST sequence homology search",
    "primer_design": "Design PCR primers",
    "fastqutils": "FASTQ quality control and preprocessing",
    "tnseq": "Transposon insertion sequencing analysis",
    "viral_assembly": "Assemble viral genomes from sequencing reads",
    "sars_genome_analysis": "SARS-CoV-2 genome assembly and analysis",
    "sars_wastewater_analysis": "SARS-CoV-2 wastewater surveillance analysis",
    "influenza_ha_subtype_conversion": "Influenza HA subtype conversion/classification",
    "subspecies_classification": "Viral subspecies/clade classification",
    "comparative_systems": "Compare pathways, subsystems, and protein families across genomes",
    "sequence_submission": "Submit sequences to public databases",
    "date": "Date/time utility service",
}


def list_services_fn() -> Dict[str, Any]:
    """List all available BV-BRC services with descriptions."""
    mapping = _get_service_mapping()

    services = []
    for friendly_name, api_name in sorted(mapping.items()):
        services.append({
            "name": friendly_name,
            "api_name": api_name,
            "category": _SERVICE_TO_CATEGORY.get(friendly_name, "Other"),
            "description": SERVICE_DESCRIPTIONS.get(friendly_name, ""),
        })

    return {
        "services": services,
        "count": len(services),
    }


def get_service_schema_fn(service_name: str) -> Dict[str, Any]:
    """Get the full parameter schema for a specific service."""
    all_params = _get_service_params()
    mapping = _get_service_mapping()

    if service_name not in all_params:
        return {
            "error": f"Unknown service: '{service_name}'",
            "available_services": sorted(all_params.keys()),
        }

    config = all_params[service_name]
    api_name = mapping.get(service_name, service_name)

    result = {
        "service_name": service_name,
        "api_name": api_name,
        "category": _SERVICE_TO_CATEGORY.get(service_name, "Other"),
        "description": SERVICE_DESCRIPTIONS.get(service_name, ""),
        "required_params": config.get("required_params", []),
        "defaults": config.get("defaults", {}),
    }

    if "enum_params" in config:
        result["enum_params"] = config["enum_params"]
    if "required_one_of" in config:
        result["required_one_of"] = config["required_one_of"]
    if "conditional_required" in config:
        result["conditional_required"] = config["conditional_required"]
    if "required_outputs" in config:
        result["required_outputs"] = config["required_outputs"]

    # Add output patterns
    output_patterns = _output_patterns(api_name)
    if output_patterns:
        result["output_patterns"] = output_patterns

    return result


# ---------------------------------------------------------------------------
# General service parameter validation
# ---------------------------------------------------------------------------

def validate_service_params(
    service_name: str,
    params: dict,
    user_id: str,
) -> dict:
    """
    General-purpose service parameter validation.

    1. Load service config from service_required_params.json
    2. Check required params
    3. Validate enum values (with fuzzy matching)
    4. Apply defaults
    5. Validate required_one_of
    6. Validate conditional_required
    7. Resolve output_path/output_file

    Returns:
        {
            "valid": True/False,
            "service_name": str,
            "api_name": str,
            "params": <validated params dict>,
            "auto_corrections": [...],
            "errors": [...],
            "missing": [...],
            "hints": {...}
        }
    """
    if not isinstance(params, dict):
        params = {}

    all_params = _get_service_params()
    mapping = _get_service_mapping()

    # 0. Check service exists
    if service_name not in all_params and service_name not in mapping.values():
        return {
            "valid": False,
            "errors": [f"Unknown service: '{service_name}'"],
            "available_services": sorted(all_params.keys()),
        }

    config = all_params.get(service_name, {})
    api_name = mapping.get(service_name, service_name)

    auto_corrections: List[str] = []
    errors: List[str] = []
    missing: List[str] = []
    hints: Dict[str, str] = {}

    # Work on a copy of params to apply defaults and corrections
    validated = dict(params)

    # 1. Apply defaults for missing optional params
    defaults = config.get("defaults", {})
    for key, default_val in defaults.items():
        if key not in validated:
            validated[key] = default_val

    # 2. Check required params (excluding output_path/output_file which get defaults)
    required = config.get("required_params", [])
    for req in required:
        if req in ("output_path", "output_file"):
            continue  # These get auto-generated defaults
        if req not in validated or validated[req] is None or validated[req] == "":
            missing.append(req)
            hints[req] = f"Required parameter '{req}' is missing."

    # 3. Validate enum params (with fuzzy matching)
    enum_params = config.get("enum_params", {})
    for param_name, valid_values_list in enum_params.items():
        if param_name not in validated:
            continue

        value = validated[param_name]
        valid_values_set = set(str(v) for v in valid_values_list)

        # Handle list values (e.g., recipe can be a list in some services)
        if isinstance(value, list):
            continue  # Skip enum validation for list values

        str_value = str(value)
        matched = _fuzzy_match_enum(str_value, valid_values_set)
        if matched is None:
            errors.append(
                f"Invalid value for '{param_name}': '{value}'. "
                f"Valid values: {sorted(valid_values_list)}"
            )
        elif matched != str_value:
            auto_corrections.append(f"{param_name}: '{value}' -> '{matched}'")
            # Try to preserve original type
            if isinstance(valid_values_list[0], int):
                try:
                    validated[param_name] = int(matched)
                except (ValueError, TypeError):
                    validated[param_name] = matched
            else:
                validated[param_name] = matched

    # 4. Validate required_one_of
    required_one_of = config.get("required_one_of", [])
    if required_one_of:
        has_any = False
        for field in required_one_of:
            val = validated.get(field)
            if val is not None:
                if isinstance(val, list) and len(val) > 0:
                    has_any = True
                    break
                elif isinstance(val, str) and val.strip():
                    has_any = True
                    break
                elif not isinstance(val, (list, str)):
                    has_any = True
                    break
        if not has_any:
            missing.append(" | ".join(required_one_of))
            hints["required_one_of"] = (
                f"At least one of these must be provided: {', '.join(required_one_of)}"
            )

    # 5. Validate conditional_required
    conditional_required = config.get("conditional_required", [])
    for condition in conditional_required:
        when = condition.get("when", {})
        condition_met = True
        for cond_key, cond_val in when.items():
            if validated.get(cond_key) != cond_val:
                condition_met = False
                break

        if condition_met:
            # Check "require" (all must be present)
            required_fields = condition.get("require", [])
            for rf in required_fields:
                if rf not in validated or validated[rf] is None or validated[rf] == "":
                    missing.append(rf)
                    cond_desc = " AND ".join(f"{k}={v}" for k, v in when.items())
                    hints[rf] = f"Required when {cond_desc}."

            # Check "require_one_of"
            require_one_of = condition.get("require_one_of", [])
            if require_one_of:
                has_any_cond = any(
                    validated.get(f) is not None and validated.get(f) != ""
                    and (not isinstance(validated.get(f), list) or len(validated.get(f)) > 0)
                    for f in require_one_of
                )
                if not has_any_cond:
                    cond_desc = " AND ".join(f"{k}={v}" for k, v in when.items())
                    missing.append(" | ".join(require_one_of))
                    hints["conditional_one_of"] = (
                        f"When {cond_desc}, at least one of "
                        f"{', '.join(require_one_of)} is required."
                    )

    # 5b. Warn about multiple SRA IDs for assembly-type services
    warnings: List[str] = []
    assembly_services = {
        "genome_assembly", "comprehensive_genome_analysis", "viral_assembly",
    }
    if service_name in assembly_services:
        srr_ids = validated.get("srr_ids", [])
        if isinstance(srr_ids, list) and len(srr_ids) > 1:
            warnings.append(
                f"Multiple SRA IDs ({len(srr_ids)}) provided for {service_name}. "
                "Each SRA ID typically represents an independent sample and should "
                "be assembled separately. Consider creating one assembly step per "
                "SRA ID unless these are replicate reads from the same sample."
            )

    # 6. Resolve output_path/output_file
    output_path = validated.get("output_path")
    output_file = validated.get("output_file")
    output_path, output_file = _default_output(user_id, api_name, output_path, output_file)
    validated["output_path"] = output_path
    validated["output_file"] = output_file

    # 7. Coerce boolean and integer fields based on defaults
    for key, default_val in defaults.items():
        if key in validated:
            if isinstance(default_val, bool):
                validated[key] = _coerce_to_bool(validated[key])
            elif isinstance(default_val, int) and not isinstance(default_val, bool):
                coerced = _coerce_to_int(validated[key], default_val)
                if coerced is not None:
                    validated[key] = coerced

    # Build result
    is_valid = len(errors) == 0 and len(missing) == 0

    result = {
        "valid": is_valid,
        "service_name": service_name,
        "api_name": api_name,
    }

    if is_valid:
        result["status"] = "planned"
        result["params"] = validated
        result["output_patterns"] = _output_patterns(api_name)
        if auto_corrections:
            result["auto_corrections"] = auto_corrections
        if warnings:
            result["warnings"] = warnings
    else:
        result["status"] = "validation_failed"
        result["params"] = validated  # Include partially validated params
        if errors:
            result["errors"] = errors
        if missing:
            result["missing"] = missing
        if hints:
            result["hints"] = hints
        if auto_corrections:
            result["auto_corrections"] = auto_corrections
        if warnings:
            result["warnings"] = warnings

    return result
