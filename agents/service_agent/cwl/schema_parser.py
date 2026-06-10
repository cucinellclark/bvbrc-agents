"""CWL Schema Parser — parse CWL v1.2 CommandLineTool definitions into
validation schemas compatible with ``validate_service_params()``.

Replaces the hand-curated ``service_required_params.json`` with schemas
derived directly from GoWe's ``.cwl`` tool files (the CWL source of truth).

Public API
----------
parse_cwl_tool        – Parse one CWL file/dict → ServiceSchema dict
parse_all_tools       – Parse all ``.cwl`` files in a directory
build_name_mapping    – Build friendly↔api name mapping dicts
extract_enum_from_doc – Extract ``[enum: ...]`` values from doc strings
extract_bvbrc_tags    – Extract ``[bvbrc:*]`` annotation tags
parse_cwl_type        – Parse a CWL type spec into structured info
parse_cwl_inputs      – Parse the ``inputs:`` section of a CWL tool
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml


# ---------------------------------------------------------------------------
# CWL type parsing
# ---------------------------------------------------------------------------

@dataclass
class CWLTypeInfo:
    """Structured information about a parsed CWL type specification."""
    base_type: str              # e.g. "string", "int", "File", "array", "record"
    is_optional: bool           # True if nullable (? suffix or union with null)
    is_array: bool              # True if array type
    record_fields: list         # Record fields (list of dicts) if record type
    items_type: Optional[str] = None  # Element type for arrays (e.g. "string")

    def to_simple_type(self) -> str:
        """Return a simplified type string for the ``all_params`` output."""
        if self.is_array and self.base_type == "record":
            return "record[]"
        if self.is_array:
            t = self.items_type or self.base_type
            return f"{t}[]"
        if self.base_type == "record":
            return "record"
        return self.base_type


def parse_cwl_type(type_spec: Any) -> CWLTypeInfo:
    """Parse a CWL type specification into structured :class:`CWLTypeInfo`.

    Handles all BV-BRC CWL patterns:
    - ``"string"`` → required string
    - ``"string?"`` → optional string
    - ``["null", "string"]`` → optional string
    - ``["null", {type: array, items: ...}]`` → optional array
    - ``{type: array, items: ...}`` → required array
    - ``{type: record, ...}`` → required record
    - ``"string[]"`` → required string array (shorthand)
    - ``"string[]?"`` → optional string array (shorthand)
    """
    if isinstance(type_spec, str):
        return _parse_type_string(type_spec)

    if isinstance(type_spec, dict):
        return _parse_type_dict(type_spec, is_optional=False)

    if isinstance(type_spec, list):
        return _parse_type_list(type_spec)

    # Fallback
    return CWLTypeInfo(
        base_type="string", is_optional=True, is_array=False, record_fields=[]
    )


def _parse_type_string(type_str: str) -> CWLTypeInfo:
    """Parse a simple CWL type string (e.g. ``"string?"``, ``"int[]?"``).

    Handles shorthand array notation (``type[]``) and optional suffix (``?``).
    """
    is_optional = type_str.endswith("?")
    if is_optional:
        type_str = type_str[:-1]

    # Array shorthand: "string[]", "int[]"
    if type_str.endswith("[]"):
        base = type_str[:-2]
        return CWLTypeInfo(
            base_type=base, is_optional=is_optional, is_array=True,
            record_fields=[], items_type=base,
        )

    return CWLTypeInfo(
        base_type=type_str, is_optional=is_optional, is_array=False,
        record_fields=[],
    )


def _parse_type_dict(type_dict: dict, is_optional: bool) -> CWLTypeInfo:
    """Parse a CWL type object (``{type: "array", items: ...}``)."""
    inner_type = type_dict.get("type", "string")

    if inner_type == "array":
        items = type_dict.get("items", {})
        if isinstance(items, str):
            items_type = items
            record_fields: list = []
            base = items_type
        elif isinstance(items, dict):
            items_type = items.get("type", "string")
            if items_type == "record":
                record_fields = items.get("fields", [])
                base = "record"
            else:
                record_fields = []
                base = items_type
        else:
            items_type = "string"
            record_fields = []
            base = "string"
        return CWLTypeInfo(
            base_type=base, is_optional=is_optional, is_array=True,
            record_fields=record_fields, items_type=items_type,
        )

    if inner_type == "record":
        record_fields = type_dict.get("fields", [])
        return CWLTypeInfo(
            base_type="record", is_optional=is_optional, is_array=False,
            record_fields=record_fields,
        )

    # Simple type inside a dict (unusual but valid)
    return CWLTypeInfo(
        base_type=str(inner_type), is_optional=is_optional, is_array=False,
        record_fields=[],
    )


def _parse_type_list(type_list: list) -> CWLTypeInfo:
    """Parse a CWL union type list (e.g. ``["null", "string"]``)."""
    non_null = [t for t in type_list if t != "null"]
    is_optional = len(non_null) < len(type_list)

    if not non_null:
        return CWLTypeInfo(
            base_type="null", is_optional=True, is_array=False, record_fields=[]
        )

    # Take the first non-null type
    first = non_null[0]
    if isinstance(first, str):
        info = _parse_type_string(first)
        info.is_optional = is_optional or info.is_optional
        return info
    if isinstance(first, dict):
        return _parse_type_dict(first, is_optional=is_optional)

    return CWLTypeInfo(
        base_type="string", is_optional=is_optional, is_array=False,
        record_fields=[],
    )


# ---------------------------------------------------------------------------
# Doc-string extraction helpers
# ---------------------------------------------------------------------------

# Matches [enum: val1, val2, val3]
_ENUM_RE = re.compile(r"\[enum:\s*(.+?)\]", re.IGNORECASE)

# Matches [bvbrc:tag]
_BVBRC_TAG_RE = re.compile(r"\[bvbrc:(\w+)\]", re.IGNORECASE)


def extract_enum_from_doc(doc_string: Optional[str]) -> Optional[List[str]]:
    """Extract enum values from a CWL doc string.

    Looks for ``[enum: val1, val2, ...]`` pattern.  Returns a list of
    stripped value strings, or ``None`` if no enum is found.

    Examples::

        >>> extract_enum_from_doc("Recipe [enum: auto, unicycler, flye]")
        ['auto', 'unicycler', 'flye']
        >>> extract_enum_from_doc("No enum here")
        None
    """
    if not doc_string:
        return None
    m = _ENUM_RE.search(doc_string)
    if not m:
        return None
    raw = m.group(1)
    return [v.strip() for v in raw.split(",") if v.strip()]


def extract_bvbrc_tags(doc_string: Optional[str]) -> Set[str]:
    """Extract BV-BRC annotation tags from a CWL doc string.

    Returns a set of tag names (without the ``bvbrc:`` prefix).

    Example::

        >>> extract_bvbrc_tags("Output path [bvbrc:folder]")
        {'folder'}
    """
    if not doc_string:
        return set()
    return {m.group(1).lower() for m in _BVBRC_TAG_RE.finditer(doc_string)}


def _clean_doc(doc_string: Optional[str]) -> str:
    """Return the doc string with annotation tags removed."""
    if not doc_string:
        return ""
    cleaned = _ENUM_RE.sub("", doc_string)
    cleaned = _BVBRC_TAG_RE.sub("", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

@dataclass
class ParamInfo:
    """Full info about a single CWL input parameter."""
    name: str
    type_info: CWLTypeInfo
    doc: str = ""
    default: Any = None
    has_default: bool = False
    enum_values: Optional[List[str]] = None
    bvbrc_tags: Set[str] = field(default_factory=set)

    @property
    def is_required(self) -> bool:
        """A parameter is required if not optional AND has no default."""
        return not self.type_info.is_optional and not self.has_default

    def to_dict(self) -> dict:
        """Serialize to dict for the ``all_params`` output."""
        d: dict = {
            "type": self.type_info.to_simple_type(),
            "optional": not self.is_required,
            "doc": _clean_doc(self.doc),
        }
        if self.has_default:
            d["default"] = self.default
        if self.enum_values:
            d["enum"] = self.enum_values
        if self.bvbrc_tags:
            d["bvbrc_tags"] = sorted(self.bvbrc_tags)
        if self.type_info.is_array:
            d["is_array"] = True
        if self.type_info.record_fields:
            d["record_fields"] = [
                _summarize_record_field(f) for f in self.type_info.record_fields
            ]
        return d


def _summarize_record_field(field_def: dict) -> dict:
    """Extract a summary dict from a CWL record field definition."""
    name = field_def.get("name", "")
    ftype = field_def.get("type", "string")
    doc = field_def.get("doc", "")
    info: dict = {"name": name, "type": str(ftype)}
    if doc:
        info["doc"] = doc
    if "default" in field_def:
        info["default"] = field_def["default"]
    return info


def parse_cwl_inputs(
    inputs_dict: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any], Dict[str, List], List[ParamInfo]]:
    """Parse the ``inputs:`` section of a CWL tool.

    Returns:
        A 4-tuple of:
        - **required_params** – list of required param names
        - **defaults** – ``{param_name: default_value}``
        - **enum_params** – ``{param_name: [values]}``
        - **all_params_info** – list of :class:`ParamInfo`
    """
    required_params: List[str] = []
    defaults: Dict[str, Any] = {}
    enum_params: Dict[str, List] = {}
    all_params: List[ParamInfo] = []

    for param_name, param_def in inputs_dict.items():
        type_spec = param_def.get("type", "string")
        doc = param_def.get("doc", "")
        has_default = "default" in param_def
        default_val = param_def.get("default")

        type_info = parse_cwl_type(type_spec)

        # Extract enum values from doc string
        enum_values = extract_enum_from_doc(doc)

        # Integer enums: try to coerce enum values to int if they look numeric
        if enum_values and type_info.base_type == "int":
            coerced: list = []
            for v in enum_values:
                try:
                    coerced.append(int(v))
                except (ValueError, TypeError):
                    coerced.append(v)
            enum_values = coerced

        # Extract BV-BRC annotation tags
        tags = extract_bvbrc_tags(doc)

        info = ParamInfo(
            name=param_name,
            type_info=type_info,
            doc=doc,
            default=default_val,
            has_default=has_default,
            enum_values=enum_values,
            bvbrc_tags=tags,
        )
        all_params.append(info)

        # Classify required vs optional
        if info.is_required:
            required_params.append(param_name)

        # Collect defaults
        if has_default:
            defaults[param_name] = default_val

        # Collect enums
        if enum_values:
            enum_params[param_name] = enum_values

    return required_params, defaults, enum_params, all_params


# ---------------------------------------------------------------------------
# Heuristic: required_one_of inference
# ---------------------------------------------------------------------------

# Known input groups that commonly form required_one_of sets
_READ_INPUT_GROUPS = [
    {"paired_end_libs", "single_end_libs", "srr_ids"},
    {"paired_end_libs", "single_end_libs", "srr_libs"},
    {"paired_end_libs", "single_end_libs", "srr_ids", "contigs"},
]


def _infer_required_one_of(all_params: List[ParamInfo]) -> List[List[str]]:
    """Heuristically infer ``required_one_of`` groups from input parameters.

    Looks for known patterns of optional input groups that should have at
    least one member present (e.g. read inputs for assembly tools).

    Returns a list of groups (each group is a list of param names).
    """
    param_names = {p.name for p in all_params}
    groups: List[List[str]] = []

    for candidate_set in _READ_INPUT_GROUPS:
        # Check if at least 2 of the candidate params exist and are optional
        matching = candidate_set & param_names
        if len(matching) >= 2:
            # Verify they're all optional
            matching_params = [p for p in all_params if p.name in matching]
            if all(not p.is_required for p in matching_params):
                groups.append(sorted(matching))
                break  # Only one read-input group per tool

    return groups


# ---------------------------------------------------------------------------
# Supplemental rules merging
# ---------------------------------------------------------------------------

def load_supplemental_rules(
    path: Union[str, Path, None] = None,
) -> Dict[str, Dict]:
    """Load supplemental domain rules from a JSON file.

    The file maps API names to additional schema fields::

        {
            "GenomeAssembly2": {
                "required_one_of": ["paired_end_libs", "single_end_libs", "srr_ids"],
                "conditional_required": [...]
            }
        }

    Returns an empty dict if path is None or the file doesn't exist.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _merge_supplemental(
    schema: Dict[str, Any], rules: Dict[str, Any],
) -> None:
    """Merge supplemental rules into a schema dict (in-place)."""
    if "required_one_of" in rules:
        schema["required_one_of"] = rules["required_one_of"]
    if "conditional_required" in rules:
        schema["conditional_required"] = rules["conditional_required"]
    # Allow overriding/adding any other field
    for key in rules:
        if key not in ("required_one_of", "conditional_required"):
            schema[key] = rules[key]


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def _extract_app_id(cwl_doc: dict) -> Optional[str]:
    """Extract the ``bvbrc_app_id`` from ``hints.gowe:Execution``."""
    hints = cwl_doc.get("hints", {})
    execution = hints.get("gowe:Execution", {})
    return execution.get("bvbrc_app_id")


def _extract_description(cwl_doc: dict) -> str:
    """Extract the tool description from the top-level ``doc`` field."""
    doc = cwl_doc.get("doc", "")
    return doc.strip() if doc else ""


def parse_cwl_tool(
    cwl_path_or_dict: Union[str, Path, dict],
    supplemental_rules: Optional[Dict[str, Dict]] = None,
    name_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Parse a CWL CommandLineTool definition into a validation schema.

    Args:
        cwl_path_or_dict: Path to a ``.cwl`` file or an already-parsed dict.
        supplemental_rules: Optional dict of ``{api_name: {rules}}`` to
            merge domain-specific rules (required_one_of, conditional_required).
        name_mapping: Optional ``{api_name: friendly_name}`` mapping.

    Returns:
        A dict with keys compatible with ``validate_service_params()``::

            {
                "api_name": "GenomeAssembly2",
                "friendly_name": "genome_assembly",
                "description": "...",
                "required_params": [...],
                "optional_params": {...},
                "defaults": {...},
                "enum_params": {...},
                "required_one_of": [...],
                "conditional_required": [...],
                "required_outputs": [],
                "output_patterns": {},
                "all_params": {...},
            }
    """
    # Load CWL document
    if isinstance(cwl_path_or_dict, dict):
        cwl_doc = cwl_path_or_dict
    else:
        path = Path(cwl_path_or_dict)
        with open(path, "r", encoding="utf-8") as f:
            cwl_doc = yaml.safe_load(f)

    api_name = _extract_app_id(cwl_doc) or ""
    description = _extract_description(cwl_doc)

    # Parse inputs
    inputs_dict = cwl_doc.get("inputs", {})
    required_params, defaults, enum_params, all_params = parse_cwl_inputs(inputs_dict)

    # Build optional_params: {name: doc} for non-required params
    optional_params: Dict[str, str] = {}
    for p in all_params:
        if not p.is_required:
            optional_params[p.name] = _clean_doc(p.doc)

    # Build all_params detail dict
    all_params_dict: Dict[str, dict] = {}
    for p in all_params:
        all_params_dict[p.name] = p.to_dict()

    # Infer required_one_of
    inferred_groups = _infer_required_one_of(all_params)
    # Flatten single group to a simple list (matches existing format)
    required_one_of: list = inferred_groups[0] if inferred_groups else []

    # Resolve friendly name
    friendly_name = ""
    if name_mapping:
        friendly_name = name_mapping.get(api_name, "")
    if not friendly_name:
        friendly_name = _api_name_to_friendly(api_name)

    # Build schema
    schema: Dict[str, Any] = {
        "api_name": api_name,
        "friendly_name": friendly_name,
        "description": description,
        "required_params": required_params,
        "optional_params": optional_params,
        "defaults": defaults,
        "enum_params": enum_params,
        "required_one_of": required_one_of,
        "conditional_required": [],
        "required_outputs": [],
        "output_patterns": {},
        "all_params": all_params_dict,
    }

    # Merge supplemental rules if available
    if supplemental_rules and api_name in supplemental_rules:
        _merge_supplemental(schema, supplemental_rules[api_name])

    return schema


# ---------------------------------------------------------------------------
# Bulk parsing
# ---------------------------------------------------------------------------

def parse_all_tools(
    tools_dir: Union[str, Path],
    supplemental_rules_path: Optional[Union[str, Path]] = None,
    service_mapping_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Parse all ``.cwl`` files in a directory.

    Args:
        tools_dir: Directory containing CWL tool files.
        supplemental_rules_path: Optional path to supplemental rules JSON.
        service_mapping_path: Optional path to ``service_mapping.json`` for
            name resolution.

    Returns:
        ``{api_name: ServiceSchema}`` dict.
    """
    tools_path = Path(tools_dir)
    supplemental = load_supplemental_rules(supplemental_rules_path)

    # Build api_to_friendly mapping
    api_to_friendly: Dict[str, str] = {}
    if service_mapping_path:
        sm_path = Path(service_mapping_path)
        if sm_path.exists():
            with open(sm_path, "r", encoding="utf-8") as f:
                sm = json.load(f)
            friendly_to_api = sm.get("friendly_to_api", sm)
            api_to_friendly = {v: k for k, v in friendly_to_api.items()}

    schemas: Dict[str, Dict[str, Any]] = {}
    for cwl_file in sorted(tools_path.glob("*.cwl")):
        try:
            schema = parse_cwl_tool(
                cwl_file,
                supplemental_rules=supplemental,
                name_mapping=api_to_friendly,
            )
            api_name = schema.get("api_name", cwl_file.stem)
            if api_name:
                schemas[api_name] = schema
        except Exception as exc:
            # Log but don't fail on individual parse errors
            import sys
            print(
                f"WARNING: Failed to parse {cwl_file.name}: {exc}",
                file=sys.stderr,
            )

    return schemas


# ---------------------------------------------------------------------------
# Name mapping
# ---------------------------------------------------------------------------

def _api_name_to_friendly(api_name: str) -> str:
    """Convert a CamelCase API name to a snake_case friendly name.

    Examples::

        >>> _api_name_to_friendly("GenomeAssembly2")
        'genome_assembly2'
        >>> _api_name_to_friendly("RNASeq")
        'rnaseq'
        >>> _api_name_to_friendly("SARS2Assembly")
        'sars2_assembly'
    """
    if not api_name:
        return ""
    # Insert underscore before uppercase letters that follow a lowercase
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", api_name)
    # Insert underscore between consecutive uppercase and following lowercase
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return s.lower()


def build_name_mapping(
    schemas: Dict[str, Dict[str, Any]],
    service_mapping_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, str]]:
    """Build bidirectional friendly↔API name mappings.

    Uses ``service_mapping.json`` as the primary source for known services,
    and falls back to auto-generated snake_case names for unknown ones.

    Args:
        schemas: Output from :func:`parse_all_tools`.
        service_mapping_path: Optional path to ``service_mapping.json``.

    Returns:
        ``{"friendly_to_api": {...}, "api_to_friendly": {...}}``
    """
    # Load existing mapping as the authority
    existing_f2a: Dict[str, str] = {}
    if service_mapping_path:
        sm_path = Path(service_mapping_path)
        if sm_path.exists():
            with open(sm_path, "r", encoding="utf-8") as f:
                sm = json.load(f)
            existing_f2a = sm.get("friendly_to_api", sm)

    existing_a2f = {v: k for k, v in existing_f2a.items()}

    friendly_to_api: Dict[str, str] = dict(existing_f2a)
    api_to_friendly: Dict[str, str] = dict(existing_a2f)

    # Add entries for any new CWL tools not in the existing mapping
    for api_name in schemas:
        if api_name not in api_to_friendly:
            friendly = _api_name_to_friendly(api_name)
            # Avoid clobbering existing mappings (e.g. GenomeAssembly
            # would auto-generate "genome_assembly" which already maps
            # to GenomeAssembly2 in the existing config)
            if friendly in friendly_to_api:
                friendly = friendly + "_alt"
            api_to_friendly[api_name] = friendly
            friendly_to_api[friendly] = api_name

    return {
        "friendly_to_api": friendly_to_api,
        "api_to_friendly": api_to_friendly,
    }
