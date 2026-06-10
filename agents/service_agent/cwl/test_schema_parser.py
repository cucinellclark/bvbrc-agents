"""Tests for CWL Schema Parser (Milestone 2).

Covers:
- Real CWL tool parsing (GenomeAssembly2, GenomeAnnotation)
- Bulk parsing of all 35+ GoWe CWL tools
- CWL type parsing (all patterns)
- Enum extraction from doc strings
- BV-BRC tag extraction
- Comparison against hardcoded service_required_params.json
- Name mapping (friendly ↔ API)
- Supplemental rules merging
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.service_agent.cwl.schema_parser import (
    CWLTypeInfo,
    build_name_mapping,
    extract_bvbrc_tags,
    extract_enum_from_doc,
    load_supplemental_rules,
    parse_all_tools,
    parse_cwl_inputs,
    parse_cwl_tool,
    parse_cwl_type,
    _api_name_to_friendly,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # bvbrc-agents/
_AGENTS_ROOT = _REPO_ROOT.parent  # Agents/
_CWL_TOOLS_DIR = _AGENTS_ROOT / "GoWe" / "cwl" / "tools"
_CONFIG_DIR = _REPO_ROOT / "mcp_server" / "config"
_SERVICE_MAPPING = _CONFIG_DIR / "service_mapping.json"
_SERVICE_REQUIRED_PARAMS = _CONFIG_DIR / "service_required_params.json"
_SERVICE_OUTPUTS = _CONFIG_DIR / "service_outputs.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def genome_assembly_cwl():
    """Path to GenomeAssembly2.cwl."""
    return _CWL_TOOLS_DIR / "GenomeAssembly2.cwl"


@pytest.fixture
def genome_annotation_cwl():
    """Path to GenomeAnnotation.cwl."""
    return _CWL_TOOLS_DIR / "GenomeAnnotation.cwl"


@pytest.fixture
def service_mapping():
    """Load service_mapping.json as dict."""
    with open(_SERVICE_MAPPING, "r") as f:
        return json.load(f)


@pytest.fixture
def service_required_params():
    """Load service_required_params.json as dict."""
    with open(_SERVICE_REQUIRED_PARAMS, "r") as f:
        return json.load(f)


@pytest.fixture
def api_to_friendly(service_mapping):
    """Build api_name -> friendly_name mapping."""
    f2a = service_mapping.get("friendly_to_api", service_mapping)
    return {v: k for k, v in f2a.items()}


# ===========================================================================
# 1. Parse a real CWL tool: GenomeAssembly2
# ===========================================================================

class TestParseGenomeAssembly2:

    def test_api_name(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert schema["api_name"] == "GenomeAssembly2"

    def test_friendly_name(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert schema["friendly_name"] == "genome_assembly"

    def test_description_present(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert schema["description"]
        assert "Assemble" in schema["description"]

    def test_required_params(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert "output_path" in schema["required_params"]
        assert "output_file" in schema["required_params"]

    def test_defaults_recipe(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert schema["defaults"]["recipe"] == "auto"

    def test_defaults_racon_iter(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert schema["defaults"]["racon_iter"] == 2

    def test_defaults_trim(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert schema["defaults"]["trim"] is False

    def test_defaults_pilon_iter(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert schema["defaults"]["pilon_iter"] == 2

    def test_defaults_min_contig_len(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert schema["defaults"]["min_contig_len"] == 300

    def test_defaults_genome_size(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert schema["defaults"]["genome_size"] == 5000000

    def test_enum_params_recipe(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert "recipe" in schema["enum_params"]
        recipe_vals = schema["enum_params"]["recipe"]
        assert len(recipe_vals) == 10
        assert "auto" in recipe_vals
        assert "unicycler" in recipe_vals
        assert "megahit" in recipe_vals

    def test_required_one_of(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        one_of = schema["required_one_of"]
        assert "paired_end_libs" in one_of
        assert "single_end_libs" in one_of
        assert "srr_ids" in one_of

    def test_optional_params_include_recipe(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert "recipe" in schema["optional_params"]
        assert "racon_iter" in schema["optional_params"]

    def test_all_params_has_recipe(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert "recipe" in schema["all_params"]
        assert schema["all_params"]["recipe"]["type"] == "string"
        assert schema["all_params"]["recipe"]["optional"] is True
        assert schema["all_params"]["recipe"]["default"] == "auto"

    def test_output_path_is_required(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert "output_path" in schema["required_params"]
        assert schema["all_params"]["output_path"]["type"] == "Directory"

    def test_paired_end_libs_is_optional(self, genome_assembly_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        assert "paired_end_libs" not in schema["required_params"]
        assert schema["all_params"]["paired_end_libs"]["optional"] is True


# ===========================================================================
# 2. Parse GenomeAnnotation.cwl
# ===========================================================================

class TestParseGenomeAnnotation:

    def test_api_name(self, genome_annotation_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_annotation_cwl, name_mapping=api_to_friendly)
        assert schema["api_name"] == "GenomeAnnotation"

    def test_required_params_include_contigs(self, genome_annotation_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_annotation_cwl, name_mapping=api_to_friendly)
        assert "contigs" in schema["required_params"]

    def test_required_params_include_scientific_name(self, genome_annotation_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_annotation_cwl, name_mapping=api_to_friendly)
        assert "scientific_name" in schema["required_params"]

    def test_enum_params_domain(self, genome_annotation_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_annotation_cwl, name_mapping=api_to_friendly)
        assert "domain" in schema["enum_params"]
        domain_vals = schema["enum_params"]["domain"]
        assert "Bacteria" in domain_vals
        assert "Archaea" in domain_vals
        assert "Viruses" in domain_vals
        assert "auto" in domain_vals

    def test_enum_params_code(self, genome_annotation_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_annotation_cwl, name_mapping=api_to_friendly)
        assert "code" in schema["enum_params"]
        code_vals = schema["enum_params"]["code"]
        assert 0 in code_vals
        assert 11 in code_vals

    def test_defaults_code(self, genome_annotation_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_annotation_cwl, name_mapping=api_to_friendly)
        assert schema["defaults"]["code"] == 0

    def test_defaults_domain(self, genome_annotation_cwl, api_to_friendly):
        schema = parse_cwl_tool(genome_annotation_cwl, name_mapping=api_to_friendly)
        assert schema["defaults"]["domain"] == "auto"

    def test_bvbrc_tags_on_contigs(self, genome_annotation_cwl):
        schema = parse_cwl_tool(genome_annotation_cwl)
        assert "wstype" in schema["all_params"]["contigs"].get("bvbrc_tags", [])

    def test_bvbrc_tags_on_output_path(self, genome_annotation_cwl):
        schema = parse_cwl_tool(genome_annotation_cwl)
        assert "folder" in schema["all_params"]["output_path"].get("bvbrc_tags", [])


# ===========================================================================
# 3. Parse all 35+ tools
# ===========================================================================

class TestParseAllTools:

    def test_all_tools_parse_without_errors(self):
        """All CWL files in GoWe/cwl/tools/ should parse without errors."""
        if not _CWL_TOOLS_DIR.exists():
            pytest.skip("GoWe CWL tools directory not found")
        schemas = parse_all_tools(
            _CWL_TOOLS_DIR, service_mapping_path=_SERVICE_MAPPING,
        )
        assert len(schemas) >= 35, f"Expected >=35 tools, got {len(schemas)}"

    def test_all_tools_have_api_name(self):
        """Every parsed tool should have an api_name."""
        if not _CWL_TOOLS_DIR.exists():
            pytest.skip("GoWe CWL tools directory not found")
        schemas = parse_all_tools(
            _CWL_TOOLS_DIR, service_mapping_path=_SERVICE_MAPPING,
        )
        for api_name, schema in schemas.items():
            assert schema["api_name"], f"Missing api_name for {api_name}"

    def test_all_tools_have_description(self):
        """Every parsed tool should have a non-empty description."""
        if not _CWL_TOOLS_DIR.exists():
            pytest.skip("GoWe CWL tools directory not found")
        schemas = parse_all_tools(
            _CWL_TOOLS_DIR, service_mapping_path=_SERVICE_MAPPING,
        )
        for api_name, schema in schemas.items():
            assert schema["description"], f"Missing description for {api_name}"

    def test_known_tools_present(self):
        """Known tools from service_mapping.json should appear in parsed results."""
        if not _CWL_TOOLS_DIR.exists():
            pytest.skip("GoWe CWL tools directory not found")
        schemas = parse_all_tools(
            _CWL_TOOLS_DIR, service_mapping_path=_SERVICE_MAPPING,
        )
        expected_tools = [
            "GenomeAssembly2", "GenomeAnnotation", "Homology",
            "Variation", "RNASeq", "CodonTree", "TaxonomicClassification",
        ]
        for tool in expected_tools:
            assert tool in schemas, f"Missing expected tool: {tool}"

    def test_all_tools_have_required_schema_keys(self):
        """Every schema should have the expected keys."""
        if not _CWL_TOOLS_DIR.exists():
            pytest.skip("GoWe CWL tools directory not found")
        schemas = parse_all_tools(
            _CWL_TOOLS_DIR, service_mapping_path=_SERVICE_MAPPING,
        )
        required_keys = {
            "api_name", "friendly_name", "description",
            "required_params", "optional_params", "defaults",
            "enum_params", "required_one_of", "conditional_required",
            "required_outputs", "output_patterns", "all_params",
        }
        for api_name, schema in schemas.items():
            missing = required_keys - set(schema.keys())
            assert not missing, f"{api_name} missing keys: {missing}"


# ===========================================================================
# 4. Type parsing
# ===========================================================================

class TestParseCWLType:

    def test_required_string(self):
        info = parse_cwl_type("string")
        assert info.base_type == "string"
        assert info.is_optional is False
        assert info.is_array is False

    def test_optional_string(self):
        info = parse_cwl_type("string?")
        assert info.base_type == "string"
        assert info.is_optional is True
        assert info.is_array is False

    def test_optional_int(self):
        info = parse_cwl_type("int?")
        assert info.base_type == "int"
        assert info.is_optional is True

    def test_required_int(self):
        info = parse_cwl_type("int")
        assert info.base_type == "int"
        assert info.is_optional is False

    def test_required_file(self):
        info = parse_cwl_type("File")
        assert info.base_type == "File"
        assert info.is_optional is False

    def test_optional_file(self):
        info = parse_cwl_type("File?")
        assert info.base_type == "File"
        assert info.is_optional is True

    def test_required_directory(self):
        info = parse_cwl_type("Directory")
        assert info.base_type == "Directory"
        assert info.is_optional is False

    def test_optional_directory(self):
        info = parse_cwl_type("Directory?")
        assert info.base_type == "Directory"
        assert info.is_optional is True

    def test_optional_boolean(self):
        info = parse_cwl_type("boolean?")
        assert info.base_type == "boolean"
        assert info.is_optional is True

    def test_required_boolean(self):
        info = parse_cwl_type("boolean")
        assert info.base_type == "boolean"
        assert info.is_optional is False

    def test_optional_float(self):
        info = parse_cwl_type("float?")
        assert info.base_type == "float"
        assert info.is_optional is True

    def test_null_union_string(self):
        """["null", "string"] → optional string."""
        info = parse_cwl_type(["null", "string"])
        assert info.base_type == "string"
        assert info.is_optional is True
        assert info.is_array is False

    def test_null_union_array(self):
        """["null", {type: array, items: ...}] → optional array."""
        type_spec = [
            "null",
            {
                "type": "array",
                "items": {
                    "type": "record",
                    "name": "paired_end_lib",
                    "fields": [
                        {"name": "read1", "type": "File"},
                        {"name": "read2", "type": "File?"},
                    ],
                },
            },
        ]
        info = parse_cwl_type(type_spec)
        assert info.is_optional is True
        assert info.is_array is True
        assert info.base_type == "record"
        assert len(info.record_fields) == 2

    def test_string_array_shorthand(self):
        """"string[]" → required string array."""
        info = parse_cwl_type("string[]")
        assert info.base_type == "string"
        assert info.is_array is True
        assert info.is_optional is False

    def test_optional_string_array_shorthand(self):
        """"string[]?" → optional string array."""
        info = parse_cwl_type("string[]?")
        assert info.base_type == "string"
        assert info.is_array is True
        assert info.is_optional is True

    def test_null_union_record_no_array(self):
        """["null", {type: record, ...}] → optional record (not array)."""
        type_spec = [
            "null",
            {
                "type": "record",
                "name": "paired_end_lib",
                "fields": [{"name": "read", "type": "File"}],
            },
        ]
        info = parse_cwl_type(type_spec)
        assert info.is_optional is True
        assert info.is_array is False
        assert info.base_type == "record"
        assert len(info.record_fields) == 1

    def test_simple_type(self):
        """CWLTypeInfo.to_simple_type() for basic types."""
        assert parse_cwl_type("string").to_simple_type() == "string"
        assert parse_cwl_type("int").to_simple_type() == "int"
        assert parse_cwl_type("File").to_simple_type() == "File"
        assert parse_cwl_type("string[]?").to_simple_type() == "string[]"


# ===========================================================================
# 5. Enum extraction
# ===========================================================================

class TestExtractEnumFromDoc:

    def test_recipe_enum(self):
        doc = "Recipe [enum: auto, unicycler, flye] [bvbrc:enum]"
        result = extract_enum_from_doc(doc)
        assert result == ["auto", "unicycler", "flye"]

    def test_no_enum(self):
        doc = "No enum here"
        assert extract_enum_from_doc(doc) is None

    def test_domain_enum(self):
        doc = "Domain [enum: Bacteria, Archaea, Viruses, auto] [bvbrc:enum]"
        result = extract_enum_from_doc(doc)
        assert result == ["Bacteria", "Archaea", "Viruses", "auto"]
        assert len(result) == 4

    def test_empty_string(self):
        assert extract_enum_from_doc("") is None

    def test_none_input(self):
        assert extract_enum_from_doc(None) is None

    def test_single_value_enum(self):
        doc = "Type [enum: single_value] [bvbrc:enum]"
        result = extract_enum_from_doc(doc)
        assert result == ["single_value"]

    def test_numeric_enum(self):
        doc = "Code [enum: 0, 1, 4, 11, 25] [bvbrc:enum]"
        result = extract_enum_from_doc(doc)
        # Returns strings; type coercion happens in parse_cwl_inputs
        assert result == ["0", "1", "4", "11", "25"]


# ===========================================================================
# 6. BV-BRC tag extraction
# ===========================================================================

class TestExtractBvbrcTags:

    def test_folder_tag(self):
        doc = "Output path [bvbrc:folder]"
        tags = extract_bvbrc_tags(doc)
        assert tags == {"folder"}

    def test_wsid_tag(self):
        doc = "Output file [bvbrc:wsid]"
        tags = extract_bvbrc_tags(doc)
        assert tags == {"wsid"}

    def test_multiple_tags(self):
        doc = "Recipe [enum: auto, flye] [bvbrc:enum]"
        tags = extract_bvbrc_tags(doc)
        assert tags == {"enum"}

    def test_group_tag(self):
        doc = " [bvbrc:group]"
        tags = extract_bvbrc_tags(doc)
        assert tags == {"group"}

    def test_wstype_tag(self):
        doc = "Input contigs [bvbrc:wstype]"
        tags = extract_bvbrc_tags(doc)
        assert tags == {"wstype"}

    def test_no_tags(self):
        doc = "Just a description"
        assert extract_bvbrc_tags(doc) == set()

    def test_empty_string(self):
        assert extract_bvbrc_tags("") == set()

    def test_none_input(self):
        assert extract_bvbrc_tags(None) == set()

    def test_bool_tag(self):
        doc = "Make public [bvbrc:bool]"
        tags = extract_bvbrc_tags(doc)
        assert tags == {"bool"}


# ===========================================================================
# 7. Comparison test: CWL-derived vs hardcoded
# ===========================================================================

class TestCompareWithHardcoded:
    """Compare CWL-derived schemas against the hardcoded service_required_params.json."""

    def test_genome_assembly_required_params(
        self, genome_assembly_cwl, service_required_params, api_to_friendly,
    ):
        """CWL-derived required_params should be a superset of the hardcoded ones."""
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        hardcoded = service_required_params["genome_assembly"]
        hardcoded_required = set(hardcoded.get("required_params", []))
        cwl_required = set(schema["required_params"])
        # The CWL schema should contain at least the same required params
        assert hardcoded_required <= cwl_required, (
            f"Missing from CWL: {hardcoded_required - cwl_required}"
        )

    def test_genome_assembly_defaults(
        self, genome_assembly_cwl, service_required_params, api_to_friendly,
    ):
        """CWL-derived defaults should match or be a superset of hardcoded defaults.

        Note: Some legacy param names differ from CWL names (e.g. the legacy
        config uses "debug" while CWL uses "debug_level"). We track known
        renames explicitly and verify all others match exactly.
        """
        # Known renames: legacy_name -> cwl_name
        known_renames = {"debug": "debug_level"}

        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        hardcoded = service_required_params["genome_assembly"]
        for key, value in hardcoded.get("defaults", {}).items():
            cwl_key = known_renames.get(key, key)
            assert cwl_key in schema["defaults"], f"Missing default: {key} (looked up as {cwl_key})"
            assert schema["defaults"][cwl_key] == value, (
                f"Default mismatch for {key}: "
                f"CWL={schema['defaults'][cwl_key]} vs hardcoded={value}"
            )

    def test_genome_assembly_enum_params(
        self, genome_assembly_cwl, service_required_params, api_to_friendly,
    ):
        """CWL-derived enum_params should match or be a superset of hardcoded enums."""
        schema = parse_cwl_tool(genome_assembly_cwl, name_mapping=api_to_friendly)
        hardcoded = service_required_params["genome_assembly"]
        for param, values in hardcoded.get("enum_params", {}).items():
            assert param in schema["enum_params"], f"Missing enum param: {param}"
            cwl_vals = set(str(v) for v in schema["enum_params"][param])
            hc_vals = set(str(v) for v in values)
            assert hc_vals <= cwl_vals, (
                f"Missing enum values for {param}: {hc_vals - cwl_vals}"
            )


# ===========================================================================
# 8. Name mapping
# ===========================================================================

class TestNameMapping:

    def test_genome_assembly_mapping(self):
        """GenomeAssembly2 ↔ genome_assembly should work via service_mapping.json."""
        if not _CWL_TOOLS_DIR.exists():
            pytest.skip("GoWe CWL tools directory not found")
        schemas = parse_all_tools(
            _CWL_TOOLS_DIR, service_mapping_path=_SERVICE_MAPPING,
        )
        mapping = build_name_mapping(schemas, service_mapping_path=_SERVICE_MAPPING)
        assert mapping["friendly_to_api"]["genome_assembly"] == "GenomeAssembly2"
        assert mapping["api_to_friendly"]["GenomeAssembly2"] == "genome_assembly"

    def test_all_existing_mappings_present(self):
        """All mappings from service_mapping.json should be present."""
        if not _CWL_TOOLS_DIR.exists() or not _SERVICE_MAPPING.exists():
            pytest.skip("Required files not found")

        with open(_SERVICE_MAPPING, "r") as f:
            existing = json.load(f)
        existing_f2a = existing.get("friendly_to_api", existing)

        schemas = parse_all_tools(
            _CWL_TOOLS_DIR, service_mapping_path=_SERVICE_MAPPING,
        )
        mapping = build_name_mapping(schemas, service_mapping_path=_SERVICE_MAPPING)

        for friendly, api in existing_f2a.items():
            assert friendly in mapping["friendly_to_api"], (
                f"Missing friendly name: {friendly}"
            )
            assert mapping["friendly_to_api"][friendly] == api

    def test_api_name_to_friendly_helper(self):
        """Test the auto-generated snake_case name helper."""
        assert _api_name_to_friendly("GenomeAssembly2") == "genome_assembly2"
        assert _api_name_to_friendly("RNASeq") == "rna_seq"
        assert _api_name_to_friendly("SARS2Assembly") == "sars2_assembly"
        assert _api_name_to_friendly("MetaCATS") == "meta_cats"
        assert _api_name_to_friendly("") == ""

    def test_new_tools_get_auto_friendly_name(self):
        """Tools not in service_mapping.json should get auto-generated names."""
        if not _CWL_TOOLS_DIR.exists():
            pytest.skip("GoWe CWL tools directory not found")
        schemas = parse_all_tools(
            _CWL_TOOLS_DIR, service_mapping_path=_SERVICE_MAPPING,
        )
        mapping = build_name_mapping(schemas, service_mapping_path=_SERVICE_MAPPING)

        # All CWL tools should have a mapping
        for api_name in schemas:
            assert api_name in mapping["api_to_friendly"], (
                f"No friendly name for {api_name}"
            )


# ===========================================================================
# Additional: parse_cwl_inputs
# ===========================================================================

class TestParseCWLInputs:

    def test_simple_inputs(self):
        """Test parsing a simple inputs dict."""
        inputs = {
            "name": {"type": "string", "doc": "A name"},
            "count": {"type": "int?", "doc": "A count", "default": 5},
            "flag": {"type": "boolean?", "doc": "A flag [bvbrc:bool]", "default": False},
        }
        required, defaults, enums, all_info = parse_cwl_inputs(inputs)
        assert required == ["name"]
        assert defaults == {"count": 5, "flag": False}
        assert enums == {}
        assert len(all_info) == 3

    def test_enum_in_doc(self):
        """Test that enums are extracted from doc strings."""
        inputs = {
            "recipe": {
                "type": "string?",
                "doc": "Recipe [enum: auto, fast, slow] [bvbrc:enum]",
                "default": "auto",
            },
        }
        _, _, enums, _ = parse_cwl_inputs(inputs)
        assert "recipe" in enums
        assert enums["recipe"] == ["auto", "fast", "slow"]

    def test_int_enum_coercion(self):
        """Integer enums should be coerced from strings to ints."""
        inputs = {
            "code": {
                "type": "int?",
                "doc": "Code [enum: 0, 1, 4, 11, 25] [bvbrc:enum]",
                "default": 0,
            },
        }
        _, _, enums, _ = parse_cwl_inputs(inputs)
        assert enums["code"] == [0, 1, 4, 11, 25]


# ===========================================================================
# Additional: Supplemental rules
# ===========================================================================

class TestSupplementalRules:

    def test_load_nonexistent_file(self, tmp_path):
        result = load_supplemental_rules(tmp_path / "nonexistent.json")
        assert result == {}

    def test_load_none(self):
        result = load_supplemental_rules(None)
        assert result == {}

    def test_merge_supplemental_rules(self, genome_assembly_cwl, api_to_friendly, tmp_path):
        """Supplemental rules should override inferred required_one_of."""
        rules = {
            "GenomeAssembly2": {
                "required_one_of": ["paired_end_libs", "single_end_libs", "srr_ids"],
                "conditional_required": [
                    {"when": {"recipe": "canu"}, "require": ["genome_size"]}
                ],
            },
        }
        rules_file = tmp_path / "supplemental.json"
        rules_file.write_text(json.dumps(rules))

        supplemental = load_supplemental_rules(rules_file)
        schema = parse_cwl_tool(
            genome_assembly_cwl,
            supplemental_rules=supplemental,
            name_mapping=api_to_friendly,
        )
        assert schema["required_one_of"] == [
            "paired_end_libs", "single_end_libs", "srr_ids",
        ]
        assert len(schema["conditional_required"]) == 1
        assert schema["conditional_required"][0]["when"]["recipe"] == "canu"


# ===========================================================================
# Additional: Parse from dict (not file)
# ===========================================================================

class TestParseFromDict:

    def test_parse_dict_input(self):
        """parse_cwl_tool should accept a pre-parsed dict."""
        cwl_doc = {
            "cwlVersion": "v1.2",
            "class": "CommandLineTool",
            "doc": "Test tool — does things",
            "hints": {
                "gowe:Execution": {
                    "bvbrc_app_id": "TestTool",
                    "executor": "bvbrc",
                },
            },
            "inputs": {
                "input_file": {"type": "File", "doc": "Input [bvbrc:wstype]"},
                "output_path": {
                    "type": "Directory",
                    "doc": "Output path [bvbrc:folder]",
                },
                "output_file": {
                    "type": "string",
                    "doc": "Output name [bvbrc:wsid]",
                },
            },
            "outputs": {
                "result": {"type": "File[]"},
            },
        }
        schema = parse_cwl_tool(cwl_doc)
        assert schema["api_name"] == "TestTool"
        assert "input_file" in schema["required_params"]
        assert "output_path" in schema["required_params"]
        assert "output_file" in schema["required_params"]
