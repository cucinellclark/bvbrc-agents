"""Comprehensive unit tests for CWL generation library.

Tests cover:
1. Single-step workflow
2. Linear pipeline (assembly -> annotation)
3. Fan-out (one step feeds two downstream)
4. Fan-in (two steps feed one downstream)
5. Workspace path handling
6. Type inference
7. Submission inputs generation
8. Output wiring (output_of references)

Run with:
    pytest agents/service_agent/cwl/test_generator.py -v
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from .generator import (
    generate_cwl_workflow,
    generate_submission_inputs,
    generate_tool_definition,
    wire_step_inputs,
)
from .types import (
    infer_cwl_type,
    infer_cwl_input_type,
    is_workspace_path,
    python_to_cwl_value,
    wrap_workspace_path,
)


# ---------------------------------------------------------------------------
# Test fixtures — lightweight stand-ins for Pydantic models
# ---------------------------------------------------------------------------

@dataclass
class FakeStep:
    """Stand-in for ValidatedStep to avoid importing the full models module."""

    step_id: str
    service_name: str
    api_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    output_patterns: dict[str, str] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    auto_corrections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FakeStepPlan:
    """Stand-in for StepPlan."""

    step_id: str
    service_name: str
    intent: str = ""
    depends_on: list[str] = field(default_factory=list)
    input_sources: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeWorkflowPlan:
    """Stand-in for WorkflowPlan with just enough interface."""

    workflow_name: str
    description: str
    steps: list[FakeStepPlan] = field(default_factory=list)
    topological_order: list[str] = field(default_factory=list)

    def compute_topological_order(self) -> list[str]:
        """Simple topological sort for tests."""
        from collections import deque

        in_degree = {s.step_id: 0 for s in self.steps}
        adj: dict[str, list[str]] = {s.step_id: [] for s in self.steps}
        for s in self.steps:
            for d in s.depends_on:
                adj[d].append(s.step_id)
                in_degree[s.step_id] += 1

        queue = deque(sid for sid, deg in in_degree.items() if deg == 0)
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for nb in adj[node]:
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    queue.append(nb)

        self.topological_order = order
        return order


# ---------------------------------------------------------------------------
# Helper factory functions
# ---------------------------------------------------------------------------

def _assembly_step(
    step_id: str = "assemble",
    output_patterns: dict[str, str] | None = None,
) -> FakeStep:
    """Create a typical genome assembly step."""
    return FakeStep(
        step_id=step_id,
        service_name="genome_assembly",
        api_name="GenomeAssembly2",
        params={
            "recipe": "auto",
            "output_path": "/user@bvbrc/home/test",
            "output_file": "assembly_output",
            "trim": False,
            "min_contig_len": 300,
            "srr_ids": ["SRR12345678"],
        },
        output_patterns=(
            output_patterns
            if output_patterns is not None
            else {"contigs": "/user@bvbrc/home/test/contigs.fasta"}
        ),
        depends_on=[],
    )


def _annotation_step(
    step_id: str = "annotate",
    depends_on: list[str] | None = None,
) -> FakeStep:
    """Create a typical genome annotation step."""
    return FakeStep(
        step_id=step_id,
        service_name="genome_annotation",
        api_name="GenomeAnnotation",
        params={
            "contigs": "output_of:assemble:contigs",
            "scientific_name": "Escherichia coli K-12",
            "taxonomy_id": 83333,
            "code": 11,
            "domain": "Bacteria",
            "output_path": "/user@bvbrc/home/test",
            "output_file": "annotation_output",
        },
        output_patterns={"annotated_genome": "*.genome"},
        depends_on=depends_on or ["assemble"],
    )


def _model_step(
    step_id: str = "model",
    depends_on: list[str] | None = None,
) -> FakeStep:
    """Create a typical model reconstruction step."""
    return FakeStep(
        step_id=step_id,
        service_name="model_reconstruction",
        api_name="ModelReconstruction",
        params={
            "genome": "output_of:annotate:annotated_genome",
            "output_path": "/user@bvbrc/home/test",
            "output_file": "model_output",
        },
        output_patterns={"model": "*.model"},
        depends_on=depends_on or ["annotate"],
    )


def _make_plan(
    steps: list[FakeStep],
    name: str = "test_workflow",
    description: str = "Test workflow",
) -> FakeWorkflowPlan:
    """Create a WorkflowPlan from FakeSteps."""
    step_plans = [
        FakeStepPlan(
            step_id=s.step_id,
            service_name=s.service_name,
            depends_on=s.depends_on,
        )
        for s in steps
    ]
    plan = FakeWorkflowPlan(
        workflow_name=name,
        description=description,
        steps=step_plans,
    )
    plan.compute_topological_order()
    return plan


# ===================================================================
# 1. Types module tests
# ===================================================================


class TestIsWorkspacePath:
    """Test workspace path detection."""

    def test_standard_bvbrc_path(self):
        assert is_workspace_path("/user@bvbrc/home/folder") is True

    def test_patricbrc_path(self):
        assert is_workspace_path("/user@patricbrc.org/home/folder") is True

    def test_deep_path(self):
        assert is_workspace_path("/user@bvbrc/home/project/subdir/file.fasta") is True

    def test_not_workspace_path_plain_string(self):
        assert is_workspace_path("some string") is False

    def test_not_workspace_path_relative(self):
        assert is_workspace_path("user@bvbrc/home") is False

    def test_not_workspace_path_no_at(self):
        assert is_workspace_path("/home/local/file.txt") is False

    def test_empty_string(self):
        assert is_workspace_path("") is False

    def test_not_workspace_path_wrong_domain(self):
        assert is_workspace_path("/user@gmail.com/home") is False


class TestWrapWorkspacePath:
    """Test workspace path wrapping to CWL objects."""

    def test_wrap_as_directory(self):
        result = wrap_workspace_path("/user@bvbrc/home/folder", as_type="Directory")
        assert result == {
            "class": "Directory",
            "location": "ws:///user@bvbrc/home/folder",
        }

    def test_wrap_as_file(self):
        result = wrap_workspace_path("/user@bvbrc/home/file.fasta", as_type="File")
        assert result == {
            "class": "File",
            "location": "ws:///user@bvbrc/home/file.fasta",
        }

    def test_ws_prefix_format(self):
        """ws:// + bare_path (bare path has leading /) -> ws:///"""
        result = wrap_workspace_path("/user@bvbrc/home/test")
        assert result["location"].startswith("ws:///")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            wrap_workspace_path("/user@bvbrc/home", as_type="Record")


class TestInferCwlType:
    """Test CWL type inference from Python values."""

    def test_string(self):
        assert infer_cwl_type("hello") == "string"

    def test_int(self):
        assert infer_cwl_type(42) == "int"

    def test_float(self):
        assert infer_cwl_type(3.14) == "float"

    def test_bool_true(self):
        assert infer_cwl_type(True) == "boolean"

    def test_bool_false(self):
        assert infer_cwl_type(False) == "boolean"

    def test_none(self):
        assert infer_cwl_type(None) == "null"

    def test_string_list(self):
        assert infer_cwl_type(["a", "b"]) == "string[]"

    def test_int_list(self):
        assert infer_cwl_type([1, 2, 3]) == "int[]"

    def test_empty_list(self):
        assert infer_cwl_type([]) == "string[]"

    def test_dict_with_file_class(self):
        assert infer_cwl_type({"class": "File", "location": "ws:///..."}) == "File"

    def test_dict_with_directory_class(self):
        assert infer_cwl_type({"class": "Directory", "location": "ws:///..."}) == "Directory"

    def test_plain_dict(self):
        assert infer_cwl_type({"key": "value"}) == "string"

    def test_bool_before_int(self):
        """bool is subclass of int — must be detected as boolean, not int."""
        assert infer_cwl_type(True) == "boolean"
        assert infer_cwl_type(False) == "boolean"


class TestInferCwlInputType:
    """Test context-aware CWL type inference."""

    def test_output_path_is_directory(self):
        assert infer_cwl_input_type("/user@bvbrc/home/test", "output_path") == "Directory"

    def test_output_file_is_string(self):
        assert infer_cwl_input_type("my_output", "output_file") == "string"

    def test_workspace_file_path(self):
        assert infer_cwl_input_type(
            "/user@bvbrc/home/test/file.fasta", "contigs"
        ) == "File"

    def test_workspace_dir_path(self):
        assert infer_cwl_input_type(
            "/user@bvbrc/home/test/folder", "some_dir"
        ) == "Directory"

    def test_plain_string(self):
        assert infer_cwl_input_type("auto", "recipe") == "string"


class TestPythonToCwlValue:
    """Test Python to CWL value conversion."""

    def test_plain_string(self):
        assert python_to_cwl_value("hello") == "hello"

    def test_int(self):
        assert python_to_cwl_value(42) == 42

    def test_float(self):
        assert python_to_cwl_value(3.14) == 3.14

    def test_bool(self):
        assert python_to_cwl_value(True) is True

    def test_none(self):
        assert python_to_cwl_value(None) is None

    def test_output_file_stays_string(self):
        result = python_to_cwl_value("my_output", param_name="output_file")
        assert result == "my_output"
        assert isinstance(result, str)

    def test_output_path_wrapped_as_directory(self):
        result = python_to_cwl_value(
            "/user@bvbrc/home/test", param_name="output_path"
        )
        assert result == {
            "class": "Directory",
            "location": "ws:///user@bvbrc/home/test",
        }

    def test_workspace_file_wrapped(self):
        result = python_to_cwl_value(
            "/user@bvbrc/home/test/reads.fastq", param_name="read1"
        )
        assert result == {
            "class": "File",
            "location": "ws:///user@bvbrc/home/test/reads.fastq",
        }

    def test_workspace_dir_wrapped(self):
        result = python_to_cwl_value(
            "/user@bvbrc/home/test/folder", param_name="some_dir"
        )
        assert result == {
            "class": "Directory",
            "location": "ws:///user@bvbrc/home/test/folder",
        }

    def test_list_of_strings(self):
        result = python_to_cwl_value(["SRR123", "SRR456"])
        assert result == ["SRR123", "SRR456"]

    def test_already_cwl_object(self):
        obj = {"class": "File", "location": "ws:///user@bvbrc/home/f.fa"}
        assert python_to_cwl_value(obj) == obj


# ===================================================================
# 2. Tool definition tests
# ===================================================================


class TestGenerateToolDefinition:
    """Test CWL CommandLineTool generation."""

    def test_basic_structure(self):
        step = _assembly_step()
        tool = generate_tool_definition(step)

        assert tool["id"] == "bvbrc-assemble"
        assert tool["class"] == "CommandLineTool"
        assert tool["baseCommand"] == ["GenomeAssembly2"]

    def test_gowe_execution_hint(self):
        step = _assembly_step()
        tool = generate_tool_definition(step)

        hints = tool["hints"]
        assert "gowe:Execution" in hints
        assert hints["gowe:Execution"]["bvbrc_app_id"] == "GenomeAssembly2"
        assert hints["gowe:Execution"]["executor"] == "bvbrc"

    def test_inputs_from_params(self):
        step = _assembly_step()
        tool = generate_tool_definition(step)

        inputs = tool["inputs"]
        assert "recipe" in inputs
        assert inputs["recipe"]["type"] == "string"
        assert "output_path" in inputs
        assert inputs["output_path"]["type"] == "Directory"
        assert "output_file" in inputs
        assert inputs["output_file"]["type"] == "string"
        assert "trim" in inputs
        assert inputs["trim"]["type"] == "boolean"
        assert "min_contig_len" in inputs
        assert inputs["min_contig_len"]["type"] == "int"

    def test_default_values_set(self):
        step = _assembly_step()
        tool = generate_tool_definition(step)

        inputs = tool["inputs"]
        assert inputs["recipe"].get("default") == "auto"
        assert inputs["trim"].get("default") is False
        assert inputs["min_contig_len"].get("default") == 300

    def test_outputs_with_patterns(self):
        step = _assembly_step(
            output_patterns={"contigs": "/user@bvbrc/home/test/contigs.fasta"}
        )
        tool = generate_tool_definition(step)

        outputs = tool["outputs"]
        assert "contigs" in outputs
        assert outputs["contigs"]["type"] == "File"
        assert "outputBinding" in outputs["contigs"]

    def test_outputs_generic_when_empty(self):
        step = _assembly_step(output_patterns={})
        tool = generate_tool_definition(step)

        outputs = tool["outputs"]
        assert "result" in outputs
        assert outputs["result"]["type"] == "File[]"
        assert "$(inputs.output_path.location)" in outputs["result"]["outputBinding"]["glob"]

    def test_underscore_in_step_id(self):
        step = FakeStep(
            step_id="my_step_name",
            service_name="test",
            api_name="TestApp",
            params={"x": "y"},
        )
        tool = generate_tool_definition(step)
        assert tool["id"] == "bvbrc-my-step-name"


# ===================================================================
# 3. Step wiring tests
# ===================================================================


class TestWireStepInputs:
    """Test inter-step input wiring resolution."""

    def test_output_of_reference(self):
        assembly = _assembly_step()
        annotation = _annotation_step()
        all_steps = {"assemble": assembly, "annotate": annotation}

        wiring = wire_step_inputs(annotation, all_steps, {})

        # contigs should be wired to upstream step output
        assert wiring["contigs"] == "assemble/contigs"

    def test_user_provided_becomes_workflow_input(self):
        assembly = _assembly_step()
        annotation = _annotation_step()
        all_steps = {"assemble": assembly, "annotate": annotation}

        wiring = wire_step_inputs(annotation, all_steps, {})

        # scientific_name is a direct value, becomes workflow input
        assert wiring["scientific_name"] == "annotate_scientific_name"

    def test_common_params_not_prefixed(self):
        assembly = _assembly_step()
        all_steps = {"assemble": assembly}

        wiring = wire_step_inputs(assembly, all_steps, {})

        # output_path and output_file should be common (no prefix)
        assert wiring["output_path"] == "output_path"
        assert wiring["output_file"] == "output_file"

    def test_chained_output_reference(self):
        """annotation -> model: genome should wire to annotate/annotated_genome."""
        annotation = _annotation_step()
        model = _model_step()
        all_steps = {"annotate": annotation, "model": model}

        wiring = wire_step_inputs(model, all_steps, {})

        assert wiring["genome"] == "annotate/annotated_genome"


# ===================================================================
# 4. Full workflow generation tests
# ===================================================================


class TestSingleStepWorkflow:
    """Test single-step workflow generation."""

    def test_structure(self):
        step = _assembly_step()
        plan = _make_plan([step], name="single_assembly")
        completed = {"assemble": step}

        cwl = generate_cwl_workflow(plan, completed)

        assert cwl["cwlVersion"] == "v1.2"
        assert cwl["$namespaces"]["gowe"] == "https://github.com/wilke/GoWe#"
        assert len(cwl["$graph"]) == 2  # 1 tool + 1 workflow

    def test_tool_in_graph(self):
        step = _assembly_step()
        plan = _make_plan([step])
        completed = {"assemble": step}

        cwl = generate_cwl_workflow(plan, completed)
        graph = cwl["$graph"]

        tool = graph[0]
        assert tool["class"] == "CommandLineTool"
        assert tool["id"] == "bvbrc-assemble"

    def test_workflow_in_graph(self):
        step = _assembly_step()
        plan = _make_plan([step])
        completed = {"assemble": step}

        cwl = generate_cwl_workflow(plan, completed)
        graph = cwl["$graph"]

        wf = graph[-1]
        assert wf["class"] == "Workflow"
        assert wf["id"] == "main"

    def test_workflow_step_references_tool(self):
        step = _assembly_step()
        plan = _make_plan([step])
        completed = {"assemble": step}

        cwl = generate_cwl_workflow(plan, completed)
        wf = cwl["$graph"][-1]

        assert "assemble" in wf["steps"]
        assert wf["steps"]["assemble"]["run"] == "#bvbrc-assemble"

    def test_workflow_has_inputs(self):
        step = _assembly_step()
        plan = _make_plan([step])
        completed = {"assemble": step}

        cwl = generate_cwl_workflow(plan, completed)
        wf = cwl["$graph"][-1]

        assert "output_path" in wf["inputs"]
        assert "output_file" in wf["inputs"]

    def test_workflow_has_outputs(self):
        step = _assembly_step()
        plan = _make_plan([step])
        completed = {"assemble": step}

        cwl = generate_cwl_workflow(plan, completed)
        wf = cwl["$graph"][-1]

        assert len(wf["outputs"]) > 0


class TestLinearPipeline:
    """Test 2-step linear pipeline: assembly -> annotation."""

    def setup_method(self):
        self.assembly = _assembly_step()
        self.annotation = _annotation_step()
        self.plan = _make_plan(
            [self.assembly, self.annotation],
            name="assembly_annotation_pipeline",
        )
        self.completed = {
            "assemble": self.assembly,
            "annotate": self.annotation,
        }

    def test_graph_has_three_entries(self):
        """2 tools + 1 workflow = 3 entries."""
        cwl = generate_cwl_workflow(self.plan, self.completed)
        assert len(cwl["$graph"]) == 3

    def test_tool_order_matches_topology(self):
        cwl = generate_cwl_workflow(self.plan, self.completed)
        graph = cwl["$graph"]

        assert graph[0]["id"] == "bvbrc-assemble"
        assert graph[1]["id"] == "bvbrc-annotate"
        assert graph[2]["id"] == "main"

    def test_annotation_wired_to_assembly_output(self):
        cwl = generate_cwl_workflow(self.plan, self.completed)
        wf = cwl["$graph"][-1]

        annotate_step = wf["steps"]["annotate"]
        assert annotate_step["in"]["contigs"] == "assemble/contigs"

    def test_annotation_step_references_tool(self):
        cwl = generate_cwl_workflow(self.plan, self.completed)
        wf = cwl["$graph"][-1]

        assert wf["steps"]["annotate"]["run"] == "#bvbrc-annotate"

    def test_assembly_step_references_tool(self):
        cwl = generate_cwl_workflow(self.plan, self.completed)
        wf = cwl["$graph"][-1]

        assert wf["steps"]["assemble"]["run"] == "#bvbrc-assemble"

    def test_only_terminal_step_in_outputs(self):
        """Only annotation (the terminal step) should appear in workflow outputs."""
        cwl = generate_cwl_workflow(self.plan, self.completed)
        wf = cwl["$graph"][-1]

        # Assembly is depended on by annotation, so it's not terminal
        for output_def in wf["outputs"].values():
            assert output_def["outputSource"].startswith("annotate/")

    def test_step_out_lists_outputs(self):
        cwl = generate_cwl_workflow(self.plan, self.completed)
        wf = cwl["$graph"][-1]

        assert "contigs" in wf["steps"]["assemble"]["out"]
        assert "annotated_genome" in wf["steps"]["annotate"]["out"]


class TestFanOut:
    """Test fan-out: one step feeds two downstream steps."""

    def setup_method(self):
        self.assembly = _assembly_step()

        # Two downstream steps both depend on assembly
        self.annotation = _annotation_step(step_id="annotate")
        self.variation = FakeStep(
            step_id="variation",
            service_name="variation_analysis",
            api_name="Variation",
            params={
                "contigs": "output_of:assemble:contigs",
                "output_path": "/user@bvbrc/home/test",
                "output_file": "variation_output",
            },
            output_patterns={"variants": "*.vcf"},
            depends_on=["assemble"],
        )

        self.plan = _make_plan(
            [self.assembly, self.annotation, self.variation],
            name="fan_out_workflow",
        )
        self.completed = {
            "assemble": self.assembly,
            "annotate": self.annotation,
            "variation": self.variation,
        }

    def test_graph_has_four_entries(self):
        """3 tools + 1 workflow."""
        cwl = generate_cwl_workflow(self.plan, self.completed)
        assert len(cwl["$graph"]) == 4

    def test_both_downstream_wired_to_assembly(self):
        cwl = generate_cwl_workflow(self.plan, self.completed)
        wf = cwl["$graph"][-1]

        assert wf["steps"]["annotate"]["in"]["contigs"] == "assemble/contigs"
        assert wf["steps"]["variation"]["in"]["contigs"] == "assemble/contigs"

    def test_two_terminal_steps(self):
        """Both annotation and variation are terminal (not depended on)."""
        cwl = generate_cwl_workflow(self.plan, self.completed)
        wf = cwl["$graph"][-1]

        output_sources = [v["outputSource"] for v in wf["outputs"].values()]
        assert any("annotate/" in s for s in output_sources)
        assert any("variation/" in s for s in output_sources)


class TestFanIn:
    """Test fan-in: two steps feed one downstream step."""

    def setup_method(self):
        self.step_a = FakeStep(
            step_id="step_a",
            service_name="service_a",
            api_name="AppA",
            params={
                "input_data": "some_data",
                "output_path": "/user@bvbrc/home/test",
                "output_file": "a_output",
            },
            output_patterns={"result_a": "*.result_a"},
            depends_on=[],
        )
        self.step_b = FakeStep(
            step_id="step_b",
            service_name="service_b",
            api_name="AppB",
            params={
                "input_data": "other_data",
                "output_path": "/user@bvbrc/home/test",
                "output_file": "b_output",
            },
            output_patterns={"result_b": "*.result_b"},
            depends_on=[],
        )
        self.step_c = FakeStep(
            step_id="step_c",
            service_name="service_c",
            api_name="AppC",
            params={
                "from_a": "output_of:step_a:result_a",
                "from_b": "output_of:step_b:result_b",
                "output_path": "/user@bvbrc/home/test",
                "output_file": "c_output",
            },
            output_patterns={"final": "*.final"},
            depends_on=["step_a", "step_b"],
        )

        self.plan = _make_plan(
            [self.step_a, self.step_b, self.step_c],
            name="fan_in_workflow",
        )
        self.completed = {
            "step_a": self.step_a,
            "step_b": self.step_b,
            "step_c": self.step_c,
        }

    def test_graph_has_four_entries(self):
        cwl = generate_cwl_workflow(self.plan, self.completed)
        assert len(cwl["$graph"]) == 4

    def test_step_c_wired_to_both_upstreams(self):
        cwl = generate_cwl_workflow(self.plan, self.completed)
        wf = cwl["$graph"][-1]

        step_c_in = wf["steps"]["step_c"]["in"]
        assert step_c_in["from_a"] == "step_a/result_a"
        assert step_c_in["from_b"] == "step_b/result_b"

    def test_only_step_c_in_outputs(self):
        """Only step_c is terminal."""
        cwl = generate_cwl_workflow(self.plan, self.completed)
        wf = cwl["$graph"][-1]

        for output_def in wf["outputs"].values():
            assert output_def["outputSource"].startswith("step_c/")

    def test_step_a_and_b_are_independent(self):
        """Steps A and B have no dependency on each other."""
        cwl = generate_cwl_workflow(self.plan, self.completed)
        wf = cwl["$graph"][-1]

        step_a_in = wf["steps"]["step_a"]["in"]
        step_b_in = wf["steps"]["step_b"]["in"]

        # Neither should reference the other step
        for source in step_a_in.values():
            if isinstance(source, str):
                assert "step_b/" not in source
        for source in step_b_in.values():
            if isinstance(source, str):
                assert "step_a/" not in source


# ===================================================================
# 5. Workspace path handling tests
# ===================================================================


class TestWorkspacePathHandling:
    """Test workspace path detection and wrapping in context."""

    def test_output_path_in_tool_is_directory_type(self):
        step = _assembly_step()
        tool = generate_tool_definition(step)

        assert tool["inputs"]["output_path"]["type"] == "Directory"

    def test_output_file_in_tool_is_string_type(self):
        step = _assembly_step()
        tool = generate_tool_definition(step)

        assert tool["inputs"]["output_file"]["type"] == "string"

    def test_submission_output_path_wrapped(self):
        step = _assembly_step()
        inputs = generate_submission_inputs({"assemble": step})

        assert inputs["output_path"] == {
            "class": "Directory",
            "location": "ws:///user@bvbrc/home/test",
        }

    def test_submission_output_file_is_string(self):
        step = _assembly_step()
        inputs = generate_submission_inputs({"assemble": step})

        assert inputs["output_file"] == "assembly_output"
        assert isinstance(inputs["output_file"], str)

    def test_file_workspace_path_wrapped(self):
        """A param with a workspace path that looks like a file should be File."""
        step = FakeStep(
            step_id="test",
            service_name="test",
            api_name="Test",
            params={
                "contigs": "/user@bvbrc/home/test/contigs.fasta",
                "output_path": "/user@bvbrc/home/test",
                "output_file": "out",
            },
        )
        inputs = generate_submission_inputs({"test": step})
        assert inputs["test_contigs"] == {
            "class": "File",
            "location": "ws:///user@bvbrc/home/test/contigs.fasta",
        }


# ===================================================================
# 6. Submission inputs generation tests
# ===================================================================


class TestGenerateSubmissionInputs:
    """Test submission inputs dict generation."""

    def test_single_step_inputs(self):
        step = _assembly_step()
        inputs = generate_submission_inputs({"assemble": step})

        assert "assemble_recipe" in inputs
        assert inputs["assemble_recipe"] == "auto"
        assert "output_path" in inputs
        assert "output_file" in inputs

    def test_output_of_references_excluded(self):
        """output_of references are inter-step wiring, not submission inputs."""
        annotation = _annotation_step()
        inputs = generate_submission_inputs({"annotate": annotation})

        # contigs is output_of:assemble:contigs — should NOT be in inputs
        assert "annotate_contigs" not in inputs

        # But other params should be present
        assert "annotate_scientific_name" in inputs
        assert "annotate_taxonomy_id" in inputs

    def test_multi_step_inputs(self):
        assembly = _assembly_step()
        annotation = _annotation_step()
        inputs = generate_submission_inputs({
            "assemble": assembly,
            "annotate": annotation,
        })

        # Assembly params
        assert "assemble_recipe" in inputs
        # Annotation params (excluding output_of refs)
        assert "annotate_scientific_name" in inputs
        # Common params (shared)
        assert "output_path" in inputs
        assert "output_file" in inputs

    def test_type_preservation(self):
        step = _assembly_step()
        inputs = generate_submission_inputs({"assemble": step})

        assert isinstance(inputs["assemble_trim"], bool)
        assert isinstance(inputs["assemble_min_contig_len"], int)
        assert isinstance(inputs["assemble_srr_ids"], list)

    def test_list_values_preserved(self):
        step = _assembly_step()
        inputs = generate_submission_inputs({"assemble": step})

        assert inputs["assemble_srr_ids"] == ["SRR12345678"]


# ===================================================================
# 7. Output wiring tests
# ===================================================================


class TestOutputWiring:
    """Test that output_of:step_id:key references produce correct CWL syntax."""

    def test_simple_output_reference(self):
        assembly = _assembly_step()
        annotation = _annotation_step()
        plan = _make_plan([assembly, annotation])
        completed = {"assemble": assembly, "annotate": annotation}

        cwl = generate_cwl_workflow(plan, completed)
        wf = cwl["$graph"][-1]

        # annotation step should wire contigs from assembly
        assert wf["steps"]["annotate"]["in"]["contigs"] == "assemble/contigs"

    def test_chained_references(self):
        """assembly -> annotation -> model: test 3-step chaining."""
        assembly = _assembly_step()
        annotation = _annotation_step()
        model = _model_step()
        plan = _make_plan([assembly, annotation, model])
        completed = {
            "assemble": assembly,
            "annotate": annotation,
            "model": model,
        }

        cwl = generate_cwl_workflow(plan, completed)
        wf = cwl["$graph"][-1]

        # annotation wired to assembly
        assert wf["steps"]["annotate"]["in"]["contigs"] == "assemble/contigs"
        # model wired to annotation
        assert wf["steps"]["model"]["in"]["genome"] == "annotate/annotated_genome"


# ===================================================================
# 8. CWL document structure validation
# ===================================================================


class TestCwlDocumentStructure:
    """Validate the overall CWL document structure."""

    def test_top_level_keys(self):
        step = _assembly_step()
        plan = _make_plan([step])
        cwl = generate_cwl_workflow(plan, {"assemble": step})

        assert "cwlVersion" in cwl
        assert "$namespaces" in cwl
        assert "$graph" in cwl

    def test_cwl_version(self):
        step = _assembly_step()
        plan = _make_plan([step])
        cwl = generate_cwl_workflow(plan, {"assemble": step})

        assert cwl["cwlVersion"] == "v1.2"

    def test_namespace(self):
        step = _assembly_step()
        plan = _make_plan([step])
        cwl = generate_cwl_workflow(plan, {"assemble": step})

        assert cwl["$namespaces"]["gowe"] == "https://github.com/wilke/GoWe#"

    def test_workflow_entry_is_last(self):
        step = _assembly_step()
        plan = _make_plan([step])
        cwl = generate_cwl_workflow(plan, {"assemble": step})

        assert cwl["$graph"][-1]["id"] == "main"
        assert cwl["$graph"][-1]["class"] == "Workflow"

    def test_all_tools_are_command_line_tools(self):
        assembly = _assembly_step()
        annotation = _annotation_step()
        plan = _make_plan([assembly, annotation])
        completed = {"assemble": assembly, "annotate": annotation}
        cwl = generate_cwl_workflow(plan, completed)

        for entry in cwl["$graph"][:-1]:
            assert entry["class"] == "CommandLineTool"

    def test_yaml_serializable(self):
        """Verify the output can be serialized to YAML."""
        import json

        assembly = _assembly_step()
        annotation = _annotation_step()
        plan = _make_plan([assembly, annotation])
        completed = {"assemble": assembly, "annotate": annotation}
        cwl = generate_cwl_workflow(plan, completed)

        # Should not raise
        serialized = json.dumps(cwl)
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_matches_gowe_convention(self):
        """Verify the generated CWL follows GoWe's annotate-and-model pattern."""
        assembly = _assembly_step()
        annotation = _annotation_step()
        plan = _make_plan([assembly, annotation])
        completed = {"assemble": assembly, "annotate": annotation}
        cwl = generate_cwl_workflow(plan, completed)

        # Each tool should have gowe:Execution hint
        for entry in cwl["$graph"][:-1]:
            assert "gowe:Execution" in entry["hints"]
            exec_hint = entry["hints"]["gowe:Execution"]
            assert "bvbrc_app_id" in exec_hint
            assert exec_hint["executor"] == "bvbrc"

        # Workflow should have steps with run references using # prefix
        wf = cwl["$graph"][-1]
        for step_def in wf["steps"].values():
            assert step_def["run"].startswith("#bvbrc-")

        # Workflow outputs should use outputSource
        for output_def in wf["outputs"].values():
            assert "outputSource" in output_def
            assert "/" in output_def["outputSource"]


# ===================================================================
# 9. Edge cases
# ===================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_step_with_no_params(self):
        step = FakeStep(
            step_id="empty",
            service_name="test",
            api_name="TestApp",
            params={},
        )
        tool = generate_tool_definition(step)
        assert tool["inputs"] == {}

    def test_step_with_no_output_patterns(self):
        step = FakeStep(
            step_id="basic",
            service_name="test",
            api_name="TestApp",
            params={"x": "y"},
            output_patterns={},
        )
        tool = generate_tool_definition(step)
        assert "result" in tool["outputs"]

    def test_step_with_none_values(self):
        step = FakeStep(
            step_id="nullable",
            service_name="test",
            api_name="TestApp",
            params={"optional_param": None, "required_param": "value"},
        )
        tool = generate_tool_definition(step)
        assert tool["inputs"]["optional_param"]["type"] == "null"
        assert tool["inputs"]["required_param"]["type"] == "string"

    def test_non_workspace_path_not_wrapped(self):
        """Plain strings that are not workspace paths should not be wrapped."""
        result = python_to_cwl_value("/tmp/local/file.txt", param_name="input")
        assert result == "/tmp/local/file.txt"

    def test_patricbrc_workspace_path(self):
        """Paths with @patricbrc.org should also be detected."""
        assert is_workspace_path("/user@patricbrc.org/home/data") is True
        result = python_to_cwl_value(
            "/user@patricbrc.org/home/data", param_name="some_dir"
        )
        assert result == {
            "class": "Directory",
            "location": "ws:///user@patricbrc.org/home/data",
        }

    def test_user_id_passthrough(self):
        """user_id parameter is accepted without error."""
        step = _assembly_step()
        plan = _make_plan([step])
        completed = {"assemble": step}

        # Should not raise
        cwl = generate_cwl_workflow(plan, completed, user_id="test@bvbrc")
        assert cwl["cwlVersion"] == "v1.2"

    def test_submission_inputs_with_user_id(self):
        step = _assembly_step()
        inputs = generate_submission_inputs({"assemble": step}, user_id="test@bvbrc")
        assert "assemble_recipe" in inputs
