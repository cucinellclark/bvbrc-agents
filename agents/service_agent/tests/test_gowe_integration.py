"""Integration tests for M3: GoWe pipeline rewiring.

Tests verify that the service agent pipeline uses GoWe instead of the
legacy workflow engine. All external calls (GoWe API) are mocked.

Tests cover:
1. compose_manifest() produces CWL documents with cwl_document and submission_inputs
2. handle_submit() uses GoWeClient.create_submission()
3. handle_status() uses GoWeClient.get_submission() with three-level hierarchy
4. handle_cancel() uses GoWeClient.cancel_submission()
5. _run_planning_pipeline() registers CWL via GoWeClient.register_workflow()
6. submission.py public functions use GoWe
7. AgentState/AgentResult carry GoWe-specific fields

Run with:
    pytest agents/service_agent/tests/test_gowe_integration.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the agents and config directories are importable.
# The service_agent package expects config/ to be on sys.path for
# llm_config, which is loaded at import time by models.py.
# ---------------------------------------------------------------------------
_AGENTS_DIR = str(Path(__file__).resolve().parent.parent.parent.parent)
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "config")
_MCP_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "mcp_server")
_SHARED_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "shared")

for _dir in (_AGENTS_DIR, _CONFIG_DIR, _MCP_DIR, _SHARED_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from service_agent.models import (
    AgentConfig,
    AgentResult,
    AgentState,
    SubmissionResult,
    ValidatedStep,
    WorkflowPlan,
    StepPlan,
)
from service_agent.phases.compose import compose_manifest


# ---------------------------------------------------------------------------
# Fixtures — reusable test data
# ---------------------------------------------------------------------------

def _make_config(**overrides: Any) -> AgentConfig:
    """Create an AgentConfig with test defaults."""
    defaults = {
        "bvbrc_auth_token": "un=testuser@bvbrc|tokenid=abc123|sig=fakesig",
        "gowe_url": "https://gowe.test.example.com",
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _make_single_step_state() -> AgentState:
    """Create an AgentState with a single-step workflow plan and completed step."""
    plan = WorkflowPlan(
        workflow_name="test_assembly",
        description="Test genome assembly workflow",
        steps=[
            StepPlan(
                step_id="assemble",
                service_name="genome_assembly",
                intent="Assemble the genome",
                depends_on=[],
                input_sources={},
            )
        ],
    )
    plan.compute_topological_order()

    step = ValidatedStep(
        step_id="assemble",
        service_name="genome_assembly",
        api_name="GenomeAssembly2",
        params={
            "paired_end_libs": [
                "/testuser@bvbrc/home/reads_1.fastq",
                "/testuser@bvbrc/home/reads_2.fastq",
            ],
            "recipe": "auto",
            "output_path": "/testuser@bvbrc/home/assembly_output",
            "output_file": "assembly_result",
        },
        output_patterns={"result": "*.fasta"},
        depends_on=[],
    )

    state = AgentState(
        query="assemble my genome",
        workflow_plan=plan,
        completed_steps={"assemble": step},
        current_phase="compose",
        status="in_progress",
    )
    return state


def _make_multi_step_state() -> AgentState:
    """Create an AgentState with a two-step pipeline (assembly -> annotation)."""
    plan = WorkflowPlan(
        workflow_name="assembly_annotation_pipeline",
        description="Assemble then annotate",
        steps=[
            StepPlan(
                step_id="assemble",
                service_name="genome_assembly",
                intent="Assemble the genome",
                depends_on=[],
            ),
            StepPlan(
                step_id="annotate",
                service_name="genome_annotation",
                intent="Annotate the assembled genome",
                depends_on=["assemble"],
            ),
        ],
    )
    plan.compute_topological_order()

    assemble_step = ValidatedStep(
        step_id="assemble",
        service_name="genome_assembly",
        api_name="GenomeAssembly2",
        params={
            "paired_end_libs": ["/testuser@bvbrc/home/reads.fastq"],
            "output_path": "/testuser@bvbrc/home/output",
            "output_file": "assembly",
        },
        output_patterns={"result": "*.fasta"},
        depends_on=[],
    )

    annotate_step = ValidatedStep(
        step_id="annotate",
        service_name="genome_annotation",
        api_name="GenomeAnnotation",
        params={
            "contigs": "output_of:assemble:result",
            "scientific_name": "Escherichia coli",
            "taxonomy_id": 562,
            "output_path": "/testuser@bvbrc/home/output",
            "output_file": "annotation",
        },
        output_patterns={"genome": "*.genome"},
        depends_on=["assemble"],
    )

    state = AgentState(
        query="assemble and annotate my genome",
        workflow_plan=plan,
        completed_steps={
            "assemble": assemble_step,
            "annotate": annotate_step,
        },
        current_phase="compose",
        status="in_progress",
    )
    return state


# ===================================================================
# Test 1: compose_manifest produces CWL
# ===================================================================

class TestComposeManifest:
    """Tests for compose_manifest() producing CWL documents."""

    def test_single_step_produces_cwl_document(self):
        """compose_manifest returns dict with cwl_document key."""
        state = _make_single_step_state()
        config = _make_config()

        result = compose_manifest(state, config)

        assert "error" not in result, f"Compose failed: {result.get('error')}"
        assert "cwl_document" in result
        assert "submission_inputs" in result
        assert "workflow_name" in result
        assert result["workflow_name"] == "test_assembly"
        assert result["step_count"] == 1

    def test_cwl_document_has_correct_structure(self):
        """CWL document has cwlVersion, $namespaces, and $graph."""
        state = _make_single_step_state()
        config = _make_config()

        result = compose_manifest(state, config)
        cwl_doc = result["cwl_document"]

        assert cwl_doc["cwlVersion"] == "v1.2"
        assert "$namespaces" in cwl_doc
        assert "$graph" in cwl_doc
        assert isinstance(cwl_doc["$graph"], list)

    def test_cwl_graph_contains_tool_and_workflow(self):
        """$graph has at least one CommandLineTool and one Workflow (main)."""
        state = _make_single_step_state()
        config = _make_config()

        result = compose_manifest(state, config)
        graph = result["cwl_document"]["$graph"]

        classes = [entry.get("class") for entry in graph]
        assert "CommandLineTool" in classes
        assert "Workflow" in classes

        # Find the main workflow entry
        workflow = next(e for e in graph if e.get("id") == "main")
        assert workflow["class"] == "Workflow"
        assert "inputs" in workflow
        assert "steps" in workflow
        assert "outputs" in workflow

    def test_submission_inputs_generated(self):
        """submission_inputs contains CWL-typed values for step params."""
        state = _make_single_step_state()
        config = _make_config()

        result = compose_manifest(state, config)
        inputs = result["submission_inputs"]

        assert isinstance(inputs, dict)
        assert len(inputs) > 0

    def test_multi_step_cwl_has_dependencies(self):
        """Multi-step workflows have inter-step wiring in the CWL."""
        state = _make_multi_step_state()
        config = _make_config()

        result = compose_manifest(state, config)

        assert "error" not in result
        cwl_doc = result["cwl_document"]
        graph = cwl_doc["$graph"]

        # Should have two tools and one workflow
        tools = [e for e in graph if e.get("class") == "CommandLineTool"]
        workflows = [e for e in graph if e.get("class") == "Workflow"]
        assert len(tools) == 2
        assert len(workflows) == 1

    def test_user_id_extracted_from_token(self):
        """User ID is extracted from the auth token."""
        state = _make_single_step_state()
        config = _make_config(
            bvbrc_auth_token="un=scientist@bvbrc|tokenid=xyz",
        )

        result = compose_manifest(state, config)
        # Should not error
        assert "error" not in result

    def test_missing_plan_returns_error(self):
        """compose_manifest returns error dict if no workflow plan."""
        state = AgentState(query="test")
        result = compose_manifest(state)
        assert "error" in result
        assert "No workflow plan" in result["error"]

    def test_missing_steps_returns_error(self):
        """compose_manifest returns error dict if no completed steps."""
        state = _make_single_step_state()
        state.completed_steps = {}
        result = compose_manifest(state)
        assert "error" in result
        assert "No completed steps" in result["error"]

    def test_incomplete_steps_returns_error(self):
        """compose_manifest returns error if a planned step is not completed."""
        state = _make_multi_step_state()
        # Remove one completed step
        del state.completed_steps["annotate"]
        result = compose_manifest(state)
        assert "error" in result
        assert "annotate" in result["error"]


# ===================================================================
# Test 2: handle_submit uses GoWe
# ===================================================================

class TestHandleSubmit:
    """Tests for handle_submit() using GoWeClient."""

    @pytest.mark.asyncio
    async def test_submit_creates_submission(self):
        """handle_submit calls GoWeClient.create_submission and sets state."""
        from service_agent.handlers.submit import handle_submit

        config = _make_config()
        state = _make_single_step_state()
        state.submission_inputs = {"input1": "value1"}

        mock_client = AsyncMock()
        mock_client.create_submission.return_value = {
            "id": "sub_test123",
            "state": "PENDING",
            "workflow_id": "wf_abc",
        }

        with patch(
            "common.gowe_client.GoWeClient",
            return_value=mock_client,
        ):
            result = await handle_submit("wf_abc", config, state)

        assert result.status == "completed"
        assert "sub_test123" in result.operation_message
        assert "PENDING" in result.operation_message
        assert result.submission_id == "sub_test123"
        mock_client.create_submission.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_regenerates_inputs_if_missing(self):
        """If state.submission_inputs is None, inputs are regenerated."""
        from service_agent.handlers.submit import handle_submit

        config = _make_config()
        state = _make_single_step_state()
        state.submission_inputs = None  # Force regeneration

        mock_client = AsyncMock()
        mock_client.create_submission.return_value = {
            "id": "sub_regen",
            "state": "PENDING",
        }

        with patch(
            "common.gowe_client.GoWeClient",
            return_value=mock_client,
        ):
            result = await handle_submit("wf_abc", config, state)

        assert result.status == "completed"
        # Verify create_submission was called with regenerated inputs
        call_kwargs = mock_client.create_submission.call_args
        assert call_kwargs.kwargs.get("inputs") is not None or call_kwargs[1].get("inputs") is not None

    @pytest.mark.asyncio
    async def test_submit_requires_auth(self):
        """handle_submit returns error if no auth token."""
        from service_agent.handlers.submit import handle_submit

        config = _make_config(bvbrc_auth_token=None)
        state = AgentState(query="test")

        result = await handle_submit("wf_abc", config, state)

        assert result.status == "error"
        assert "Authentication required" in result.error_message

    @pytest.mark.asyncio
    async def test_submit_handles_gowe_error(self):
        """handle_submit returns error if GoWe raises an exception."""
        from service_agent.handlers.submit import handle_submit

        config = _make_config()
        state = _make_single_step_state()

        mock_client = AsyncMock()
        mock_client.create_submission.side_effect = Exception("GoWe down")

        with patch(
            "common.gowe_client.GoWeClient",
            return_value=mock_client,
        ):
            result = await handle_submit("wf_abc", config, state)

        assert result.status == "error"
        assert "GoWe down" in result.error_message


# ===================================================================
# Test 3: handle_status uses GoWe with three-level hierarchy
# ===================================================================

class TestHandleStatus:
    """Tests for handle_status() using GoWeClient."""

    @pytest.mark.asyncio
    async def test_status_returns_submission_state(self):
        """handle_status queries GoWe and formats the status."""
        from service_agent.handlers.status import handle_status

        config = _make_config()
        state = AgentState(query="check status")

        mock_client = AsyncMock()
        mock_client.get_submission.return_value = {
            "id": "sub_abc123",
            "state": "RUNNING",
            "step_instances": [
                {
                    "step_id": "assemble",
                    "state": "COMPLETED",
                },
                {
                    "step_id": "annotate",
                    "state": "RUNNING",
                    "tasks": [
                        {
                            "id": "t_1",
                            "state": "RUNNING",
                            "executor_type": "bvbrc",
                        }
                    ],
                },
            ],
        }

        with patch(
            "common.gowe_client.GoWeClient",
            return_value=mock_client,
        ):
            result = await handle_status("sub_abc123", config, state)

        assert result.status == "completed"
        msg = result.operation_message
        assert "sub_abc123" in msg
        assert "RUNNING" in msg
        assert "assemble" in msg
        assert "COMPLETED" in msg
        assert "annotate" in msg
        assert "t_1" in msg
        assert "bvbrc" in msg

    @pytest.mark.asyncio
    async def test_status_uses_submission_id_from_state(self):
        """If state has submission_id, use it instead of workflow_id."""
        from service_agent.handlers.status import handle_status

        config = _make_config()
        state = AgentState(query="check status")
        state.submission_id = "sub_from_state"

        mock_client = AsyncMock()
        mock_client.get_submission.return_value = {
            "id": "sub_from_state",
            "state": "COMPLETED",
        }

        with patch(
            "common.gowe_client.GoWeClient",
            return_value=mock_client,
        ):
            result = await handle_status("wf_old_id", config, state)

        # Should query with submission_id, not workflow_id
        mock_client.get_submission.assert_called_once_with(
            "sub_from_state", auth_token=config.bvbrc_auth_token,
        )

    @pytest.mark.asyncio
    async def test_status_handles_error(self):
        """handle_status returns error if GoWe raises."""
        from service_agent.handlers.status import handle_status

        config = _make_config()
        state = AgentState(query="status")

        mock_client = AsyncMock()
        mock_client.get_submission.side_effect = Exception("Connection refused")

        with patch(
            "common.gowe_client.GoWeClient",
            return_value=mock_client,
        ):
            result = await handle_status("sub_abc", config, state)

        assert result.status == "error"
        assert "Connection refused" in result.error_message


# ===================================================================
# Test 4: handle_cancel uses GoWe
# ===================================================================

class TestHandleCancel:
    """Tests for handle_cancel() using GoWeClient."""

    @pytest.mark.asyncio
    async def test_cancel_cancels_submission(self):
        """handle_cancel calls GoWeClient.cancel_submission."""
        from service_agent.handlers.cancel import handle_cancel

        config = _make_config()
        state = AgentState(query="cancel")

        mock_client = AsyncMock()
        mock_client.cancel_submission.return_value = {
            "id": "sub_cancel",
            "state": "CANCELLED",
        }

        with patch(
            "common.gowe_client.GoWeClient",
            return_value=mock_client,
        ):
            result = await handle_cancel("sub_cancel", config, state)

        assert result.status == "completed"
        assert "CANCELLED" in result.operation_message
        mock_client.cancel_submission.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_uses_submission_id_from_state(self):
        """If state has submission_id, use it."""
        from service_agent.handlers.cancel import handle_cancel

        config = _make_config()
        state = AgentState(query="cancel")
        state.submission_id = "sub_state_id"

        mock_client = AsyncMock()
        mock_client.cancel_submission.return_value = {
            "id": "sub_state_id",
            "state": "CANCELLED",
        }

        with patch(
            "common.gowe_client.GoWeClient",
            return_value=mock_client,
        ):
            result = await handle_cancel("wf_old", config, state)

        mock_client.cancel_submission.assert_called_once_with(
            "sub_state_id", auth_token=config.bvbrc_auth_token,
        )

    @pytest.mark.asyncio
    async def test_cancel_requires_auth(self):
        """handle_cancel returns error if no auth token."""
        from service_agent.handlers.cancel import handle_cancel

        config = _make_config(bvbrc_auth_token=None)
        state = AgentState(query="cancel")

        result = await handle_cancel("sub_abc", config, state)

        assert result.status == "error"
        assert "Authentication required" in result.error_message

    @pytest.mark.asyncio
    async def test_cancel_handles_error(self):
        """handle_cancel returns error if GoWe raises."""
        from service_agent.handlers.cancel import handle_cancel

        config = _make_config()
        state = AgentState(query="cancel")

        mock_client = AsyncMock()
        mock_client.cancel_submission.side_effect = Exception("Not found")

        with patch(
            "common.gowe_client.GoWeClient",
            return_value=mock_client,
        ):
            result = await handle_cancel("sub_abc", config, state)

        assert result.status == "error"
        assert "Not found" in result.error_message


# ===================================================================
# Test 5: _run_planning_pipeline registers CWL with GoWe
# ===================================================================

class TestPipelinePersistence:
    """Tests for GoWe registration in _run_planning_pipeline()."""

    @pytest.mark.asyncio
    async def test_pipeline_registers_cwl_with_gowe(self):
        """After compose, the pipeline registers the CWL document with GoWe."""
        from service_agent.agent import _run_planning_pipeline

        config = _make_config()
        state = _make_single_step_state()
        # Pre-set as if phases 1 and 2 are done — skip to compose
        state.current_phase = "compose"

        mock_client = AsyncMock()
        mock_client.register_workflow.return_value = {
            "id": "wf_gowe_123",
            "name": "test_assembly",
            "class": "Workflow",
        }

        with patch(
            "common.gowe_client.GoWeClient",
            return_value=mock_client,
        ):
            result = await _run_planning_pipeline(
                "assemble genome", config, state,
            )

        assert result.workflow_id == "wf_gowe_123"
        assert result.persisted is True
        assert result.cwl_document is not None
        assert result.cwl_document.get("cwlVersion") == "v1.2"
        mock_client.register_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_stores_cwl_and_inputs_in_state(self):
        """After GoWe registration, state has cwl_document and submission_inputs."""
        from service_agent.agent import _run_planning_pipeline

        config = _make_config()
        state = _make_single_step_state()

        mock_client = AsyncMock()
        mock_client.register_workflow.return_value = {"id": "wf_xyz"}

        with patch(
            "common.gowe_client.GoWeClient",
            return_value=mock_client,
        ):
            result = await _run_planning_pipeline(
                "assemble", config, state,
            )

        # Check the manifest contains CWL data
        manifest = result.manifest
        assert manifest is not None
        assert "cwl_document" in manifest
        assert "submission_inputs" in manifest

    @pytest.mark.asyncio
    async def test_pipeline_handles_gowe_failure_gracefully(self):
        """If GoWe registration fails, persisted=False but no error status."""
        from service_agent.agent import _run_planning_pipeline

        config = _make_config()
        state = _make_single_step_state()

        mock_client = AsyncMock()
        mock_client.register_workflow.side_effect = Exception("GoWe timeout")

        with patch(
            "common.gowe_client.GoWeClient",
            return_value=mock_client,
        ):
            result = await _run_planning_pipeline(
                "assemble", config, state,
            )

        # Should complete but with persisted=False
        assert result.status == "completed"
        assert result.persisted is False
        assert result.workflow_id is None


# ===================================================================
# Test 6: submission.py public functions
# ===================================================================

class TestSubmissionModule:
    """Tests for submission.py public functions."""

    def test_extract_workflow_definition_cwl(self):
        """extract_workflow_definition extracts cwl_document from manifest."""
        from service_agent.submission import extract_workflow_definition

        result = AgentResult(
            status="completed",
            manifest={
                "cwl_document": {
                    "cwlVersion": "v1.2",
                    "$graph": [{"id": "main", "class": "Workflow"}],
                },
                "submission_inputs": {"input1": "val"},
            },
        )
        cwl = extract_workflow_definition(result)
        assert cwl["cwlVersion"] == "v1.2"
        assert "$graph" in cwl

    def test_extract_workflow_definition_from_cwl_document_field(self):
        """extract_workflow_definition falls back to agent_result.cwl_document."""
        from service_agent.submission import extract_workflow_definition

        result = AgentResult(
            status="completed",
            manifest={"some": "data"},
            cwl_document={
                "cwlVersion": "v1.2",
                "$graph": [{"id": "main", "class": "Workflow"}],
            },
        )
        cwl = extract_workflow_definition(result)
        assert cwl["cwlVersion"] == "v1.2"

    def test_extract_workflow_definition_raises_on_empty(self):
        """extract_workflow_definition raises ValueError if no manifest."""
        from service_agent.submission import extract_workflow_definition

        result = AgentResult(status="completed", manifest=None)
        with pytest.raises(ValueError):
            extract_workflow_definition(result)

    @pytest.mark.asyncio
    async def test_check_engine_health(self):
        """check_engine_health uses GoWeClient.is_healthy."""
        from service_agent.submission import check_engine_health

        config = _make_config()

        mock_client = AsyncMock()
        mock_client.is_healthy.return_value = True

        with patch(
            "common.gowe_client.GoWeClient",
            return_value=mock_client,
        ):
            healthy = await check_engine_health(config)

        assert healthy is True
        mock_client.is_healthy.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_engine_health_returns_false_on_error(self):
        """check_engine_health returns False if GoWe is unreachable."""
        from service_agent.submission import check_engine_health

        config = _make_config()

        mock_client = AsyncMock()
        mock_client.is_healthy.side_effect = Exception("Connection refused")

        with patch(
            "common.gowe_client.GoWeClient",
            return_value=mock_client,
        ):
            healthy = await check_engine_health(config)

        assert healthy is False


# ===================================================================
# Test 7: Model changes — GoWe fields on AgentState and AgentResult
# ===================================================================

class TestModelChanges:
    """Tests for GoWe-specific fields on models."""

    def test_agent_config_has_gowe_url(self):
        """AgentConfig has gowe_url with default."""
        config = AgentConfig()
        assert config.gowe_url == "https://gowe.software-smithy.org"

    def test_agent_state_has_gowe_fields(self):
        """AgentState has submission_id, cwl_document, submission_inputs."""
        state = AgentState(query="test")
        assert state.submission_id is None
        assert state.cwl_document is None
        assert state.submission_inputs is None

    def test_agent_state_to_result_passes_gowe_fields(self):
        """AgentState.to_result() passes submission_id and cwl_document."""
        state = AgentState(
            query="test",
            status="completed",
            submission_id="sub_abc",
            cwl_document={"cwlVersion": "v1.2"},
        )
        result = state.to_result()
        assert result.submission_id == "sub_abc"
        assert result.cwl_document == {"cwlVersion": "v1.2"}

    def test_agent_result_has_gowe_fields(self):
        """AgentResult has submission_id and cwl_document fields."""
        result = AgentResult(
            status="completed",
            submission_id="sub_xyz",
            cwl_document={"test": True},
        )
        assert result.submission_id == "sub_xyz"
        assert result.cwl_document == {"test": True}

    def test_submission_result_has_submission_id(self):
        """SubmissionResult has submission_id field."""
        sr = SubmissionResult(
            workflow_id="wf_1",
            submission_id="sub_1",
            status="PENDING",
            engine_url="https://gowe.test",
            status_url="https://gowe.test/api/v1/submissions/sub_1",
        )
        assert sr.submission_id == "sub_1"
        assert sr.workflow_id == "wf_1"

    def test_submission_result_backward_compat(self):
        """SubmissionResult still works without submission_id (backward compat)."""
        sr = SubmissionResult(
            workflow_id="wf_1",
            status="pending",
            engine_url="http://legacy",
            status_url="http://legacy/status",
        )
        assert sr.submission_id is None
