"""
Workflow Engine HTTP Client

This module provides a client for interacting with the workflow engine REST API.

Uses httpx (async) for HTTP calls.  Earlier versions used aiohttp, but aiohttp's
async DNS resolver can deadlock when sharing an event loop with the OpenAI SDK
(which uses httpx internally).  Switching to httpx avoids this issue and removes
the aiohttp dependency from the agent's direct-import path.
"""

import httpx
import asyncio
from typing import Dict, Any, Optional
import sys
import json


class WorkflowEngineClient:
    """HTTP client for workflow engine REST API."""

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize workflow engine client.

        Args:
            base_url: Base URL for workflow engine API (e.g., "http://localhost:8000/api/v1")
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = httpx.Timeout(timeout)

    def _sanitize_workflow_payload(self, workflow_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize tool/planner wrapper payloads into a pure WorkflowDefinition payload.

        This is a defensive last line of protection so workflow engine endpoints never
        receive helper fields like workflow_description/prompt_payload even if upstream
        callers accidentally pass wrapper objects.
        """
        payload: Dict[str, Any] = workflow_json or {}

        # Accept either a raw workflow manifest or a wrapped planner response.
        if isinstance(payload, dict) and isinstance(payload.get("workflow_json"), dict):
            payload = payload["workflow_json"]

        if not isinstance(payload, dict):
            return {}

        cleaned = json.loads(json.dumps(payload))

        # Planner/wrapper metadata that does not belong to WorkflowDefinition.
        disallowed_top_level_fields = {
            "workflow_json",
            "workflow_description",
            "message",
            "ready_for_submission",
            "validation",
            "prompt_payload",
            "source",
            "hint",
            "error",
            "errorType",
            "stage",
            "traceback",
            "partial_workflow",
            # Engine/runtime metadata that should not be sent to validate/submit.
            "workflow_id",
            "status",
            "created_at",
            "updated_at",
            "submitted_at",
            "started_at",
            "completed_at",
            "error_message",
            "execution_metadata",
            "log_file_path",
            "auth_token",
        }
        for field in disallowed_top_level_fields:
            cleaned.pop(field, None)

        # Strip step execution metadata when present.
        if isinstance(cleaned.get("steps"), list):
            for step in cleaned["steps"]:
                if not isinstance(step, dict):
                    continue
                for step_field in (
                    "step_id",
                    "status",
                    "task_id",
                    "submitted_at",
                    "started_at",
                    "completed_at",
                    "elapsed_time",
                    "error_message",
                ):
                    step.pop(step_field, None)

        return cleaned

    async def submit_workflow(self, workflow_json: Dict[str, Any], auth_token: str) -> Dict[str, Any]:
        """
        Validate and submit a workflow specification to the workflow engine.

        Args:
            workflow_json: Complete workflow manifest dictionary
            auth_token: BV-BRC authentication token

        Returns:
            Dictionary with:
            - workflow_id: Submitted workflow ID
            - status: Initial status (typically "pending")
            - message: Confirmation message

        Raises:
            WorkflowEngineError: If submission fails
        """
        url = f"{self.base_url}/workflows/submit"
        headers = {
            "Content-Type": "application/json",
            "Authorization": auth_token
        }

        try:
            sanitized_payload = self._sanitize_workflow_payload(workflow_json)
            print(f"Submitting workflow spec to workflow engine: {url}", file=sys.stderr)

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=sanitized_payload, headers=headers)

                if response.status_code in (200, 201):
                    result = response.json()
                    print(f"Workflow submitted successfully: {result.get('workflow_id')}", file=sys.stderr)
                    return result
                elif response.status_code == 400:
                    # Validation error
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('detail', response.text)
                    except Exception:
                        error_msg = response.text
                    raise WorkflowEngineError(
                        f"Workflow validation/submission failed: {error_msg}",
                        error_type="VALIDATION_FAILED",
                        status_code=400
                    )
                elif response.status_code == 500:
                    # Server error
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('detail', response.text)
                    except Exception:
                        error_msg = response.text
                    raise WorkflowEngineError(
                        f"Workflow engine internal error: {error_msg}",
                        error_type="ENGINE_ERROR",
                        status_code=500
                    )
                else:
                    raise WorkflowEngineError(
                        f"Unexpected response from workflow engine: {response.status_code} - {response.text}",
                        error_type="UNKNOWN_ERROR",
                        status_code=response.status_code
                    )

        except WorkflowEngineError:
            # Re-raise our custom errors (must be first to avoid being caught by other handlers)
            raise
        except httpx.ConnectError as e:
            print(f"Failed to connect to workflow engine at {url}: {e}", file=sys.stderr)
            raise WorkflowEngineError(
                f"Cannot connect to workflow engine at {self.base_url}. Is it running?",
                error_type="CONNECTION_FAILED"
            ) from e
        except httpx.TimeoutException as e:
            print(f"Workflow engine request timed out: {e}", file=sys.stderr)
            raise WorkflowEngineError(
                f"Workflow engine request timed out after {self.timeout.connect}s",
                error_type="TIMEOUT"
            ) from e
        except Exception as e:
            print(f"Unexpected error submitting workflow: {e}", file=sys.stderr)
            raise WorkflowEngineError(
                f"Unexpected error: {str(e)}",
                error_type="UNKNOWN_ERROR"
            ) from e

    async def register_workflow(self, workflow_json: Dict[str, Any], auth_token: str) -> Dict[str, Any]:
        """
        Validate and register a workflow without submitting it for execution.

        Args:
            workflow_json: Complete workflow manifest dictionary
            auth_token: BV-BRC authentication token

        Returns:
            Dictionary with workflow_id, status, workflow_name, and step_count
        """
        url = f"{self.base_url}/workflows/register"
        headers = {
            "Content-Type": "application/json",
            "Authorization": auth_token
        }

        try:
            sanitized_payload = self._sanitize_workflow_payload(workflow_json)
            print(f"Registering workflow in workflow engine: {url}", file=sys.stderr)

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=sanitized_payload, headers=headers)

                if response.status_code == 201:
                    result = response.json()
                    print(f"Workflow registered successfully: {result.get('workflow_id')}", file=sys.stderr)
                    return result
                elif response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('detail', response.text)
                    except Exception:
                        error_msg = response.text
                    raise WorkflowEngineError(
                        f"Workflow registration validation failed: {error_msg}",
                        error_type="VALIDATION_FAILED",
                        status_code=400
                    )
                elif response.status_code == 500:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('detail', response.text)
                    except Exception:
                        error_msg = response.text
                    raise WorkflowEngineError(
                        f"Workflow engine internal error during registration: {error_msg}",
                        error_type="ENGINE_ERROR",
                        status_code=500
                    )
                else:
                    raise WorkflowEngineError(
                        f"Unexpected response from workflow registration endpoint: {response.status_code} - {response.text}",
                        error_type="UNKNOWN_ERROR",
                        status_code=response.status_code
                    )
        except WorkflowEngineError:
            raise
        except httpx.ConnectError as e:
            raise WorkflowEngineError(
                f"Cannot connect to workflow engine at {self.base_url}. Is it running?",
                error_type="CONNECTION_FAILED"
            ) from e
        except httpx.TimeoutException as e:
            raise WorkflowEngineError(
                f"Workflow registration request timed out after {self.timeout.connect}s",
                error_type="TIMEOUT"
            ) from e
        except Exception as e:
            raise WorkflowEngineError(
                f"Unexpected error registering workflow: {str(e)}",
                error_type="UNKNOWN_ERROR"
            ) from e

    async def submit_planned_workflow(self, workflow_id: str, auth_token: str) -> Dict[str, Any]:
        """
        Submit a previously planned workflow by ID.

        Args:
            workflow_id: Planned workflow identifier
            auth_token: BV-BRC authentication token

        Returns:
            Dictionary with workflow_id, status, and message
        """
        url = f"{self.base_url}/workflows/{workflow_id}/submit"
        headers = {
            "Content-Type": "application/json",
            "Authorization": auth_token
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers)
                if response.status_code in (200, 201):
                    return response.json()
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('detail', response.text)
                    except Exception:
                        error_msg = response.text
                    raise WorkflowEngineError(
                        f"Planned workflow submission failed: {error_msg}",
                        error_type="VALIDATION_FAILED",
                        status_code=400
                    )
                if response.status_code == 404:
                    raise WorkflowEngineError(
                        f"Workflow {workflow_id} not found",
                        error_type="NOT_FOUND",
                        status_code=404
                    )
                raise WorkflowEngineError(
                    f"Unexpected response from planned workflow submit endpoint: {response.status_code} - {response.text}",
                    error_type="UNKNOWN_ERROR",
                    status_code=response.status_code
                )
        except WorkflowEngineError:
            raise
        except httpx.ConnectError as e:
            raise WorkflowEngineError(
                f"Cannot connect to workflow engine at {self.base_url}. Is it running?",
                error_type="CONNECTION_FAILED"
            ) from e
        except httpx.TimeoutException as e:
            raise WorkflowEngineError(
                f"Planned workflow submission timed out after {self.timeout.connect}s",
                error_type="TIMEOUT"
            ) from e
        except Exception as e:
            raise WorkflowEngineError(
                f"Unexpected error submitting planned workflow: {str(e)}",
                error_type="UNKNOWN_ERROR"
            ) from e

    async def plan_workflow(self, workflow_json: Dict[str, Any], auth_token: str) -> Dict[str, Any]:
        """
        Persist a workflow plan without validation or execution side effects.

        Args:
            workflow_json: Complete workflow manifest dictionary
            auth_token: BV-BRC authentication token

        Returns:
            Dictionary with workflow_id, status, workflow_name, and step_count
        """
        url = f"{self.base_url}/workflows/plan"
        headers = {
            "Content-Type": "application/json",
            "Authorization": auth_token
        }

        try:
            sanitized_payload = self._sanitize_workflow_payload(workflow_json)
            print(f"Planning workflow in workflow engine: {url}", file=sys.stderr)

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=sanitized_payload, headers=headers)

                if response.status_code == 201:
                    result = response.json()
                    print(f"Workflow planned successfully: {result.get('workflow_id')}", file=sys.stderr)
                    return result
                elif response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('detail', response.text)
                    except Exception:
                        error_msg = response.text
                    raise WorkflowEngineError(
                        f"Workflow planning validation failed: {error_msg}",
                        error_type="VALIDATION_FAILED",
                        status_code=400
                    )
                elif response.status_code == 500:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('detail', response.text)
                    except Exception:
                        error_msg = response.text
                    raise WorkflowEngineError(
                        f"Workflow engine internal error during planning: {error_msg}",
                        error_type="ENGINE_ERROR",
                        status_code=500
                    )
                else:
                    raise WorkflowEngineError(
                        f"Unexpected response from workflow planning endpoint: {response.status_code} - {response.text}",
                        error_type="UNKNOWN_ERROR",
                        status_code=response.status_code
                    )
        except WorkflowEngineError:
            raise
        except httpx.ConnectError as e:
            raise WorkflowEngineError(
                f"Cannot connect to workflow engine at {self.base_url}. Is it running?",
                error_type="CONNECTION_FAILED"
            ) from e
        except httpx.TimeoutException as e:
            raise WorkflowEngineError(
                f"Workflow planning request timed out after {self.timeout.connect}s",
                error_type="TIMEOUT"
            ) from e
        except Exception as e:
            raise WorkflowEngineError(
                f"Unexpected error planning workflow: {str(e)}",
                error_type="UNKNOWN_ERROR"
            ) from e

    async def validate_workflow(self, workflow_json: Dict[str, Any], auth_token: str) -> Dict[str, Any]:
        """
        Validate and normalize a workflow in the workflow engine without submitting it.

        Args:
            workflow_json: Complete workflow manifest dictionary
            auth_token: BV-BRC authentication token

        Returns:
            Dictionary with:
            - valid: True if validation succeeded
            - workflow_json: Normalized/validated workflow manifest
            - warnings: Optional validation warnings
            - auto_fixes: Optional list of auto-applied fixes
            - message: Validation status message

        Raises:
            WorkflowEngineError: If validation fails
        """
        url = f"{self.base_url}/workflows/validate"
        headers = {
            "Content-Type": "application/json",
            "Authorization": auth_token
        }

        try:
            sanitized_payload = self._sanitize_workflow_payload(workflow_json)
            print(f"Validating workflow in workflow engine: {url}", file=sys.stderr)

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=sanitized_payload, headers=headers)

                if response.status_code == 200:
                    result = response.json()
                    print("Workflow validated successfully", file=sys.stderr)
                    return result
                elif response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('detail', response.text)
                    except Exception:
                        error_msg = response.text
                    raise WorkflowEngineError(
                        f"Workflow validation failed: {error_msg}",
                        error_type="VALIDATION_FAILED",
                        status_code=400
                    )
                elif response.status_code == 404:
                    raise WorkflowEngineError(
                        "Workflow engine validate endpoint not found",
                        error_type="ENDPOINT_NOT_FOUND",
                        status_code=404
                    )
                elif response.status_code == 500:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('detail', response.text)
                    except Exception:
                        error_msg = response.text
                    raise WorkflowEngineError(
                        f"Workflow engine internal error during validation: {error_msg}",
                        error_type="ENGINE_ERROR",
                        status_code=500
                    )
                else:
                    raise WorkflowEngineError(
                        f"Unexpected response from workflow engine validation: {response.status_code} - {response.text}",
                        error_type="UNKNOWN_ERROR",
                        status_code=response.status_code
                    )

        except WorkflowEngineError:
            raise
        except httpx.ConnectError as e:
            print(f"Failed to connect to workflow engine at {url}: {e}", file=sys.stderr)
            raise WorkflowEngineError(
                f"Cannot connect to workflow engine at {self.base_url}. Is it running?",
                error_type="CONNECTION_FAILED"
            ) from e
        except httpx.TimeoutException as e:
            print(f"Workflow engine validation request timed out: {e}", file=sys.stderr)
            raise WorkflowEngineError(
                f"Workflow engine validation request timed out after {self.timeout.connect}s",
                error_type="TIMEOUT"
            ) from e
        except Exception as e:
            print(f"Unexpected error validating workflow: {e}", file=sys.stderr)
            raise WorkflowEngineError(
                f"Unexpected error: {str(e)}",
                error_type="UNKNOWN_ERROR"
            ) from e

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the status of a workflow.

        Args:
            workflow_id: The workflow ID to query

        Returns:
            Dictionary with workflow status information including:
            - workflow_id
            - workflow_name
            - status
            - created_at
            - updated_at
            - steps (array of step status objects)

        Raises:
            WorkflowEngineError: If query fails
        """
        url = f"{self.base_url}/workflows/{workflow_id}/status"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    raise WorkflowEngineError(
                        f"Workflow {workflow_id} not found",
                        error_type="NOT_FOUND",
                        status_code=404
                    )
                else:
                    raise WorkflowEngineError(
                        f"Failed to get workflow status: {response.status_code} - {response.text}",
                        error_type="QUERY_FAILED",
                        status_code=response.status_code
                    )

        except WorkflowEngineError:
            # Re-raise our custom errors (must be first)
            raise
        except httpx.ConnectError as e:
            raise WorkflowEngineError(
                f"Cannot connect to workflow engine at {self.base_url}",
                error_type="CONNECTION_FAILED"
            ) from e
        except Exception as e:
            raise WorkflowEngineError(
                f"Unexpected error querying workflow status: {str(e)}",
                error_type="UNKNOWN_ERROR"
            ) from e

    async def get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Retrieve complete workflow document by ID.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Full workflow document
        """
        url = f"{self.base_url}/workflows/{workflow_id}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                if response.status_code == 404:
                    raise WorkflowEngineError(
                        f"Workflow {workflow_id} not found",
                        error_type="NOT_FOUND",
                        status_code=404
                    )
                raise WorkflowEngineError(
                    f"Failed to retrieve workflow: {response.status_code} - {response.text}",
                    error_type="QUERY_FAILED",
                    status_code=response.status_code
                )
        except WorkflowEngineError:
            raise
        except httpx.ConnectError as e:
            raise WorkflowEngineError(
                f"Cannot connect to workflow engine at {self.base_url}",
                error_type="CONNECTION_FAILED"
            ) from e
        except Exception as e:
            raise WorkflowEngineError(
                f"Unexpected error retrieving workflow: {str(e)}",
                error_type="UNKNOWN_ERROR"
            ) from e

    async def health_check(self) -> bool:
        """
        Check if workflow engine is available and healthy.

        Returns:
            True if workflow engine is healthy, False otherwise
        """
        url = f"{self.base_url}/health"

        try:
            # Use a shorter timeout for health checks
            quick_timeout = httpx.Timeout(5)
            async with httpx.AsyncClient(timeout=quick_timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    # Check if MongoDB is connected
                    return data.get('mongodb') == 'connected'
                return False
        except Exception as e:
            print(f"Workflow engine health check failed: {e}", file=sys.stderr)
            return False


class WorkflowEngineError(Exception):
    """Custom exception for workflow engine errors."""

    def __init__(self, message: str, error_type: str = "UNKNOWN_ERROR", status_code: Optional[int] = None):
        """
        Initialize workflow engine error.

        Args:
            message: Error message
            error_type: Type of error (e.g., "CONNECTION_FAILED", "VALIDATION_FAILED")
            status_code: HTTP status code if applicable
        """
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
