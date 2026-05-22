"""
Workflow manifest generation functions
"""
import os
import json
import sys
import time
import re
import uuid
from typing import Dict, List, Any, Optional
from common.llm_client import LLMClient
from common.json_rpc import JsonRpcCaller
from functions.service_functions import enumerate_apps, get_service_info

# Service catalog built once on first use and reused for all subsequent calls
# Since the service catalog is the same for all users, we build it once and keep it
_service_catalog: Optional[Dict[str, Any]] = None


def load_config_file(filename: str) -> Dict:
    """Load a JSON config file from the config directory."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, 'config', filename)

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompt_file(filename: str) -> str:
    """Load a prompt file from the prompts directory."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(script_dir, 'prompts', filename)

    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def clear_service_catalog():
    """
    Clear the service catalog.

    Useful for testing or forcing a refresh of the catalog.
    """
    global _service_catalog
    _service_catalog = None
    print("Service catalog cleared", file=sys.stderr)


async def initialize_service_catalog(api: JsonRpcCaller, token: str, user_id: str = None) -> bool:
    """
    Initialize the service catalog at startup if a token is available.

    This is optional - the catalog will be built on first use if not initialized here.
    Building it at startup ensures the first request is fast.

    Args:
        api: JsonRpcCaller instance
        token: Authentication token
        user_id: User ID (optional)

    Returns:
        True if catalog was built successfully, False otherwise
    """
    try:
        print("Initializing service catalog at startup...", file=sys.stderr)
        await build_service_catalog(api, token, user_id, force_rebuild=False)
        return True
    except Exception as e:
        print(f"Warning: Could not initialize service catalog at startup: {e}", file=sys.stderr)
        print("  Catalog will be built on first use instead", file=sys.stderr)
        return False


async def build_service_catalog(api: JsonRpcCaller, token: str, user_id: str = None, force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Build a comprehensive service catalog with names, descriptions, and schemas.

    The catalog is built once on first use and reused for all subsequent calls,
    since the service catalog is the same for all users. This avoids expensive
    API calls on every request.

    Args:
        api: JsonRpcCaller instance
        token: Authentication token (required for first build)
        user_id: User ID (optional, not used but kept for API compatibility)
        force_rebuild: Force rebuilding the catalog even if it exists (default: False)

    Returns:
        Dictionary with service information
    """
    global _service_catalog

    # Return cached catalog if it exists and we're not forcing a rebuild
    if not force_rebuild and _service_catalog is not None:
        print("Using pre-built service catalog", file=sys.stderr)
        return _service_catalog

    # Build the catalog (only happens once, or when force_rebuild=True)
    print("Building service catalog (this happens once at startup or first use)...", file=sys.stderr)
    start_time = time.time()
    services_json = await enumerate_apps(api, token, user_id)
    api_time = time.time() - start_time
    print(f"API call took {api_time:.2f} seconds", file=sys.stderr)

    services_data = json.loads(services_json) if isinstance(services_json, str) else services_json

    # Load service name mapping
    service_mapping = load_config_file('service_mapping.json')
    friendly_to_api = service_mapping['friendly_to_api']
    api_to_friendly = {v: k for k, v in friendly_to_api.items()}

    # Extract service list
    if isinstance(services_data, list) and len(services_data) > 0:
        if isinstance(services_data[0], list):
            apps_list = services_data[0]
        else:
            apps_list = services_data
    else:
        apps_list = []

    # Build catalog
    catalog = {
        "services": [],
        "mapping": {
            "friendly_to_api": friendly_to_api,
            "api_to_friendly": api_to_friendly
        }
    }

    for app in apps_list:
        if not isinstance(app, dict) or 'id' not in app:
            continue

        api_name = app['id']
        friendly_name = api_to_friendly.get(api_name)

        if not friendly_name:
            continue

        # Get service description from prompt file
        try:
            description = get_service_info(friendly_name)
        except Exception:
            description = f"Service: {friendly_name}"

        catalog["services"].append({
            "friendly_name": friendly_name,
            "api_name": api_name,
            "description": description
        })

    # Store the catalog for reuse
    _service_catalog = catalog
    print(f"Service catalog built with {len(catalog['services'])} services", file=sys.stderr)

    return catalog


async def select_services_for_workflow(
    user_query: str,
    api: JsonRpcCaller,
    token: str,
    user_id: str,
    llm_client: LLMClient,
    session_id: Optional[str] = None,
    workspace_items: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    STEP 1: Select which services are needed for the workflow.

    Args:
        user_query: User's workflow description
        api: JsonRpcCaller instance
        token: Authentication token
        user_id: User ID for workspace paths
        llm_client: LLM client instance
        session_id: Optional session ID for retrieving session facts
        workspace_items: Optional list of workspace items to include in the prompt

    Returns:
        Dictionary with:
        - services: List of friendly service names
        - reasoning: Why these services were selected
        - error: Error message if selection failed
    """
    try:
        print("STEP 1: Selecting services for workflow...", file=sys.stderr)

        # Load service selection system prompt
        system_prompt = load_prompt_file('workflow_service_selection.txt')

        # Get session facts if session_id is provided
        session_facts_text = ""
        if session_id:
            try:
                from common.session_facts_service import format_session_facts_for_llm
                session_facts_text = format_session_facts_for_llm(session_id, user_id)
                if session_facts_text and session_facts_text != "No session facts available.":
                    print(f"STEP 1: Including session facts from session {session_id}", file=sys.stderr)
                    session_facts_text = f"\n\n{session_facts_text}\n"
                else:
                    session_facts_text = ""
            except Exception as e:
                print(f"Warning: Could not retrieve session facts: {e}", file=sys.stderr)
                session_facts_text = ""

        # Format workspace items if provided
        workspace_str = ""
        if workspace_items and isinstance(workspace_items, list) and len(workspace_items) > 0:
            print(f"STEP 1: Including {len(workspace_items)} workspace items", file=sys.stderr)
            workspace_str = f"\n\nWORKSPACE ITEMS (available for reference):\n{json.dumps(workspace_items, indent=2)}\n\nThese files are in the user's workspace and may be relevant to the query."

        # Build simple user prompt with just the query
        user_prompt = f"""Select the appropriate BV-BRC services for this user request:

USER QUERY: {user_query}
{session_facts_text}{workspace_str}
Return a JSON object with "services" array and "reasoning" string."""

        # Make the LLM call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = llm_client.chat_completion(messages)
        print(f"Service selection response: {response[:200]}...", file=sys.stderr)

        # Parse the response
        response_text = response.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            lines = lines[1:]  # Remove first line
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove last line
            response_text = "\n".join(lines)

        selection = json.loads(response_text)

        if "services" not in selection or not isinstance(selection["services"], list):
            return {
                "error": "Invalid service selection format",
                "raw_response": response
            }

        print(f"Selected services: {selection['services']}", file=sys.stderr)
        print(f"Reasoning: {selection.get('reasoning', 'N/A')}", file=sys.stderr)

        return selection

    except json.JSONDecodeError as e:
        print(f"Failed to parse service selection as JSON: {e}", file=sys.stderr)
        return {
            "error": f"Failed to parse service selection: {str(e)}",
            "raw_response": response
        }
    except Exception as e:
        import traceback
        print(f"Error in select_services_for_workflow: {traceback.format_exc()}", file=sys.stderr)
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def inject_workspace_items_into_workflow(
    workflow_manifest: Dict[str, Any],
    workspace_items: Optional[List[Dict[str, Any]]],
    user_id: str
) -> Dict[str, Any]:
    """
    Inject workspace_items into workflow steps by mapping them to appropriate file input parameters.

    This function modifies the workflow manifest to include workspace file paths in step parameters.
    It intelligently maps workspace items to common file input parameters based on file types and
    service requirements.

    Args:
        workflow_manifest: The generated workflow manifest
        workspace_items: List of workspace items with type, path, name properties
        user_id: User ID for path normalization

    Returns:
        Modified workflow manifest with workspace_items injected into step parameters
    """
    if not workspace_items or not isinstance(workspace_items, list) or len(workspace_items) == 0:
        return workflow_manifest

    print(f"Injecting {len(workspace_items)} workspace items into workflow steps...", file=sys.stderr)

    # Create a copy to avoid modifying the original
    workflow = json.loads(json.dumps(workflow_manifest))

    # Common file input parameter patterns by service type
    # Maps file extensions/types to likely parameter names
    file_type_to_params = {
        # Sequence files
        'fasta': ['input_file', 'genome_file', 'query_file', 'reference_file', 'sequence_file'],
        'fa': ['input_file', 'genome_file', 'query_file', 'reference_file', 'sequence_file'],
        'fna': ['input_file', 'genome_file', 'reference_file'],
        'faa': ['input_file', 'protein_file', 'query_file'],
        'fastq': ['reads_file', 'input_file', 'read_file', 'reads1', 'reads2'],
        'fq': ['reads_file', 'input_file', 'read_file', 'reads1', 'reads2'],
        # Annotation files
        'gff': ['annotation_file', 'input_file', 'gff_file'],
        'gff3': ['annotation_file', 'input_file', 'gff_file'],
        'gtf': ['annotation_file', 'input_file', 'gtf_file'],
        'gb': ['genbank_file', 'input_file'],
        'gbk': ['genbank_file', 'input_file'],
        # Alignment files
        'sam': ['alignment_file', 'input_file', 'sam_file'],
        'bam': ['alignment_file', 'input_file', 'bam_file'],
        'vcf': ['vcf_file', 'input_file', 'variation_file'],
        # Tabular data
        'csv': ['input_file', 'data_file', 'csv_file'],
        'tsv': ['input_file', 'data_file', 'tsv_file'],
        'txt': ['input_file', 'data_file', 'text_file'],
    }

    # Get file extension from path
    def get_file_extension(path: str) -> str:
        if not path:
            return ''
        parts = path.split('.')
        if len(parts) > 1:
            return parts[-1].lower()
        return ''

    # Normalize workspace path (ensure it starts with /user_id/)
    def normalize_workspace_path(path: str) -> str:
        if not path:
            return path
        # If path doesn't start with /, assume it's relative to user home
        if not path.startswith('/'):
            return f"/{user_id}/home/{path}"
        # If path starts with / but not /user_id/, prepend user_id
        if not path.startswith(f"/{user_id}/"):
            # Remove leading / and add user_id
            path = path.lstrip('/')
            return f"/{user_id}/{path}"
        return path

    # Process each step in the workflow
    if 'steps' not in workflow or not isinstance(workflow['steps'], list):
        print("Warning: Workflow has no steps, skipping workspace_items injection", file=sys.stderr)
        return workflow

    # Track which workspace items have been used
    used_items = set()

    for step_idx, step in enumerate(workflow['steps']):
        if not isinstance(step, dict) or 'params' not in step:
            continue

        step_name = step.get('step_name', f'step_{step_idx}')
        params = step.get('params', {})
        app = step.get('app', '')

        print(f"Processing step {step_idx + 1}: {step_name} (app: {app})", file=sys.stderr)

        # Find matching workspace items for this step
        for item_idx, item in enumerate(workspace_items):
            if item_idx in used_items:
                continue

            item_path = item.get('path', '')
            item_name = item.get('name', '')
            item_type = item.get('type', '')

            if not item_path:
                continue

            # Normalize the path
            normalized_path = normalize_workspace_path(item_path)
            file_ext = get_file_extension(item_path)

            # Get potential parameter names for this file type
            potential_params = file_type_to_params.get(file_ext, ['input_file'])

            # Try to find a matching parameter in the step
            # Priority: prefer parameters that match the file type, but override any existing values
            matched_param = None
            for param_name in potential_params:
                # Always override if parameter exists (workspace_items take precedence)
                if param_name in params:
                    matched_param = param_name
                    break
                else:
                    # Parameter doesn't exist, we can add it
                    matched_param = param_name
                    break

            # If we found a match, inject the workspace item
            if matched_param:
                params[matched_param] = normalized_path
                used_items.add(item_idx)
                print(f"  -> Injected workspace item '{item_name}' into parameter '{matched_param}': {normalized_path}", file=sys.stderr)
                break  # Use one workspace item per step for now

        # If this step didn't get a workspace item but has file parameters that need filling,
        # try to inject the first unused workspace item as a generic input_file
        step_has_file_params = any(
            param in params for param in
            ['input_file', 'reads_file', 'genome_file', 'query_file', 'reference_file', 'annotation_file']
        )

        # Check if this step's input_file (or similar) is empty and we have unused workspace items
        needs_file_input = (
            step_has_file_params and
            ('input_file' not in params or not params.get('input_file') or
             params.get('input_file') in ['', None, '${params.input_file}', 'TBD'])
        )

        if needs_file_input:
            # Find first unused workspace item
            for item_idx, item in enumerate(workspace_items):
                if item_idx not in used_items:
                    item_path = item.get('path', '')
                    if item_path:
                        normalized_path = normalize_workspace_path(item_path)
                        # Use input_file as fallback
                        params['input_file'] = normalized_path
                        used_items.add(item_idx)
                        print(f"  -> Injected workspace item '{item.get('name', '')}' as fallback input_file: {normalized_path}", file=sys.stderr)
                        break

    print(f"Injected {len(used_items)} workspace items into workflow", file=sys.stderr)
    return workflow


async def generate_workflow_with_services(
    user_query: str,
    selected_services: List[str],
    api: JsonRpcCaller,
    token: str,
    user_id: str,
    llm_client: LLMClient,
    validation_error: Optional[str] = None,
    previous_workflow: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    workspace_items: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    STEP 2: Generate workflow manifest with detailed parameters for selected services.

    Args:
        user_query: User's workflow description
        selected_services: List of friendly service names (from step 1)
        api: JsonRpcCaller instance
        token: Authentication token
        user_id: User ID for workspace paths
        llm_client: LLM client instance
        validation_error: Optional validation error message from previous attempt
        previous_workflow: Optional previous workflow JSON for fixing errors
        session_id: Optional session ID for retrieving session facts
        workspace_items: Optional list of workspace items to include in the prompt

    Returns:
        Dict containing workflow_json and prompt_payload, or error details
    """
    try:
        print(f"STEP 2: Generating workflow for services: {selected_services}", file=sys.stderr)

        # Get service catalog
        catalog = await build_service_catalog(api, token, user_id, force_rebuild=False)

        # Load configuration files
        output_patterns = load_config_file('service_outputs.json')
        required_params_config = load_config_file('service_required_params.json')

        # Add job_output_path field to every service output
        for service_name, service_outputs in output_patterns.items():
            if isinstance(service_outputs, dict):
                service_outputs['job_output_path'] = "${params.output_path}/.${params.output_file}"

        # Get DETAILED specifications for ONLY the selected services
        selected_service_specs = []
        for service in catalog['services']:
            if service['friendly_name'] in selected_services:
                selected_service_specs.append({
                    "friendly_name": service['friendly_name'],
                    "api_name": service['api_name'],
                    "description": service['description']
                })

        # Get output patterns for ONLY the selected services
        selected_output_patterns = {}
        friendly_to_api = catalog.get('mapping', {}).get('friendly_to_api', {})
        for service_name in selected_services:
            api_name = friendly_to_api.get(service_name, service_name)
            if api_name in output_patterns:
                selected_output_patterns[service_name] = output_patterns[api_name]

        # Get required parameter rules for ONLY the selected services
        selected_required_params = {}
        for service_name in selected_services:
            api_name = friendly_to_api.get(service_name, service_name)
            required_params = (
                required_params_config.get(service_name)
                or required_params_config.get(api_name)
            )
            if required_params:
                selected_required_params[service_name] = required_params

        # Get session facts if session_id is provided
        session_facts_text = ""
        if session_id:
            try:
                from common.session_facts_service import format_session_facts_for_llm
                session_facts_text = format_session_facts_for_llm(session_id, user_id)
                if session_facts_text and session_facts_text != "No session facts available.":
                    print(f"STEP 2: Including session facts from session {session_id}", file=sys.stderr)
                    session_facts_text = f"\n\n{session_facts_text}\n"
                else:
                    session_facts_text = ""
            except Exception as e:
                print(f"Warning: Could not retrieve session facts: {e}", file=sys.stderr)
                session_facts_text = ""

        # Format workspace items if provided
        workspace_str = ""
        if workspace_items and isinstance(workspace_items, list) and len(workspace_items) > 0:
            print(f"STEP 2: Including {len(workspace_items)} workspace items", file=sys.stderr)
            workspace_str = f"\n\nWORKSPACE ITEMS (available for reference):\n{json.dumps(workspace_items, indent=2)}\n\nThese files are in the user's workspace and may be relevant to the query."

        # Load parameter generation system prompt
        system_prompt = load_prompt_file('workflow_parameter_generation.txt').replace("__USER_ID__", user_id)

        # Replace __USER_ID__ in service specs (descriptions loaded via get_service_info from prompts/services/*.txt)
        for spec in selected_service_specs:
            spec["description"] = spec["description"].replace("__USER_ID__", user_id)

        # Build focused user prompt with ONLY selected service details
        session_context = ""
        if session_facts_text:
            session_context = f"\n\nSESSION CONTEXT:\nThe following session facts provide contextual information from previous user interactions. Use this knowledge base to enhance your understanding and populate workflow parameters appropriately.\n{session_facts_text}"

        user_prompt = f"""Generate a complete workflow manifest for the following user request using THESE SPECIFIC SERVICES:

USER QUERY: {user_query}
{session_context}{workspace_str}
SELECTED SERVICES: {json.dumps(selected_services)}

DETAILED SERVICE SPECIFICATIONS (read these carefully for exact parameter names and requirements):
{json.dumps(selected_service_specs, indent=2)}

REQUIRED PARAMETER RULES (source of truth; do not omit any required params):
{json.dumps(selected_required_params, indent=2)}

SERVICE OUTPUT PATTERNS for selected services:
{json.dumps(selected_output_patterns, indent=2)}

Generate a complete workflow manifest with ALL required parameters for each service. Return ONLY the JSON manifest."""

        if validation_error:
            user_prompt += f"""

VALIDATION ERRORS FROM WORKFLOW ENGINE (fix these):
{validation_error}
"""
        if previous_workflow:
            user_prompt += f"""

PREVIOUS WORKFLOW (fix only what is needed to address validation errors):
{json.dumps(previous_workflow, indent=2)}
"""

        prompt_payload = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }
        if os.environ.get("BVBRC_LOG_LLM_PROMPTS") == "1":
            print(
                f"LLM parameter generation prompt payload:\n{json.dumps(prompt_payload, indent=2)}",
                file=sys.stderr
            )

        # Make the LLM call
        print("Generating workflow parameters...", file=sys.stderr)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        llm_start = time.time()
        response = llm_client.chat_completion(messages)
        llm_time = time.time() - llm_start
        print(f"Parameter generation completed in {llm_time:.2f} seconds", file=sys.stderr)

        # Parse the response
        response_text = response.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)

        # Parse JSON
        workflow_manifest = json.loads(response_text)

        # Inject workspace_items into workflow steps if provided
        if workspace_items:
            workflow_manifest = inject_workspace_items_into_workflow(
                workflow_manifest,
                workspace_items,
                user_id
            )

        # Ensure base_context and workspace_output_folder defaults if missing
        if 'base_context' not in workflow_manifest:
            workflow_manifest['base_context'] = {}
        if 'workspace_output_folder' not in workflow_manifest.get('base_context', {}):
            workflow_manifest['base_context']['workspace_output_folder'] = f"/{user_id}/home/CopilotWorkflows"
        if 'base_url' not in workflow_manifest.get('base_context', {}):
            workflow_manifest['base_context']['base_url'] = "https://www.bv-brc.org"
        # Remove workspace_root if it exists
        if 'workspace_root' in workflow_manifest.get('base_context', {}):
            del workflow_manifest['base_context']['workspace_root']

        return {
            "workflow_json": workflow_manifest,
            "prompt_payload": prompt_payload
        }

    except json.JSONDecodeError as e:
        print(f"Failed to parse workflow as JSON: {e}", file=sys.stderr)
        return {
            "error": f"Failed to parse workflow: {str(e)}",
            "raw_response": response,
            "hint": "The LLM response was not valid JSON",
            "prompt_payload": prompt_payload if 'prompt_payload' in locals() else None
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in generate_workflow_with_services: {error_trace}", file=sys.stderr)
        return {
            "error": str(e),
            "traceback": error_trace,
            "prompt_payload": prompt_payload if 'prompt_payload' in locals() else None
        }


async def generate_workflow_manifest_internal(
    user_query: str,
    api: JsonRpcCaller,
    token: str,
    user_id: str,
    llm_client: LLMClient,
    validation_error: Optional[str] = None,
    previous_workflow: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    workspace_items: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Generate a workflow manifest using TWO-STEP LLM-based planning.

    Step 1: Select appropriate services
    Step 2: Generate detailed parameters for selected services

    Args:
        user_query: User's workflow description
        api: JsonRpcCaller instance
        token: Authentication token
        user_id: User ID for workspace paths
        llm_client: LLM client instance
        validation_error: Optional validation error message from previous attempt
        previous_workflow: Optional previous workflow JSON for fixing errors
        session_id: Optional session ID for retrieving session facts
        workspace_items: Optional list of workspace items to include in prompts

    Returns:
        Dict containing workflow_json and prompt_payload, or error details
    """
    try:
        # STEP 1: Select services
        selection_result = await select_services_for_workflow(
            user_query=user_query,
            api=api,
            token=token,
            user_id=user_id,
            llm_client=llm_client,
            session_id=session_id,
            workspace_items=workspace_items
        )

        # Check if selection failed
        if "error" in selection_result:
            return json.dumps({
                "error": "Service selection failed",
                "details": selection_result.get("error"),
                "hint": "Failed to determine which services are needed"
            }, indent=2)

        selected_services = selection_result.get("services", [])
        if not selected_services:
            return json.dumps({
                "error": "No services selected",
                "hint": "Could not identify appropriate services for the query"
            }, indent=2)

        # STEP 2: Generate workflow with detailed parameters
        workflow_result = await generate_workflow_with_services(
            user_query=user_query,
            selected_services=selected_services,
            api=api,
            token=token,
            user_id=user_id,
            llm_client=llm_client,
            validation_error=validation_error,
            previous_workflow=previous_workflow,
            session_id=session_id,
            workspace_items=workspace_items
        )

        return workflow_result

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in generate_workflow_manifest_internal: {error_trace}", file=sys.stderr)
        return {
            "error": str(e),
            "traceback": error_trace
        }


def validate_workflow_structure(workflow_json: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Perform basic structural validation of workflow JSON before submission.

    This only checks for basic required fields. Complex validation (circular dependencies,
    DAG structure, parameter validation, etc.) is handled by the workflow engine itself.

    Args:
        workflow_json: The workflow manifest to validate

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if basic structure is valid
        - (False, error_message) if basic structure is invalid
    """
    try:
        # Check required top-level fields
        required_fields = ['workflow_name', 'steps', 'base_context']
        for field in required_fields:
            if field not in workflow_json:
                return False, f"Missing required field: {field}"

        # Check steps is a non-empty list
        steps = workflow_json.get('steps', [])
        if not isinstance(steps, list):
            return False, "steps must be a list"
        if len(steps) == 0:
            return False, "Workflow must contain at least one step"

        # Basic step structure check
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                return False, f"Step {i} is not a dictionary"

            # Check required step fields
            step_required = ['step_name', 'app', 'params']
            for field in step_required:
                if field not in step:
                    return False, f"Step {i} missing required field: {field}"

        # All basic checks passed - let workflow engine handle complex validation
        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def build_workflow_description(workflow_json: Dict[str, Any]) -> str:
    """
    Build a deterministic plain-text description from workflow steps.

    This summary is intentionally minimal so it can be regenerated easily
    after users edit planned steps.
    """
    steps = workflow_json.get('steps', [])
    if not isinstance(steps, list) or len(steps) == 0:
        return "Planned workflow with no steps."

    step_labels: List[str] = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue

        raw_label = step.get('step_name') or step.get('app') or f"step {idx + 1}"
        if not isinstance(raw_label, str):
            raw_label = str(raw_label)

        label = re.sub(r'[_\-]+', ' ', raw_label).strip()
        label = re.sub(r'\s+', ' ', label)
        step_labels.append(label)

    if not step_labels:
        return "Planned workflow with no steps."

    step_count = len(step_labels)
    if step_count == 1:
        return f"Planned workflow with 1 step: {step_labels[0]}."

    preview_count = min(3, step_count)
    listed_steps = ", ".join(step_labels[:preview_count])
    if step_count > preview_count:
        return f"Planned workflow with {step_count} steps: {listed_steps}, plus {step_count - preview_count} more."

    return f"Planned workflow with {step_count} steps: {listed_steps}."


def apply_parameter_defaults(workflow_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply default values for missing or invalid parameters in workflow steps.
    
    This function fills in sensible defaults for parameters that are missing or invalid
    according to the service_required_params.json configuration. This allows workflows
    to be planned without strict validation failures.
    
    Args:
        workflow_json: The workflow manifest to apply defaults to
        
    Returns:
        Modified workflow manifest with defaults applied
    """
    # Load required params configuration
    try:
        required_params_config = load_config_file('service_required_params.json')
    except Exception as e:
        print(f"Warning: Could not load service_required_params.json: {e}", file=sys.stderr)
        return workflow_json
    
    # Load service mapping to convert friendly names to API names
    try:
        service_mapping = load_config_file('service_mapping.json')
        friendly_to_api = service_mapping.get('friendly_to_api', {})
        api_to_friendly = {v: k for k, v in friendly_to_api.items()}
    except Exception as e:
        print(f"Warning: Could not load service_mapping.json: {e}", file=sys.stderr)
        friendly_to_api = {}
        api_to_friendly = {}
    
    # Create a copy to avoid modifying the original
    workflow = json.loads(json.dumps(workflow_json))
    
    # Process each step
    steps = workflow.get('steps', [])
    if not isinstance(steps, list):
        return workflow
    
    for step_idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
            
        app = step.get('app', '')
        params = step.get('params', {})
        
        if not isinstance(params, dict):
            step['params'] = {}
            params = step['params']
        
        # Try to find the service config by app name
        # App could be API name (e.g., "ComprehensiveGenomeAnalysis") or friendly name (e.g., "comprehensive_genome_analysis")
        service_config = None
        service_name = None
        
        # First try as-is
        if app in required_params_config:
            service_config = required_params_config[app]
            service_name = app
        # Try friendly name lookup
        elif app in friendly_to_api:
            api_name = friendly_to_api[app]
            if api_name in required_params_config:
                service_config = required_params_config[api_name]
                service_name = api_name
            elif app in required_params_config:  # Try friendly name directly
                service_config = required_params_config[app]
                service_name = app
        # Try API to friendly lookup
        elif app in api_to_friendly:
            friendly_name = api_to_friendly[app]
            if friendly_name in required_params_config:
                service_config = required_params_config[friendly_name]
                service_name = friendly_name
        
        if not service_config:
            print(f"  Step {step_idx}: No parameter config found for app '{app}', skipping defaults", file=sys.stderr)
            continue
        
        print(f"  Step {step_idx}: Applying defaults for {app} (config: {service_name})", file=sys.stderr)
        
        # Get defaults from config
        defaults = service_config.get('defaults', {})
        
        # Apply defaults for missing or invalid parameters
        for param_name, default_value in defaults.items():
            current_value = params.get(param_name)
            
            # Check if parameter is missing, None, empty string, or placeholder
            needs_default = (
                current_value is None or
                current_value == '' or
                (isinstance(current_value, str) and current_value in ['TBD', '${params.' + param_name + '}'])
            )
            
            if needs_default:
                params[param_name] = default_value
                print(f"    Applied default for '{param_name}': {default_value}", file=sys.stderr)
    
    return workflow


def prepare_workflow_for_engine_validation(workflow_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove engine-assigned or execution metadata fields before engine validation/submission.

    The workflow engine assigns workflow_id and step_id values at submission time.
    Planning/validation payloads should exclude these fields.
    """
    workflow_for_engine = json.loads(json.dumps(workflow_json))

    # Top-level execution metadata fields
    workflow_for_engine.pop('workflow_id', None)
    workflow_for_engine.pop('status', None)
    workflow_for_engine.pop('created_at', None)
    workflow_for_engine.pop('updated_at', None)
    workflow_for_engine.pop('submitted_at', None)
    workflow_for_engine.pop('started_at', None)
    workflow_for_engine.pop('completed_at', None)
    workflow_for_engine.pop('error_message', None)
    workflow_for_engine.pop('execution_metadata', None)
    workflow_for_engine.pop('log_file_path', None)
    workflow_for_engine.pop('auth_token', None)
    workflow_for_engine.pop('workflow_description', None)

    # Step-level execution metadata
    if 'steps' in workflow_for_engine and isinstance(workflow_for_engine['steps'], list):
        for step in workflow_for_engine['steps']:
            if not isinstance(step, dict):
                continue
            step.pop('step_id', None)
            step.pop('status', None)
            step.pop('task_id', None)
            step.pop('submitted_at', None)
            step.pop('started_at', None)
            step.pop('completed_at', None)
            step.pop('elapsed_time', None)
            step.pop('error_message', None)

    return workflow_for_engine


def normalize_workflow_output_references(workflow_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Repair common workflow_outputs reference issues from generated manifests.

    - ${${steps.x.outputs.y}} -> ${steps.x.outputs.y}
    - ${steps.step1.outputs.y} -> ${steps.<actual_step_name>.outputs.y}
    """
    workflow = json.loads(json.dumps(workflow_json or {}))
    steps = workflow.get("steps", [])
    if not isinstance(steps, list):
        return workflow

    step_names: List[str] = []
    for step in steps:
        if isinstance(step, dict):
            raw_name = step.get("step_name")
            if isinstance(raw_name, str) and raw_name.strip():
                step_names.append(raw_name.strip())

    alias_map: Dict[str, str] = {f"step{i + 1}": name for i, name in enumerate(step_names)}
    step_name_set = set(step_names)

    outputs = workflow.get("workflow_outputs", [])
    if not isinstance(outputs, list):
        return workflow

    nested_ref_pattern = re.compile(r"^\$\{\$\{(.+)\}\}$")
    ref_pattern = re.compile(r"^\$\{steps\.([A-Za-z_][A-Za-z0-9_]*)\.outputs\.([A-Za-z_][A-Za-z0-9_]*)\}$")

    fixed_outputs: List[Any] = []
    for item in outputs:
        if not isinstance(item, str):
            fixed_outputs.append(item)
            continue

        candidate = item.strip()
        nested_match = nested_ref_pattern.match(candidate)
        if nested_match:
            candidate = f"${{{nested_match.group(1)}}}"

        ref_match = ref_pattern.match(candidate)
        if ref_match:
            step_ref, output_name = ref_match.group(1), ref_match.group(2)
            if step_ref in alias_map and step_ref not in step_name_set:
                candidate = f"${{steps.{alias_map[step_ref]}.outputs.{output_name}}}"

        fixed_outputs.append(candidate)

    workflow["workflow_outputs"] = fixed_outputs
    return workflow


def resolve_workflow_variables_locally(workflow_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort local variable resolution for planned workflows.

    This mirrors the most important planner-facing behavior from the workflow engine:
    1) Resolve simple base_context vars (e.g. ${workspace_output_folder})
    2) Resolve ${params.xxx} inside step outputs
    3) Resolve ${steps.step.outputs.name} in workflow_outputs
    """
    workflow = normalize_workflow_output_references(json.loads(json.dumps(workflow_json)))
    var_pattern = re.compile(r"\$\{([^}]+)\}")

    def resolve_simple_vars(value: Any, variables: Dict[str, Any]) -> Any:
        if isinstance(value, str):
            resolved = value
            for match in var_pattern.findall(value):
                # Only resolve simple identifiers (no dotted expressions).
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", match) and match in variables:
                    resolved = resolved.replace(f"${{{match}}}", str(variables[match]))
            return resolved
        if isinstance(value, dict):
            return {k: resolve_simple_vars(v, variables) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve_simple_vars(v, variables) for v in value]
        return value

    # Pass 1: resolve simple base_context references everywhere except inside base_context.
    base_context = workflow.get("base_context", {})
    variables = dict(base_context) if isinstance(base_context, dict) else {}
    for key, value in list(workflow.items()):
        if key == "base_context":
            continue
        workflow[key] = resolve_simple_vars(value, variables)

    # Pass 2: resolve ${params.x} inside each step outputs.
    steps = workflow.get("steps", [])
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            params = step.get("params", {})
            outputs = step.get("outputs")
            if not isinstance(params, dict) or not isinstance(outputs, dict):
                continue

            def resolve_params_refs(value: Any) -> Any:
                if isinstance(value, str):
                    resolved = value
                    for match in var_pattern.findall(value):
                        if match.startswith("params."):
                            param_name = match.split(".", 1)[1]
                            if param_name in params:
                                resolved = resolved.replace(f"${{{match}}}", str(params[param_name]))
                    return resolved
                if isinstance(value, dict):
                    return {k: resolve_params_refs(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [resolve_params_refs(v) for v in value]
                return value

            step["outputs"] = resolve_params_refs(outputs)

    # Pass 3: resolve workflow_outputs references to step outputs.
    if isinstance(steps, list):
        step_outputs_map: Dict[str, Dict[str, Any]] = {}
        for idx, step in enumerate(steps):
            if isinstance(step, dict) and isinstance(step.get("step_name"), str):
                step_outputs_map[step["step_name"]] = step.get("outputs", {}) or {}
                step_outputs_map[f"step{idx + 1}"] = step.get("outputs", {}) or {}

        resolved_workflow_outputs = []
        for item in workflow.get("workflow_outputs", []) or []:
            if not isinstance(item, str):
                resolved_workflow_outputs.append(item)
                continue
            m = re.fullmatch(r"\$\{steps\.([A-Za-z_][A-Za-z0-9_]*)\.outputs\.([A-Za-z_][A-Za-z0-9_]*)\}", item)
            if not m:
                resolved_workflow_outputs.append(item)
                continue
            step_name, output_name = m.group(1), m.group(2)
            resolved_value = step_outputs_map.get(step_name, {}).get(output_name, item)
            resolved_workflow_outputs.append(resolved_value)
        workflow["workflow_outputs"] = resolved_workflow_outputs

    return workflow


async def create_and_execute_workflow_internal(
    user_query: str,
    api: JsonRpcCaller,
    token: str,
    user_id: str,
    llm_client: LLMClient,
    auto_execute: bool = True,
    workflow_engine_config: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    workspace_items: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Generate a workflow from natural language, then optionally persist/submit it.

    This is the complete end-to-end workflow creation and submission function.

    Args:
        user_query: User's workflow description
        api: JsonRpcCaller instance
        token: Authentication token
        user_id: User ID for workspace paths
        llm_client: LLM client instance
        auto_execute: If True, submit workflow to engine after generation (default: True)
        workflow_engine_config: Workflow engine configuration dict (optional)
        session_id: Optional session ID for retrieving session facts
        workspace_items: Optional list of workspace items to include in workflow planning prompts

    Returns:
        Dictionary with workflow_id, status, and workflow_json if successful,
        or error information if any stage fails
    """
    from common.workflow_engine_client import WorkflowEngineClient, WorkflowEngineError

    # Stage 1: Generate workflow
    print("Stage 1: Generating workflow manifest...", file=sys.stderr)
    workflow_json = None
    prompt_payload = None
    previous_workflow = None
    validation_error = None
    attempt = 0
    max_attempts = 2
    while attempt < max_attempts:
        try:
            workflow_result = await generate_workflow_manifest_internal(
                user_query=user_query,
                api=api,
                token=token,
                user_id=user_id,
                llm_client=llm_client,
                validation_error=validation_error,
                previous_workflow=previous_workflow,
                session_id=session_id,
                workspace_items=workspace_items
            )

            # Check if generation returned an error
            if "error" in workflow_result:
                return {
                    "error": workflow_result.get("error"),
                    "errorType": "GENERATION_FAILED",
                    "stage": "generation",
                    "hint": workflow_result.get("hint", "Failed to generate workflow"),
                    "prompt_payload": workflow_result.get("prompt_payload"),
                    "source": "bvbrc-service"
                }

            workflow_json = workflow_result.get("workflow_json")
            prompt_payload = workflow_result.get("prompt_payload")

            print('********* workflow_json *********', file=sys.stderr)
            print(json.dumps(workflow_json, indent=2), file=sys.stderr)

            if not workflow_json:
                return {
                    "error": "Failed to generate workflow JSON",
                    "errorType": "GENERATION_FAILED",
                    "stage": "generation",
                    "prompt_payload": prompt_payload,
                    "source": "bvbrc-service"
                }

            print("Stage 1: Workflow generated successfully", file=sys.stderr)

        except Exception as e:
            import traceback
            return {
                "error": str(e),
                "errorType": "GENERATION_FAILED",
                "stage": "generation",
                "traceback": traceback.format_exc(),
                "source": "bvbrc-service"
            }

        # Stage 2: Planning preparation (validation is deferred to validation/submit endpoints).
        print("Stage 2: Preparing workflow for planning/submission...", file=sys.stderr)

        # If auto_execute is False, return a planned workflow JSON without validation.
        # Apply parameter defaults before workflow engine planning operations.
        if not auto_execute:
            print("Stage 2b: Applying parameter defaults...", file=sys.stderr)
            workflow_json = apply_parameter_defaults(workflow_json)
            workflow_json = normalize_workflow_output_references(workflow_json)

            planned_workflow_id = None
            planned_workflow_status = "planned"
            planning_warnings: List[Dict[str, Any]] = []

            if workflow_engine_config and workflow_engine_config.get('enabled', False):
                try:
                    engine_url = workflow_engine_config.get('api_url', 'http://localhost:8000/api/v1')
                    engine_timeout = workflow_engine_config.get('timeout', 30)
                    client = WorkflowEngineClient(base_url=engine_url, timeout=engine_timeout)

                    # Health checks can be stricter than planning endpoints; do not block planning solely on health.
                    is_healthy = await client.health_check()
                    if not is_healthy:
                        print("Workflow engine health check failed; attempting planning anyway", file=sys.stderr)

                    print("Stage 2b: Persisting workflow plan in workflow engine...", file=sys.stderr)
                    workflow_for_planning = prepare_workflow_for_engine_validation(workflow_json)
                    plan_result = await client.plan_workflow(workflow_for_planning, token)
                    planned_workflow_id = (
                        plan_result.get('workflow_id')
                        or plan_result.get('id')
                        or (plan_result.get('workflow') or {}).get('workflow_id')
                    )
                    planned_workflow_status = plan_result.get('status', 'planned')

                    # Fallback for engines that expose registration instead of plan semantics.
                    if not planned_workflow_id:
                        print("Stage 2b: plan_workflow returned no workflow_id; trying register_workflow fallback", file=sys.stderr)
                        register_result = await client.register_workflow(workflow_for_planning, token)
                        planned_workflow_id = (
                            register_result.get('workflow_id')
                            or register_result.get('id')
                            or (register_result.get('workflow') or {}).get('workflow_id')
                        )
                        planned_workflow_status = register_result.get('status', planned_workflow_status)

                    print(f"Stage 2b: Workflow planned and persisted: {planned_workflow_id}", file=sys.stderr)
                except WorkflowEngineError as e:
                    planning_warnings.append({
                        "type": e.error_type,
                        "status_code": e.status_code,
                        "message": str(e),
                        "stage": "planning",
                        "hint": "Workflow plan could not be persisted in workflow engine; using local workflow_id"
                    })
                except Exception as e:
                    planning_warnings.append({
                        "type": "PLANNING_FAILED",
                        "message": str(e),
                        "stage": "planning",
                        "hint": "Unexpected error while persisting workflow plan; using local workflow_id"
                    })

            # Planning should never block ID assignment except malformed generation payloads.
            if not planned_workflow_id:
                planned_workflow_id = f"wf_local_{uuid.uuid4().hex[:12]}"
                planned_workflow_status = "planned"
                planning_warnings.append({
                    "type": "PLANNING_NOT_PERSISTED",
                    "message": "Workflow ID assigned locally because workflow engine did not return workflow_id",
                    "stage": "planning",
                    "hint": "Re-plan when engine validation issues are resolved to get an engine-registered ID"
                })

            # Resolve variables and return workflow JSON.
            print("auto_execute=False, returning persisted plan", file=sys.stderr)
            locally_resolved_workflow = resolve_workflow_variables_locally(workflow_json)
            locally_resolved_workflow['workflow_id'] = planned_workflow_id
            locally_resolved_workflow['status'] = planned_workflow_status
            workflow_description = build_workflow_description(locally_resolved_workflow)

            response = {
                "workflow_id": planned_workflow_id,
                "status": planned_workflow_status,
                "workflow_json": locally_resolved_workflow,
                "workflow_description": workflow_description,
                "warnings": planning_warnings,
                "message": "Workflow plan generated and saved (not submitted for execution)",
                "prompt_payload": prompt_payload,
                "source": "bvbrc-service"
            }
            return response

        # Check if workflow engine is enabled
        if not workflow_engine_config or not workflow_engine_config.get('enabled', False):
            print("Workflow engine is disabled, returning workflow JSON only", file=sys.stderr)
            return {
                "workflow_json": workflow_json,
                "warning": "Workflow engine is disabled in configuration",
                "message": "Workflow generated but not submitted",
                "hint": "Enable workflow_engine in config.json to submit workflows for execution",
                "source": "bvbrc-service"
            }

        # Stage 3: Validate and submit to workflow engine.
        print("Stage 3: Validating and submitting workflow to execution engine...", file=sys.stderr)
        try:
            engine_url = workflow_engine_config.get('api_url', 'http://localhost:8000/api/v1')
            engine_timeout = workflow_engine_config.get('timeout', 30)

            client = WorkflowEngineClient(base_url=engine_url, timeout=engine_timeout)

            # First check if engine is healthy
            is_healthy = await client.health_check()
            if not is_healthy:
                print("Workflow engine health check failed", file=sys.stderr)
                return {
                    "workflow_json": workflow_json,
                    "warning": "Workflow engine is not available",
                    "message": "Workflow generated but not submitted",
                    "hint": f"Ensure workflow engine is running at {engine_url}",
                    "submission_url": f"{engine_url}/workflows/submit",
                    "prompt_payload": prompt_payload,
                    "source": "bvbrc-service"
                }

            # Clean the workflow before submission - remove engine-assigned/runtime fields.
            workflow_for_submission = prepare_workflow_for_engine_validation(
                normalize_workflow_output_references(workflow_json)
            )
            result = await client.submit_workflow(workflow_for_submission, token)
            workflow_id = result.get('workflow_id')

            print(f"Stage 3: Workflow submitted successfully: {result.get('workflow_id')}", file=sys.stderr)

            # Update the workflow_json with the real workflow_id from the engine
            # This ensures the returned workflow_json has the actual ID, not the placeholder
            updated_workflow_json = workflow_json.copy()
            updated_workflow_json['workflow_id'] = result.get('workflow_id', workflow_id)

            # Also update status and timestamps if available
            updated_workflow_json['status'] = result.get('status', 'pending')
            updated_workflow_json['submitted_at'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            print(f"Updated workflow_json with real workflow_id: {result.get('workflow_id')}", file=sys.stderr)

            # Return success response
            return {
                "workflow_id": result.get('workflow_id', workflow_id),
                "status": result.get('status', 'pending'),
                "workflow_json": updated_workflow_json,
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": result.get('message', 'Workflow created and submitted for execution'),
                "status_url": f"{engine_url}/workflows/{result.get('workflow_id')}/status",
                "prompt_payload": prompt_payload,
                "source": "bvbrc-service"
            }

        except WorkflowEngineError as e:
            print(f"Workflow engine error: {e}", file=sys.stderr)
            if e.error_type == "VALIDATION_FAILED" and attempt < max_attempts - 1:
                validation_error = str(e)
                previous_workflow = workflow_json
                attempt += 1
                print(
                    f"Validation failed in workflow engine; regenerating workflow (attempt {attempt + 1}/{max_attempts})",
                    file=sys.stderr
                )
                continue
            return {
                "error": str(e),
                "errorType": e.error_type,
                "stage": "submission",
                "workflow_json": workflow_json,
                "prompt_payload": prompt_payload,
                "hint": "Workflow was generated but could not be submitted. You can submit it manually.",
                "submission_url": f"{workflow_engine_config.get('api_url', 'http://localhost:8000/api/v1')}/workflows/submit",
                "source": "bvbrc-service"
            }
        except Exception as e:
            import traceback
            print(f"Unexpected error during submission: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return {
                "error": str(e),
                "errorType": "SUBMISSION_FAILED",
                "stage": "submission",
                "workflow_json": workflow_json,
                "prompt_payload": prompt_payload,
                "hint": "Workflow was generated but submission failed unexpectedly",
                "traceback": traceback.format_exc(),
                "source": "bvbrc-service"
            }

