from common.json_rpc import JsonRpcCaller
from typing import List, Dict, Any, Optional
import asyncio
import uuid
import time
import json
import os
import sys
import httpx

CGA_INPUT_TYPES = {"reads", "contigs", "genbank"}
CGA_RECIPES = {
    "auto",
    "unicycler",
    "canu",
    "spades",
    "meta-spades",
    "plasmid-spades",
    "single-cell",
    "flye",
}
CGA_DOMAINS = {"Bacteria", "Archaea", "Viruses", "auto"}
CGA_CODES = {0, 1, 4, 11, 25}
CGA_PLATFORMS = {"infer", "illumina", "pacbio", "pacbio_hifi", "nanopore"}

CGA_FIELD_ALIASES = {
    "tax_id": "taxonomy_id",
    "taxon_id": "taxonomy_id",
    "output_folder": "output_path",
    "output_name": "output_file",
    "srr_accession": "srr_ids",
    "srr_accessions": "srr_ids",
    "organism_name": "scientific_name",
    "scientificName": "scientific_name",
}
CGA_INPUT_TYPE_ALIASES = {
    "read": "reads",
    "raw_reads": "reads",
    "fastq": "reads",
    "contig": "contigs",
    "assembled_contigs": "contigs",
    "gbk": "genbank",
    "genbank_file": "genbank",
}
CGA_RECIPE_ALIASES = {
    "meta_flye": "flye",
    "meta-flye": "flye",
    "metaflye": "flye",
    "single_cell": "single-cell",
    "meta_spades": "meta-spades",
    "plasmid_spades": "plasmid-spades",
}
CGA_DOMAIN_ALIASES = {
    "bacteria": "Bacteria",
    "bacterial": "Bacteria",
    "archaea": "Archaea",
    "archaeal": "Archaea",
    "virus": "Viruses",
    "viruses": "Viruses",
    "viral": "Viruses",
    "auto": "auto",
}
CGA_PLATFORM_ALIASES = {
    "pacbio-hifi": "pacbio_hifi",
    "hifi": "pacbio_hifi",
}

def _generate_numerical_uuid() -> int:
    """Generate a numerical UUID for JSON RPC call IDs."""
    return int(str(uuid.uuid4().int)[:10])  # Take first 10 digits to ensure it fits in int range

def _set_default_output_paths(user_id: str, app_name: str, output_path: str = None, output_file: str = None):
    """Helper function to set default output_path and output_file values."""
    if output_path is None:
        output_path = '/' + user_id + '/home/CopilotWorkflows'
    if output_file is None:
        output_file = app_name + '_' + time.strftime('%Y%m%d_%H%M%S')
    return output_path, output_file

def _filter_none_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to filter out parameters with None values."""
    return {k: v for k, v in params.items() if v is not None}

def _resolve_output_path(output_path: str, user_id: str) -> str:
    """Helper function to resolve relative paths to absolute paths."""
    if output_path and user_id and not output_path.startswith('/'):
        output_path = f"/{user_id}/home/{output_path}"
    elif output_path and user_id and output_path.startswith('home/'):
        output_path = f"/{user_id}/{output_path}"
    return output_path


def _normalize_and_validate_comprehensive_genome_analysis_params(
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Normalize CGA aliases and reject ambiguous/invalid values early."""
    normalized = dict(params)

    for alias_key, canonical_key in CGA_FIELD_ALIASES.items():
        if alias_key in normalized and canonical_key not in normalized:
            normalized[canonical_key] = normalized.pop(alias_key)

    if "coverage" in normalized and "target_depth" not in normalized:
        normalized["target_depth"] = normalized["coverage"]
    normalized.pop("coverage", None)

    if "expected_genome_size" in normalized and "genome_size" not in normalized:
        units = str(normalized.get("genome_size_units", "M")).strip().upper()
        unit_multipliers = {"K": 1000, "M": 1000000, "G": 1000000000}
        try:
            size_value = int(normalized["expected_genome_size"])
            if units in unit_multipliers:
                normalized["genome_size"] = size_value * unit_multipliers[units]
        except (TypeError, ValueError):
            pass
    normalized.pop("expected_genome_size", None)
    normalized.pop("genome_size_units", None)

    sci_name = normalized.get("scientific_name")
    if not isinstance(sci_name, str) or not sci_name.strip():
        fallback_name = normalized.get("output_file")
        if isinstance(fallback_name, str) and fallback_name.strip():
            normalized["scientific_name"] = fallback_name.strip()
        else:
            raise ValueError(
                "scientific_name must be a non-empty string (or output_file must be set to derive it)."
            )

    input_type = normalized.get("input_type")
    if isinstance(input_type, str):
        candidate = input_type.strip().lower()
        normalized["input_type"] = CGA_INPUT_TYPE_ALIASES.get(candidate, candidate)
    if normalized.get("input_type") not in CGA_INPUT_TYPES:
        raise ValueError(
            f"input_type must be one of {sorted(CGA_INPUT_TYPES)}; got {normalized.get('input_type')!r}."
        )

    recipe = normalized.get("recipe")
    if isinstance(recipe, str):
        candidate = recipe.strip().lower()
        normalized["recipe"] = CGA_RECIPE_ALIASES.get(candidate, candidate)
    if normalized.get("recipe") not in CGA_RECIPES:
        raise ValueError(
            f"recipe must be one of {sorted(CGA_RECIPES)}; got {normalized.get('recipe')!r}."
        )

    domain = normalized.get("domain")
    if isinstance(domain, str):
        candidate = domain.strip().lower()
        normalized["domain"] = CGA_DOMAIN_ALIASES.get(candidate, domain.strip())
    if normalized.get("domain") not in CGA_DOMAINS:
        raise ValueError(
            f"domain must be one of {sorted(CGA_DOMAINS)}; got {normalized.get('domain')!r}."
        )

    try:
        normalized["code"] = int(normalized.get("code"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"code must be an integer in {sorted(CGA_CODES)}.") from exc
    if normalized["code"] not in CGA_CODES:
        raise ValueError(f"code must be one of {sorted(CGA_CODES)}; got {normalized['code']!r}.")

    if "taxonomy_id" in normalized and normalized["taxonomy_id"] is not None:
        try:
            normalized["taxonomy_id"] = int(normalized["taxonomy_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError("taxonomy_id must be an integer.") from exc
        if normalized["taxonomy_id"] <= 0:
            raise ValueError("taxonomy_id must be a positive integer.")

    for lib_key, read_key in (("paired_end_libs", "read1"), ("single_end_libs", "read")):
        libs = normalized.get(lib_key)
        if libs is None:
            continue
        if not isinstance(libs, list) or len(libs) == 0:
            raise ValueError(f"{lib_key} must be a non-empty list when provided.")
        for i, lib in enumerate(libs):
            if not isinstance(lib, dict):
                raise ValueError(f"{lib_key}[{i}] must be an object.")
            read_val = lib.get(read_key)
            if not isinstance(read_val, str) or not read_val.strip():
                raise ValueError(f"{lib_key}[{i}] must include non-empty '{read_key}'.")
            platform = lib.get("platform", "infer")
            if isinstance(platform, str):
                platform = CGA_PLATFORM_ALIASES.get(platform.strip().lower(), platform.strip().lower())
            if platform not in CGA_PLATFORMS:
                raise ValueError(
                    f"{lib_key}[{i}].platform must be one of {sorted(CGA_PLATFORMS)}; got {lib.get('platform')!r}."
                )
            lib["platform"] = platform

    input_type = normalized["input_type"]
    has_reads = any(
        normalized.get(name) not in (None, "", [])
        for name in ("paired_end_libs", "single_end_libs", "srr_ids")
    )
    has_contigs = any(
        normalized.get(name) not in (None, "", [])
        for name in ("contigs", "reference_assembly")
    )
    has_genbank = any(
        normalized.get(name) not in (None, "", [])
        for name in ("genbank_file", "gto")
    )

    if input_type == "reads":
        if not has_reads:
            raise ValueError(
                "When input_type='reads', provide at least one of paired_end_libs, single_end_libs, or srr_ids."
            )
        if has_contigs or has_genbank:
            raise ValueError("When input_type='reads', do not include contigs/genbank input fields.")
    elif input_type == "contigs":
        if not has_contigs:
            raise ValueError("When input_type='contigs', provide contigs or reference_assembly.")
        if has_reads or has_genbank:
            raise ValueError("When input_type='contigs', do not include reads/genbank input fields.")
    else:  # genbank
        if not has_genbank:
            raise ValueError("When input_type='genbank', provide genbank_file or gto.")
        if has_reads or has_contigs:
            raise ValueError("When input_type='genbank', do not include reads/contigs input fields.")

    return normalized

def _build_grid_payload(
    entity_type: str,
    items: list,
    source: str = "bvbrc-service",
    result_type: str = "list_result",
    pagination: Dict[str, Any] = None,
    sort: Dict[str, Any] = None,
    columns: List[Dict[str, Any]] = None,
    selectable: bool = True,
    multi_select: bool = True,
    sortable: bool = True
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "entity_type": entity_type,
        "source": source,
        "result_type": result_type,
        "capabilities": {
            "selectable": selectable,
            "multi_select": multi_select,
            "sortable": sortable
        },
        "pagination": pagination,
        "sort": sort,
        "columns": columns or [],
        "items": items if isinstance(items, list) else []
    }

def get_service_info(service_name: str) -> str:
    """
    Get service information from prompt files.

    Args:
        service_name: Name of the service (e.g., 'genome_assembly', 'blast', 'date')

    Returns:
        String containing the service information/parameters from the prompt file

    Raises:
        FileNotFoundError: If the prompt file for the service doesn't exist
        IOError: If there's an error reading the file
    """
    try:
        # Get the directory where this script is located (functions/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the parent directory (bvbrc-mcp-server/)
        parent_dir = os.path.dirname(script_dir)

        # Construct the path to the prompt file
        prompt_file_path = os.path.join(parent_dir, 'prompts', 'services', f'{service_name}.txt')

        # Check if file exists
        if not os.path.exists(prompt_file_path):
            raise FileNotFoundError(f"Service prompt file not found: {service_name}.txt")

        # Read and return the file contents
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Append output_path instruction to every service prompt
        output_path_note = "\n\nNote: output_path is relative to the user's home directory. Do not start the path with a '/' or with 'home'."
        return content + output_path_note

    except Exception as e:
        raise Exception(f"Error reading service info for '{service_name}': {str(e)}")

async def enumerate_apps(api: JsonRpcCaller, token: str = None, user_id: str = None) -> dict:
    try:
        result = await api.acall("AppService.enumerate_apps", {}, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_date_app(api: JsonRpcCaller, token: str = None, user_id: str = None, output_path: str = None, output_file: str = None) -> str:
    app_name = "Date"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "output_path": output_path,
            "output_file": output_file
        })

        data = ["Date", params, {}]

        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_genome_annotation_app(api: JsonRpcCaller, token: str = None, user_id: str = None, genome_id: str = None, contigs: str = None, scientific_name: str = None, tax_id: str = None, my_label: str = None, reference_genome_id: str = None, taxonomy_id: str = None, code: int = 0, domain: str = "auto", public: bool = False, queue_nowait: bool = False, skip_indexing: bool = False, skip_workspace_output: bool = False, output_path: str = None, output_file: str = None, lowvan_min_contig_length: int = 300, lowvan_max_contig_length: int = 35000, reference_virus_name: str = None, fix_errors: bool = None, fix_frameshifts: bool = None, verbose_level: int = None, workflow: str = None, recipe: str = None, disable_replication: bool = None, analyze_quality: bool = None, assembly_output: str = None, custom_pipeline: Dict = None) -> str:
    app_name = "GenomeAnnotation"
    try:
        if genome_id is None:
            genome_id = ""
        if contigs is None:
            contigs = ""
        if scientific_name is None:
            scientific_name = ""
        if tax_id is None:
            tax_id = ""
        if my_label is None:
            my_label = ""
        if reference_genome_id is None:
            reference_genome_id = ""
        if taxonomy_id is None:
            taxonomy_id = ""
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "genome_id": genome_id,
            "contigs": contigs,
            "scientific_name": scientific_name,
            "tax_id": tax_id,
            "my_label": my_label,
            "reference_genome_id": reference_genome_id,
            "taxonomy_id": taxonomy_id,
            "code": code,
            "domain": domain,
            "public": public,
            "queue_nowait": queue_nowait,
            "skip_indexing": skip_indexing,
            "skip_workspace_output": skip_workspace_output,
            "output_path": output_path,
            "output_file": output_file,
            "lowvan_min_contig_length": lowvan_min_contig_length,
            "lowvan_max_contig_length": lowvan_max_contig_length,
            "reference_virus_name": reference_virus_name,
            "fix_errors": fix_errors,
            "fix_frameshifts": fix_frameshifts,
            "verbose_level": verbose_level,
            "workflow": workflow,
            "recipe": recipe,
            "disable_replication": disable_replication,
            "analyze_quality": analyze_quality,
            "assembly_output": assembly_output,
            "custom_pipeline": custom_pipeline
        })
        data = ["GenomeAnnotation", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def query_tasks(
    api: JsonRpcCaller, 
    token: str = None, 
    user_id: str = None, 
    params: Dict[str, Any] = None,
    fetch_stdout: bool = False,
    fetch_stderr: bool = False
) -> dict:
    """
    Query task details and optionally fetch stdout/stderr content.
    
    Args:
        api: JSON RPC caller instance
        token: Authentication token
        user_id: User ID
        params: Dictionary containing 'task_ids' list
        fetch_stdout: If True, fetch last 100 lines of stdout from task_info endpoint
        fetch_stderr: If True, fetch last 100 lines of stderr from task_info endpoint
    
    Note:
        Only the last 100 lines of stdout/stderr are returned to keep responses manageable.
    """
    try:
        result = await api.acall("AppService.query_tasks", [params['task_ids']], _generate_numerical_uuid(), token)
        
        # If we need to fetch stdout or stderr, process each task
        if (fetch_stdout or fetch_stderr) and isinstance(result, list):
            # Get the base service URL from the api client
            # The service_url is like: https://p3.theseed.org/services/app_service
            base_service_url = api.service_url
            
            async def fetch_log(client, task_data, log_type, url):
                """Fetch a single log file and attach only the last 100 lines to the task."""
                try:
                    headers = {
                        'Authorization': f'Oauth {token}',
                        'X-Requested-With': 'false'
                    }
                    response = await client.get(url, headers=headers)
                    if response.status_code == 200:
                        content = response.text
                        
                        # Get only the last 100 lines
                        lines = content.splitlines()
                        last_100_lines = lines[-100:] if len(lines) > 100 else lines
                        truncated_content = '\n'.join(last_100_lines)
                        
                        task_data[log_type] = truncated_content
                    else:
                        task_data[log_type] = f"Error fetching {log_type}: HTTP {response.status_code}"
                except Exception as e:
                    task_data[log_type] = f"Error fetching {log_type}: {str(e)}"
            
            # Collect all fetch operations
            fetch_operations = []
            async with httpx.AsyncClient() as client:
                for task_wrapper in result:
                    if isinstance(task_wrapper, dict):
                        # The result structure is: [{'task_id': {task_data}}]
                        # Get the task_id (key) and task_data (value)
                        for task_id, task_data in task_wrapper.items():
                            if not isinstance(task_data, dict):
                                continue
                            
                            # Construct stdout URL: {base_url}/task_info/{task_id}/stdout
                            if fetch_stdout:
                                stdout_url = f"{base_service_url}/task_info/{task_id}/stdout"
                                fetch_operations.append(
                                    fetch_log(client, task_data, 'stdout', stdout_url)
                                )
                            
                            # Construct stderr URL: {base_url}/task_info/{task_id}/stderr
                            if fetch_stderr:
                                stderr_url = f"{base_service_url}/task_info/{task_id}/stderr"
                                fetch_operations.append(
                                    fetch_log(client, task_data, 'stderr', stderr_url)
                                )
                
                # Wait for all fetch operations to complete
                if fetch_operations:
                    await asyncio.gather(*fetch_operations)
        
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def list_jobs(
    api: JsonRpcCaller,
    token: str = None,
    user_id: str = None,
    limit: int = 20,
    offset: int = 0,
    sort_by: str = "submit_time",
    sort_dir: str = "desc",
    status: Optional[str] = None,
    service: Optional[str] = None,
    search: Optional[str] = None,
    include_archived: bool = False
) -> dict:
    """
    List user jobs with optional filtering/sorting.

    Uses AppService.enumerate_tasks_filtered(offset, limit, simpleFilter).
    """
    try:
        simple_filter: Dict[str, Any] = {}

        if status:
            simple_filter["status"] = status
        if service:
            simple_filter["app"] = service
        if search:
            simple_filter["search"] = search
        if include_archived:
            # API expects numeric 1, mirroring frontend behavior.
            simple_filter["include_archived"] = 1

        sort_field_map = {
            "creation_time": "submit_time",
            "submit_time": "submit_time",
            "start_time": "start_time",
            "completed_time": "finish_time",
            "finish_time": "finish_time",
            "application_name": "application_id",
            "app": "application_id",
            "status": "service_status",
            "id": "id",
            "output_name": "output_name",
            "parameters": "output_name"
        }
        simple_filter["sort_field"] = sort_field_map.get(sort_by, "submit_time")
        simple_filter["sort_order"] = sort_dir if sort_dir in ("asc", "desc") else "desc"

        result = await api.acall(
            "AppService.enumerate_tasks_filtered",
            [offset, limit, simple_filter],
            _generate_numerical_uuid(),
            token
        )

        # Expected shape: [jobs, total]
        jobs: List[Dict[str, Any]] = []
        total = 0
        if isinstance(result, list) and len(result) >= 2:
            jobs = result[0] if isinstance(result[0], list) else []
            total = result[1] if isinstance(result[1], (int, float)) else len(jobs)

        # Keep parity with frontend behavior.
        jobs = [job for job in jobs if isinstance(job, dict) and job.get("status") != "deleted"]
        
        return {
            "items": jobs,
            "total": total,
            "count": len(jobs),
            "offset": offset,
            "limit": limit,
            "query": search,
            "status": status,
            "sort_by": sort_by,
            "sort_dir": sort_dir,
            "call": {
                "tool": "list_jobs",
                "backend_method": "AppService.enumerate_tasks_filtered",
                # Keep replay metadata aligned with the MCP tool schema.
                "arguments_executed": {
                    "offset": offset,
                    "limit": limit,
                    "sort_by": sort_by,
                    "sort_dir": sort_dir,
                    "status": status,
                    "service": service,
                    "search": search,
                    "include_archived": include_archived
                },
                "replayable": True
            },
            "source": "bvbrc-service",
            "ui_grid": _build_grid_payload(
                entity_type="job",
                items=jobs,
                source="bvbrc-service",
                result_type="list_result",
                pagination={
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "has_more": (offset + len(jobs)) < total
                },
                sort={
                    "sort_by": sort_by,
                    "sort_dir": sort_dir
                },
                columns=[
                    {"key": "status", "label": "Status", "sortable": True},
                    {"key": "id", "label": "Job ID", "sortable": True},
                    {"key": "application_name", "label": "Service", "sortable": True},
                    {"key": "submit_time", "label": "Submit", "sortable": True},
                    {"key": "completed_time", "label": "Completed", "sortable": True}
                ],
                selectable=True,
                multi_select=True,
                sortable=True
            )
        }
    except Exception as e:
        print(f"Error in list_jobs: {e}", file=sys.stderr)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_genome_assembly_app(api: JsonRpcCaller, token: str = None, user_id: str = None, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_ids: List[str] = None, max_bases: int = 10000000000, recipe: str = "auto", racon_iter: int = 2, pilon_iter: int = 2, trim: bool = False, target_depth: int = 200, normalize: bool = False, filtlong: bool = False, genome_size: int = 5000000, min_contig_len: int = 300, min_contig_cov: float = 5.0, output_path: str = None, output_file: str = None, debug: int = 0) -> str:
    app_name = "GenomeAssembly2"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "paired_end_libs": paired_end_libs,
            "single_end_libs": single_end_libs,
            "srr_ids": srr_ids,
            "max_bases": max_bases,
            "recipe": recipe,
            "racon_iter": racon_iter,
            "pilon_iter": pilon_iter,
            "trim": trim,
            "target_depth": target_depth,
            "normalize": normalize,
            "filtlong": filtlong,
            "genome_size": genome_size,
            "min_contig_len": min_contig_len,
            "min_contig_cov": min_contig_cov,
            "output_path": output_path,
            "output_file": output_file,
            "debug": debug
        })
        data = [app_name, params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": str(result),
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_comprehensive_genome_analysis_app(api: JsonRpcCaller, token: str = None, user_id: str = None, input_type: str = None, output_path: str = None, output_file: str = None, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_ids: List[str] = None, reference_assembly: str = None, recipe: str = "auto", racon_iter: int = 2, pilon_iter: int = 2, trim: bool = False, normalize: bool = False, filtlong: bool = False, target_depth: int = 200, genome_size: int = 5000000, min_contig_len: int = 300, min_contig_cov: float = 5.0, gto: str = None, genbank_file: str = None, contigs: str = None, scientific_name: str = None, taxonomy_id: int = None, code: int = 0, domain: str = "auto", public: bool = False, queue_nowait: bool = False, skip_indexing: bool = False, reference_genome_id: str = None, analyze_quality: bool = None, debug_level: int = 0) -> str:
    app_name = "ComprehensiveGenomeAnalysis"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "input_type": input_type,
            "output_path": output_path,
            "output_file": output_file,
            "paired_end_libs": paired_end_libs,
            "single_end_libs": single_end_libs,
            "srr_ids": srr_ids,
            "reference_assembly": reference_assembly,
            "recipe": recipe,
            "racon_iter": racon_iter,
            "pilon_iter": pilon_iter,
            "trim": trim,
            "normalize": normalize,
            "filtlong": filtlong,
            "target_depth": target_depth,
            "genome_size": genome_size,
            "min_contig_len": min_contig_len,
            "min_contig_cov": min_contig_cov,
            "gto": gto,
            "genbank_file": genbank_file,
            "contigs": contigs,
            "scientific_name": scientific_name,
            "taxonomy_id": taxonomy_id,
            "code": code,
            "domain": domain,
            "public": public,
            "queue_nowait": queue_nowait,
            "skip_indexing": skip_indexing,
            "reference_genome_id": reference_genome_id,
            "analyze_quality": analyze_quality,
            "debug_level": debug_level
        })
        params = _normalize_and_validate_comprehensive_genome_analysis_params(params)
        data = [app_name, params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_blast_app(api: JsonRpcCaller, token: str = None, user_id: str = None, input_type: str = None, input_source: str = None, input_fasta_data: str = None, input_id_list: List[str] = None, input_fasta_file: str = None, input_feature_group: str = None, input_genome_group: str = None, db_type: str = None, db_source: str = None, db_fasta_data: str = None, db_fasta_file: str = None, db_id_list: List[str] = None, db_feature_group: str = None, db_genome_group: str = None, db_genome_list: List[str] = None, db_taxon_list: List[str] = None, db_precomputed_database: str = None, blast_program: str = None, blast_evalue_cutoff: float = 1e-5, blast_max_hits: int = 300, blast_min_coverage: int = None, output_path: str = None, output_file: str = None) -> str:
    app_name = "Homology"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "input_type": input_type,
            "input_source": input_source,
            "input_fasta_data": input_fasta_data,
            "input_id_list": input_id_list,
            "input_fasta_file": input_fasta_file,
            "input_feature_group": input_feature_group,
            "input_genome_group": input_genome_group,
            "db_type": db_type,
            "db_source": db_source,
            "db_fasta_data": db_fasta_data,
            "db_fasta_file": db_fasta_file,
            "db_id_list": db_id_list,
            "db_feature_group": db_feature_group,
            "db_genome_group": db_genome_group,
            "db_genome_list": db_genome_list,
            "db_taxon_list": db_taxon_list,
            "db_precomputed_database": db_precomputed_database,
            "blast_program": blast_program,
            "blast_evalue_cutoff": blast_evalue_cutoff,
            "blast_max_hits": blast_max_hits,
            "blast_min_coverage": blast_min_coverage,
            "output_path": output_path,
            "output_file": output_file
        })
        data = [app_name, params, {}]
        print("data", data, file=sys.stdout)
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        print(f"result type: {type(result)}, result: {result}", file=sys.stdout)
        if result is None:
            return {
                "error": "No result returned from API",
                "errorType": "API_ERROR",
                "source": "bvbrc-service"
            }
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Exception in start_blast_app: {error_trace}", file=sys.stderr)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "traceback": error_trace,
            "source": "bvbrc-service"
        }

async def start_primer_design_app(api: JsonRpcCaller, token: str = None, user_id: str = None, output_file: str = None, output_path: str = None, input_type: str = None, sequence_input: str = None, SEQUENCE_ID: str = None, SEQUENCE_TARGET: List[List[int]] = None, SEQUENCE_INCLUDED_REGION: List[int] = None, SEQUENCE_EXCLUDED_REGION: List[int] = None, SEQUENCE_OVERLAP_JUNCTION_LIST: List[List[int]] = None, PRIMER_PICK_INTERNAL_OLIGO: int = None, PRIMER_PRODUCT_SIZE_RANGE: List[List[int]] = None, PRIMER_NUM_RETURN: int = None, PRIMER_MIN_SIZE: int = None, PRIMER_OPT_SIZE: int = None, PRIMER_MAX_SIZE: int = None, PRIMER_MAX_TM: float = None, PRIMER_MIN_TM: float = None, PRIMER_OPT_TM: float = None, PRIMER_PAIR_MAX_DIFF_TM: float = None, PRIMER_MAX_GC: float = None, PRIMER_MIN_GC: float = None, PRIMER_OPT_GC: float = None, PRIMER_SALT_MONOVALENT: float = None, PRIMER_SALT_DIVALENT: float = None, PRIMER_DNA_CONC: float = None, PRIMER_DNTP_CONC: float = None) -> str:
    app_name = "PrimerDesign"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "output_file": output_file,
            "output_path": output_path,
            "input_type": input_type,
            "sequence_input": sequence_input,
            "SEQUENCE_ID": SEQUENCE_ID,
            "SEQUENCE_TARGET": SEQUENCE_TARGET,
            "SEQUENCE_INCLUDED_REGION": SEQUENCE_INCLUDED_REGION,
            "SEQUENCE_EXCLUDED_REGION": SEQUENCE_EXCLUDED_REGION,
            "SEQUENCE_OVERLAP_JUNCTION_LIST": SEQUENCE_OVERLAP_JUNCTION_LIST,
            "PRIMER_PICK_INTERNAL_OLIGO": PRIMER_PICK_INTERNAL_OLIGO,
            "PRIMER_PRODUCT_SIZE_RANGE": PRIMER_PRODUCT_SIZE_RANGE,
            "PRIMER_NUM_RETURN": PRIMER_NUM_RETURN,
            "PRIMER_MIN_SIZE": PRIMER_MIN_SIZE,
            "PRIMER_OPT_SIZE": PRIMER_OPT_SIZE,
            "PRIMER_MAX_SIZE": PRIMER_MAX_SIZE,
            "PRIMER_MAX_TM": PRIMER_MAX_TM,
            "PRIMER_MIN_TM": PRIMER_MIN_TM,
            "PRIMER_OPT_TM": PRIMER_OPT_TM,
            "PRIMER_PAIR_MAX_DIFF_TM": PRIMER_PAIR_MAX_DIFF_TM,
            "PRIMER_MAX_GC": PRIMER_MAX_GC,
            "PRIMER_MIN_GC": PRIMER_MIN_GC,
            "PRIMER_OPT_GC": PRIMER_OPT_GC,
            "PRIMER_SALT_MONOVALENT": PRIMER_SALT_MONOVALENT,
            "PRIMER_SALT_DIVALENT": PRIMER_SALT_DIVALENT,
            "PRIMER_DNA_CONC": PRIMER_DNA_CONC,
            "PRIMER_DNTP_CONC": PRIMER_DNTP_CONC
        })
        data = ["PrimerDesign", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_variation_app(api: JsonRpcCaller, token: str = None, user_id: str = None, reference_genome_id: str = None, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_ids: List[str] = None, mapper: str = "BWA-mem", caller: str = "FreeBayes", output_path: str = None, output_file: str = None, debug: bool = False) -> str:
    app_name = "Variation"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "reference_genome_id": reference_genome_id,
            "paired_end_libs": paired_end_libs,
            "single_end_libs": single_end_libs,
            "srr_ids": srr_ids,
            "mapper": mapper,
            "caller": caller,
            "output_path": output_path,
            "output_file": output_file,
            "debug": debug
        })
        data = ["Variation", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_tnseq_app(api: JsonRpcCaller, token: str = None, user_id: str = None, experimental_conditions: List[str] = None, contrasts: List[str] = None, read_files: List[Dict] = None, reference_genome_id: str = None, recipe: str = "gumbel", protocol: str = "sassetti", primer: str = "", output_path: str = None, output_file: str = None) -> str:
    app_name = "TnSeq"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "experimental_conditions": experimental_conditions,
            "contrasts": contrasts,
            "read_files": read_files,
            "reference_genome_id": reference_genome_id,
            "recipe": recipe,
            "protocol": protocol,
            "primer": primer,
            "output_path": output_path,
            "output_file": output_file
        })
        data = ["TnSeq", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_bacterial_genome_tree_app(api: JsonRpcCaller, token: str = None, user_id: str = None, output_path: str = None, output_file: str = None, genome_ids: List[str] = None, genome_groups: List[str] = None, optional_genome_ids: List[str] = None, genome_metadata_fields: str = None, number_of_genes: int = 20, bootstraps: int = 100, max_genomes_missing: int = 0, max_allowed_dups: int = 0) -> str:
    app_name = "CodonTree"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        if isinstance(genome_metadata_fields, list):
            genome_metadata_fields = ",".join(genome_metadata_fields)
        if genome_groups and not genome_ids:
            genome_ids = []
        params = _filter_none_params({
            "output_path": output_path,
            "output_file": output_file,
            "genome_ids": genome_ids,
            "genome_groups": genome_groups,
            "optional_genome_ids": optional_genome_ids,
            "genome_metadata_fields": genome_metadata_fields,
            "number_of_genes": number_of_genes,
            "bootstraps": bootstraps,
            "max_genomes_missing": max_genomes_missing,
            "max_allowed_dups": max_allowed_dups
        })
        data = [app_name, params, { 'base_url': 'https://www.bv-brc.org' }]
        print("data", data)
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_gene_tree_app(api: JsonRpcCaller, token: str = None, user_id: str = None, sequences: List[str] = None, alignment_program: str = None, trim_threshold: float = None, gap_threshold: float = None, alphabet: str = None, substitution_model: str = None, bootstrap: int = None, recipe: str = "RAxML", tree_type: str = None, feature_metadata_fields: str = None, genome_metadata_fields: str = None, output_path: str = None, output_file: str = None) -> str:
    app_name = "GeneTree"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "sequences": sequences,
            "alignment_program": alignment_program,
            "trim_threshold": trim_threshold,
            "gap_threshold": gap_threshold,
            "alphabet": alphabet,
            "substitution_model": substitution_model,
            "bootstrap": bootstrap,
            "recipe": recipe,
            "tree_type": tree_type,
            "feature_metadata_fields": feature_metadata_fields,
            "genome_metadata_fields": genome_metadata_fields,
            "output_path": output_path,
            "output_file": output_file
        })
        data = [app_name, params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_core_genome_mlst_app(api: JsonRpcCaller, token: str = None, user_id: str = None, input_genome_type: str = "genome_group", analysis_type: str = "chewbbaca", input_genome_group: str = None, input_genome_fasta: str = None, schema_location: str = None, input_schema_selection: str = None, output_path: str = None, output_file: str = None) -> str:
    app_name = "CoreGenomeMLST"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "input_genome_type": input_genome_type,
            "analysis_type": analysis_type,
            "input_genome_group": input_genome_group,
            "input_genome_fasta": input_genome_fasta,
            "schema_location": schema_location,
            "input_schema_selection": input_schema_selection,
            "output_path": output_path,
            "output_file": output_file
        })
        data = ["CoreGenomeMLST", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_whole_genome_snp_app(api: JsonRpcCaller, token: str = None, user_id: str = None, input_genome_type: str = None, majority_threshold: float = 0.5, min_mid_linkage: int = 10, max_mid_linkage: int = 40, analysis_type: str = "Whole Genome SNP Analysis", input_genome_group: str = None, input_genome_fasta: str = None, output_path: str = None, output_file: str = None) -> str:
    app_name = "WholeGenomeSNPAnalysis"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "input_genome_type": input_genome_type,
            "majority_threshold": majority_threshold,
            "min_mid_linkage": min_mid_linkage,
            "max_mid_linkage": max_mid_linkage,
            "analysis_type": analysis_type,
            "input_genome_group": input_genome_group,
            "input_genome_fasta": input_genome_fasta,
            "output_path": output_path,
            "output_file": output_file
        })
        data = [app_name, params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_taxonomic_classification_app(api: JsonRpcCaller, token: str = None, user_id: str = None, host_genome: str = "no_host", analysis_type: str = "16S", paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_libs: List[Dict] = None, database: str = "SILVA", save_classified_sequences: bool = False, save_unclassified_sequences: bool = False, confidence_interval: float = 0.1, output_path: str = None, output_file: str = None) -> str:
    app_name = "TaxonomicClassification"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "host_genome": host_genome,
            "analysis_type": analysis_type,
            "paired_end_libs": paired_end_libs,
            "single_end_libs": single_end_libs,
            "srr_libs": srr_libs,
            "database": database,
            "save_classified_sequences": save_classified_sequences,
            "save_unclassified_sequences": save_unclassified_sequences,
            "confidence_interval": confidence_interval,
            "output_path": output_path,
            "output_file": output_file
        })
        data = ["TaxonomicClassification", params, { 'base_url': 'https://www.bv-brc.org' }]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_metagenomic_binning_app(api: JsonRpcCaller, token: str = None, user_id: str = None, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_ids: List[str] = None, contigs: str = None, genome_group: str = None, skip_indexing: bool = False, recipe: str = None, viral_recipe: str = None, output_path: str = None, output_file: str = None, force_local_assembly: bool = False, force_inline_annotation: bool = True, perform_bacterial_binning: bool = True, perform_viral_binning: bool = False, perform_viral_annotation: bool = False, perform_bacterial_annotation: bool = True, assembler: str = "", danglen: str = "50", min_contig_len: int = 400, min_contig_cov: float = 4.0) -> str:
    app_name = "MetagenomeBinning"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "paired_end_libs": paired_end_libs,
            "single_end_libs": single_end_libs,
            "srr_ids": srr_ids,
            "contigs": contigs,
            "genome_group": genome_group,
            "skip_indexing": skip_indexing,
            "recipe": recipe,
            "viral_recipe": viral_recipe,
            "output_path": output_path,
            "output_file": output_file,
            "force_local_assembly": force_local_assembly,
            "force_inline_annotation": force_inline_annotation,
            "perform_bacterial_binning": perform_bacterial_binning,
            "perform_viral_binning": perform_viral_binning,
            "perform_viral_annotation": perform_viral_annotation,
            "perform_bacterial_annotation": perform_bacterial_annotation,
            "assembler": assembler,
            "danglen": danglen,
            "min_contig_len": min_contig_len,
            "min_contig_cov": min_contig_cov
        })
        data = [app_name, params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_metagenomic_read_mapping_app(api: JsonRpcCaller, token: str = None, user_id: str = None, gene_set_type: str = None, gene_set_name: str = None, gene_set_fasta: str = None, gene_set_feature_group: str = None, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_ids: List[str] = None, output_path: str = None, output_file: str = None) -> str:
    app_name = "MetagenomicReadMapping"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "gene_set_type": gene_set_type,
            "gene_set_name": gene_set_name,
            "gene_set_fasta": gene_set_fasta,
            "gene_set_feature_group": gene_set_feature_group,
            "paired_end_libs": paired_end_libs,
            "single_end_libs": single_end_libs,
            "srr_ids": srr_ids,
            "output_path": output_path,
            "output_file": output_file
        })
        data = [app_name, params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_rnaseq_app(api: JsonRpcCaller, token: str = None, user_id: str = None, experimental_conditions: List[str] = None, contrasts: str = None, strand_specific: bool = True, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_libs: List[Dict] = None, reference_genome_id: str = None, genome_type: str = None, recipe: str = "HTSeq-DESeq", host_ftp: str = None, output_path: str = None, output_file: str = None, trimming: bool = False, unit_test: str = None, skip_sampling: str = None) -> str:
    app_name = "RNASeq"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "experimental_conditions": experimental_conditions,
            "contrasts": contrasts,
            "strand_specific": strand_specific,
            "paired_end_libs": paired_end_libs,
            "single_end_libs": single_end_libs,
            "srr_libs": srr_libs,
            "reference_genome_id": reference_genome_id,
            "genome_type": genome_type,
            "recipe": recipe,
            "host_ftp": host_ftp,
            "output_path": output_path,
            "output_file": output_file,
            "trimming": trimming,
            "unit_test": unit_test,
            "skip_sampling": skip_sampling
        })
        data = [app_name, params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_expression_import_app(api: JsonRpcCaller, token: str = None, user_id: str = None, xfile: str = None, mfile: str = None, ustring: str = None, output_path: str = None, output_file: str = None) -> str:
    app_name = "ExpressionImport"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "xfile": xfile,
            "mfile": mfile,
            "ustring": ustring,
            "output_path": output_path,
            "output_file": output_file
        })
        data = ["ExpressionImport", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_sars_wastewater_analysis_app(api: JsonRpcCaller, token: str = None, user_id: str = None, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_libs: List[Dict] = None, recipe: str = "auto", primers: str = "ARTIC", minimum_base_quality_score: int = 20, minimum_genome_coverage: int = 60, agg_minimum_lineage_abundance: float = 0.01, minimum_coverage_depth: int = 0, confirmedonly: bool = False, minimum_lineage_abundance: float = 0.001, coverage_estimate: int = 10, timeseries_plot_interval: str = "0", primer_version: str = None, barcode_csv: str = None, sample_metadata_csv: str = None, keep_intermediates: bool = True, output_path: str = None, output_file: str = None, debug_level: int = 0) -> str:
    app_name = "SARSWastewaterAnalysis"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "paired_end_libs": paired_end_libs,
            "single_end_libs": single_end_libs,
            "srr_libs": srr_libs,
            "recipe": recipe,
            "primers": primers,
            "minimum_base_quality_score": minimum_base_quality_score,
            "minimum_genome_coverage": minimum_genome_coverage,
            "agg_minimum_lineage_abundance": agg_minimum_lineage_abundance,
            "minimum_coverage_depth": minimum_coverage_depth,
            "confirmedonly": confirmedonly,
            "minimum_lineage_abundance": minimum_lineage_abundance,
            "coverage_estimate": coverage_estimate,
            "timeseries_plot_interval": timeseries_plot_interval,
            "primer_version": primer_version,
            "barcode_csv": barcode_csv,
            "sample_metadata_csv": sample_metadata_csv,
            "keep_intermediates": keep_intermediates,
            "output_path": output_path,
            "output_file": output_file,
            "debug_level": debug_level
        })
        data = ["SARSWastewaterAnalysis", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_sequence_submission_app(api: JsonRpcCaller, token: str = None, user_id: str = None, input_source: str = None, input_fasta_data: str = None, input_fasta_file: str = None, input_genome_group: str = None, metadata: str = None, affiliation: str = None, first_name: str = None, last_name: str = None, email: str = None, consortium: str = None, country: str = None, phoneNumber: str = None, street: str = None, postal_code: str = None, city: str = None, state: str = None, numberOfSequences: str = None, output_path: str = None, output_file: str = None) -> str:
    app_name = "SequenceSubmission"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "input_source": input_source,
            "input_fasta_data": input_fasta_data,
            "input_fasta_file": input_fasta_file,
            "input_genome_group": input_genome_group,
            "metadata": metadata,
            "affiliation": affiliation,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "consortium": consortium,
            "country": country,
            "phoneNumber": phoneNumber,
            "street": street,
            "postal_code": postal_code,
            "city": city,
            "state": state,
            "numberOfSequences": numberOfSequences,
            "output_path": output_path,
            "output_file": output_file
        })
        data = ["SequenceSubmission", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_influenza_ha_subtype_conversion_app(api: JsonRpcCaller, token: str = None, user_id: str = None, input_source: str = None, input_fasta_data: str = None, input_fasta_file: str = None, input_feature_group: str = None, input_feature_list: str = None, types: str = None, output_path: str = None, output_file: str = None) -> str:
    app_name = "InfluenzaHASubtypeConversion"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "input_source": input_source,
            "input_fasta_data": input_fasta_data,
            "input_fasta_file": input_fasta_file,
            "input_feature_group": input_feature_group,
            "input_feature_list": input_feature_list,
            "types": types,
            "output_path": output_path,
            "output_file": output_file
        })
        data = ["InfluenzaHASubtypeConversion", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_subspecies_classification_app(api: JsonRpcCaller, token: str = None, user_id: str = None, input_source: str = None, input_fasta_data: str = None, input_fasta_file: str = None, input_genome_group: str = None, ref_msa_fasta: str = None, virus_type: str = None, output_path: str = None, output_file: str = None) -> str:
    app_name = "SubspeciesClassification"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "input_source": input_source,
            "input_fasta_data": input_fasta_data,
            "input_fasta_file": input_fasta_file,
            "input_genome_group": input_genome_group,
            "ref_msa_fasta": ref_msa_fasta,
            "virus_type": virus_type,
            "output_path": output_path,
            "output_file": output_file
        })
        data = ["SubspeciesClassification", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_viral_assembly_app(api: JsonRpcCaller, token: str = None, user_id: str = None, paired_end_lib: Dict = None, single_end_lib: Dict = None, srr_id: str = None, recipe: str = "auto", module: str = None, viral_size: str = "5M", output_path: str = None, output_file: str = None, debug: int = 0) -> str:
    app_name = "ViralAssembly"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "paired_end_lib": paired_end_lib,
            "single_end_lib": single_end_lib,
            "srr_id": srr_id,
            "recipe": recipe,
            "module": module,
            "viral_size": viral_size,
            "output_path": output_path,
            "output_file": output_file,
            "debug": debug
        })
        data = ["ViralAssembly", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_fastq_utils_app(api: JsonRpcCaller, token: str = None, user_id: str = None, reference_genome_id: str = None, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_libs: List[Dict] = None, output_path: str = None, output_file: str = None, recipe: List[str] = None) -> str:
    app_name = "FastqUtils"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "reference_genome_id": reference_genome_id,
            "paired_end_libs": paired_end_libs,
            "single_end_libs": single_end_libs,
            "srr_libs": srr_libs,
            "output_path": output_path,
            "output_file": output_file,
            "recipe": recipe
        })
        print(json.dumps(params, indent=4))
        data = [app_name, params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_genome_alignment_app(api: JsonRpcCaller, token: str = None, user_id: str = None, genome_ids: List[str] = None, recipe: str = "progressiveMauve", seedWeight: float = None, maxGappedAlignerLength: float = None, maxBreakpointDistanceScale: float = None, conservationDistanceScale: float = None, weight: float = None, minScaledPenalty: float = None, hmmPGoHomologous: float = None, hmmPGoUnrelated: float = None, output_path: str = None, output_file: str = None) -> str:
    app_name = "GenomeAlignment"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "genome_ids": genome_ids,
            "recipe": recipe,
            "seedWeight": seedWeight,
            "maxGappedAlignerLength": maxGappedAlignerLength,
            "maxBreakpointDistanceScale": maxBreakpointDistanceScale,
            "conservationDistanceScale": conservationDistanceScale,
            "weight": weight,
            "minScaledPenalty": minScaledPenalty,
            "hmmPGoHomologous": hmmPGoHomologous,
            "hmmPGoUnrelated": hmmPGoUnrelated,
            "output_path": output_path,
            "output_file": output_file
        })
        data = ["GenomeAlignment", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_sars_genome_analysis_app(api: JsonRpcCaller, token: str = None, user_id: str = None, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_ids: List[str] = None, recipe: str = "auto", primers: str = "ARTIC", primer_version: str = None, min_depth: int = 100, max_depth: int = 8000, keep_intermediates: int = 0, output_path: str = None, output_file: str = None, debug_level: int = 0) -> str:
    app_name = "SARS2Assembly"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "paired_end_libs": paired_end_libs,
            "single_end_libs": single_end_libs,
            "srr_ids": srr_ids,
            "recipe": recipe,
            "primers": primers,
            "primer_version": primer_version,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "keep_intermediates": keep_intermediates,
            "output_path": output_path,
            "output_file": output_file,
            "debug_level": debug_level
        })
        data = ["SARS2Assembly", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_msa_snp_analysis_app(api: JsonRpcCaller, token: str = None, user_id: str = None, input_status: str = "unaligned", input_type: str = "input_group", fasta_files: List[Dict] = None, select_genomegroup: List[str] = None, feature_groups: List[str] = None, feature_list: List[str] = None, genome_list: List[str] = None, aligner: str = "Muscle", alphabet: str = "dna", fasta_keyboard_input: str = "", ref_type: str = "none", strategy: str = "auto", ref_string: str = "", output_path: str = None, output_file: str = None) -> str:
    app_name = "MSA"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "input_status": input_status,
            "input_type": input_type,
            "fasta_files": fasta_files,
            "select_genomegroup": select_genomegroup,
            "feature_groups": feature_groups,
            "feature_list": feature_list,
            "genome_list": genome_list,
            "aligner": aligner,
            "alphabet": alphabet,
            "fasta_keyboard_input": fasta_keyboard_input,
            "ref_type": ref_type,
            "strategy": strategy,
            "ref_string": ref_string,
            "output_path": output_path,
            "output_file": output_file
        })
        data = ["MSA", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_metacats_app(api: JsonRpcCaller, token: str = None, user_id: str = None, output_path: str = None, output_file: str = None, p_value: float = 0.05, year_ranges: str = None, metadata_group: str = None, input_type: str = None, alphabet: str = "na", groups: List[str] = None, alignment_file: str = None, group_file: str = None, alignment_type: str = None, auto_groups: List[Dict] = None) -> str:
    app_name = "MetaCATS"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "output_path": output_path,
            "output_file": output_file,
            "p_value": p_value,
            "year_ranges": year_ranges,
            "metadata_group": metadata_group,
            "input_type": input_type,
            "alphabet": alphabet,
            "groups": groups,
            "alignment_file": alignment_file,
            "group_file": group_file,
            "alignment_type": alignment_type,
            "auto_groups": auto_groups
        })
        data = ["MetaCATS", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_proteome_comparison_app(api: JsonRpcCaller, token: str = None, user_id: str = None, genome_ids: List[str] = None, user_genomes: List[str] = None, user_feature_groups: List[str] = None, reference_genome_index: int = 1, min_seq_cov: float = 0.30, max_e_val: float = 1e-5, min_ident: float = 0.1, min_positives: float = 0.2, output_path: str = None, output_file: str = None) -> str:
    app_name = "GenomeComparison"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "genome_ids": genome_ids,
            "user_genomes": user_genomes,
            "user_feature_groups": user_feature_groups,
            "reference_genome_index": reference_genome_index,
            "min_seq_cov": min_seq_cov,
            "max_e_val": max_e_val,
            "min_ident": min_ident,
            "min_positives": min_positives,
            "output_path": output_path,
            "output_file": output_file
        })
        data = ["GenomeComparison", params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_comparative_systems_app(api: JsonRpcCaller, token: str = None, user_id: str = None, output_path: str = None, output_file: str = None, genome_ids: List[str] = None, genome_groups: List[str] = None) -> str:
    app_name = "ComparativeSystems"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "output_path": output_path,
            "output_file": output_file,
            "genome_ids": genome_ids,
            "genome_groups": genome_groups
        })
        data = [app_name, params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_docking_app(api: JsonRpcCaller, token: str = None, user_id: str = None, protein_input_type: str = None, input_pdb: List[str] = None, user_pdb_file: List[str] = None, ligand_library_type: str = None, ligand_named_library: str = None, ligand_smiles_list: List[str] = None, ligand_ws_file: str = None, top_n: int = None, batch_size: int = 10, output_path: str = None, output_file: str = None) -> str:
    app_name = "Docking"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)
        params = _filter_none_params({
            "protein_input_type": protein_input_type,
            "input_pdb": input_pdb,
            "user_pdb_file": user_pdb_file,
            "ligand_library_type": ligand_library_type,
            "ligand_named_library": ligand_named_library,
            "ligand_smiles_list": ligand_smiles_list,
            "ligand_ws_file": ligand_ws_file,
            "top_n": top_n,
            "batch_size": batch_size,
            "output_path": output_path,
            "output_file": output_file
        })
        data = [app_name, params, {}]
        result = await api.acall("AppService.start_app2", data, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }

async def start_similar_genome_finder_app(api: JsonRpcCaller, token: str = None, user_id: str = None, selectedGenomeId: str = None, fasta_file: str = None, max_pvalue: float = None, max_distance: float = None, max_hits: int = None, include_reference: bool = None, include_representative: bool = None, include_bacterial: bool = None, include_viral: bool = None, output_path: str = None, output_file: str = None) -> str:
    app_name = "SimilarGenomeFinder"
    try:
        # Set default values if not provided
        output_path, output_file = _set_default_output_paths(user_id, app_name, output_path, output_file)
        # Resolve relative paths to absolute paths
        output_path = _resolve_output_path(output_path, user_id)

        # Call the Minhash.compute_genome_distance_for_genome2 method
        function_call = ""
        if selectedGenomeId:
            params = [selectedGenomeId, max_pvalue, max_distance, max_hits, include_reference, include_representative, include_bacterial, include_viral]
            function_call = "Minhash.compute_genome_distance_for_genome2"
        elif fasta_file:
            params = [fasta_file, max_pvalue, max_distance, max_hits, include_reference, include_representative, include_bacterial, include_viral]
            function_call = "Minhash.compute_genome_distance_for_fasta2"
        else:
            return "Error: selectedGenomeId or fasta_file is required"

        result = await api.acall(function_call, params, _generate_numerical_uuid(), token)
        if isinstance(result, (list, dict)):
            return {
                "data": result,
                "source": "bvbrc-service"
            }
        return {
            "data": result,
            "source": "bvbrc-service"
        }
    except Exception as e:
        print(e)
        return {
            "error": str(e),
            "errorType": "API_ERROR",
            "source": "bvbrc-service"
        }