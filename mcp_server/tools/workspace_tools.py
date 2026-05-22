
from fastmcp import FastMCP
from functions.workspace_functions import (
    workspace_get_file_metadata, workspace_download_file,
    workspace_upload as workspace_upload_func,
    workspace_read_range, workspace_browse
)
from common.json_rpc import JsonRpcCaller
from common.token_provider import TokenProvider
import json
from typing import List, Optional, Union
import sys
import os
import csv
import base64
import mimetypes
from pymongo import MongoClient
from pymongo.errors import PyMongoError

_file_registry_client: Optional[MongoClient] = None
SUPPORTED_WORKSPACE_TYPES = [
    "csv",
    "diffexp_input_data",
    "diffexp_input_metadata",
    "doc",
    "docx",
    "embl",
    "feature_dna_fasta",
    "feature_protein_fasta",
    "genbank_file",
    "gff",
    "gif",
    "graph",
    "jpg",
    "json",
    "nwk",
    "pdf",
    "phyloxml",
    "png",
    "pdb",
    "ppt",
    "pptx",
    "reads",
    "string",
    "svg",
    "tar_gz",
    "tbi",
    "tsv",
    "txt",
    "unspecified",
    "vcf",
    "vcf_gz",
    "wig",
    "xls",
    "xlsx",
    "xml",
]

def _normalize_string_or_list(value: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    """Normalize a single string or list of strings to a cleaned list."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else None
    if isinstance(value, list):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        return normalized or None
    return None

def extract_userid_from_token(token: str = None) -> str:
    """
    Extract user ID from JWT token.
    Returns a default user ID if token is None or invalid.
    """
    if not token:
        return None

    try:
        user_id = token.split('|')[0].replace('un=','')
        return user_id

    except Exception as e:
        print(f"Error extracting user ID from token: {e}")
        return None

def get_user_home_path(user_id: str) -> str:
    """
    Get the user's home path in the workspace.

    Args:
        user_id: User ID extracted from token

    Returns:
        User's home path in format /{user_id}/home
    """
    if not user_id:
        return "/"
    return f"/{user_id}/home"

def resolve_relative_paths(paths: List[str], user_id: str) -> List[str]:
    """
    Convert relative paths to absolute paths by prepending user's home directory.

    Args:
        paths: List of relative paths
        user_id: User ID extracted from token

    Returns:
        List of absolute paths
    """
    if not paths:
        return [get_user_home_path(user_id)]

    home_path = get_user_home_path(user_id)
    resolved_paths = []

    for path in paths:
        # Strip /workspace prefix if present (orchestrator may add this)
        if path.startswith('/workspace/'):
            path = path[len('/workspace'):]
        
        # Check if path already contains the user_id to prevent duplication
        if path.startswith(f'/{user_id}/'):
            # Path already has correct user_id prefix, return as-is
            resolved_paths.append(path)
        elif path.startswith('/'):
            # Absolute path but doesn't start with user_id - could be another user's path or system path
            # Return as-is (don't modify other users' paths)
            resolved_paths.append(path)
        elif path == 'home':
            # If path is just "home", return home_path directly to avoid /home/home
            resolved_paths.append(home_path)
        elif path.startswith(f'{user_id}/'):
            # Path starts with user_id but no leading / - add leading / and return
            resolved_paths.append(f'/{path}')
        else:
            # Treat as relative to home directory
            resolved_paths.append(f"{home_path}/{path}")

    return resolved_paths

def resolve_relative_path(path: str, user_id: str) -> str:
    """
    Convert a relative path to an absolute path by prepending user's home directory.

    Args:
        path: Relative path
        user_id: User ID extracted from token

    Returns:
        Absolute path
    """
    if not path or path == '/':
        return get_user_home_path(user_id)

    # Strip /workspace prefix if present (orchestrator may add this)
    if path.startswith('/workspace/'):
        path = path[len('/workspace'):]

    home_path = get_user_home_path(user_id)

    # Check if path already contains the user_id to prevent duplication
    if path.startswith(f'/{user_id}/'):
        # Path already has correct user_id prefix, return as-is
        return path
    elif path.startswith('/'):
        # Absolute path but doesn't start with user_id - could be another user's path or system path
        # Return as-is (don't modify other users' paths)
        return path
    elif path == 'home':
        # If path is just "home", return home_path directly to avoid /home/home
        return home_path
    elif path.startswith(f'{user_id}/'):
        # Path starts with user_id but no leading / - add leading / and return
        return f'/{path}'
    else:
        # Treat as relative to home directory
        return f"{home_path}/{path}"

def _is_user_placeholder_segment(segment: str) -> bool:
    """Return True when a path segment looks like a user-id placeholder."""
    if not segment:
        return False

    normalized = segment.strip().lower()
    placeholder_aliases = {
        "user",
        "username",
        "{username}",
        "<username>",
        "user_id",
        "{user_id}",
        "<user_id>",
        "userid",
        "{userid}",
        "<userid>",
        "__user_id__",  # __USER_ID__ (template token from prompts)
    }
    if normalized in placeholder_aliases:
        return True

    # Backward compatibility for historic examples shown in tool docs/prompts.
    if normalized in ("user1@patricbrc.org", "user@patricbrc.org",
                      "user@domain", "user@domain.com"):
        return True

    return False

def _sanitize_workspace_browse_path(path: Optional[str], user_id: Optional[str]) -> Optional[str]:
    """Clean common LLM path placeholders for workspace browsing."""
    if path is None:
        return None

    cleaned = path.strip()
    if not cleaned:
        return None

    is_absolute = cleaned.startswith("/")
    segments = [segment for segment in cleaned.split("/") if segment]
    had_placeholder_segment = any(_is_user_placeholder_segment(segment) for segment in segments)

    # Remove placeholder segments in any common form.
    segments = [segment for segment in segments if not _is_user_placeholder_segment(segment)]

    # Collapse duplicate adjacent "home" segments.
    collapsed = []
    for segment in segments:
        if segment == "home" and collapsed and collapsed[-1] == "home":
            continue
        collapsed.append(segment)
    segments = collapsed

    if not segments:
        return None

    # If placeholder cleanup detected placeholder-driven paths, always anchor to
    # the authenticated user's home path.
    if user_id and had_placeholder_segment:
        if segments and segments[0] == "home":
            suffix = "/".join(segments[1:])
        else:
            suffix = "/".join(segments)
        base = get_user_home_path(user_id)
        return f"{base}/{suffix}" if suffix else base

    reconstructed = "/".join(segments)
    if is_absolute:
        reconstructed = f"/{reconstructed}"
    return reconstructed

def _get_file_registry_client(file_utilities_config: dict) -> Optional[MongoClient]:
    """Create or reuse a MongoDB client for session file lookups."""
    global _file_registry_client
    mongo_url = (file_utilities_config or {}).get("mongo_url")
    if not mongo_url:
        return None
    if _file_registry_client is None:
        _file_registry_client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
    return _file_registry_client

def _get_registered_file_path(session_id: str, file_id: str, file_utilities_config: dict) -> Optional[str]:
    """Resolve filePath from configured session_files registry when available."""
    client = _get_file_registry_client(file_utilities_config)
    if client is None:
        return None

    db_name = file_utilities_config.get("mongo_database", "copilot")
    collection_name = file_utilities_config.get("mongo_collection", "session_files")
    collection = client[db_name][collection_name]
    queries = [
        {"session_id": session_id, "fileId": file_id},
        {"sessionId": session_id, "fileId": file_id},
        {"session_id": session_id, "file_id": file_id},
    ]

    try:
        for query in queries:
            record = collection.find_one(query)
            if record and record.get("filePath"):
                return record["filePath"]
    except PyMongoError as exc:
        print(f"Mongo file lookup failed: {exc}", file=sys.stderr)
    return None

def _get_registered_file_record(session_id: str, file_id: str, file_utilities_config: dict) -> Optional[dict]:
    """Resolve full file registry record for a session file when available."""
    client = _get_file_registry_client(file_utilities_config)
    if client is None:
        return None

    db_name = file_utilities_config.get("mongo_database", "copilot")
    collection_name = file_utilities_config.get("mongo_collection", "session_files")
    collection = client[db_name][collection_name]
    queries = [
        {"session_id": session_id, "fileId": file_id},
        {"sessionId": session_id, "fileId": file_id},
        {"session_id": session_id, "file_id": file_id},
    ]

    try:
        for query in queries:
            record = collection.find_one(query)
            if record:
                return record
    except PyMongoError as exc:
        print(f"Mongo file lookup failed: {exc}", file=sys.stderr)
    return None

def _is_within_base_path(candidate_path: str, base_path: str) -> bool:
    """Return True when candidate_path resolves under base_path."""
    try:
        base_real = os.path.realpath(base_path)
        candidate_real = os.path.realpath(candidate_path)
        return os.path.commonpath([candidate_real, base_real]) == base_real
    except ValueError:
        return False

def _build_local_metadata(file_path: str, session_id: Optional[str] = None, file_id: Optional[str] = None, registry_record: Optional[dict] = None) -> dict:
    """Build normalized metadata shape for local files."""
    stat_result = os.stat(file_path)
    guessed_content_type, _ = mimetypes.guess_type(file_path)
    content_type = guessed_content_type or "application/octet-stream"
    ext = os.path.splitext(file_path)[1].lower()
    binary_exts = {".gz", ".zip", ".tar", ".bam", ".sam", ".png", ".jpg", ".jpeg", ".pdf", ".bin"}
    is_binary = ext in binary_exts

    record = registry_record or {}
    return {
        "source_type": "local",
        "source": "bvbrc-workspace",
        "identifier": file_id or record.get("fileId") or os.path.basename(file_path),
        "name": os.path.basename(file_path),
        "path": file_path,
        "size_bytes": stat_result.st_size,
        "content_type": content_type,
        "is_binary": is_binary,
        "created_at": stat_result.st_ctime,
        "updated_at": stat_result.st_mtime,
        "checksum_sha256": record.get("sha256"),
        "session_id": session_id or record.get("session_id") or record.get("sessionId"),
        "file_id": file_id or record.get("fileId") or record.get("file_id"),
        "workspace_path": record.get("workspace_path") or record.get("workspacePath"),
    }

def _workspace_meta_array_to_dict(meta_array: list) -> dict:
    """Convert Workspace.get metadata array to dict when needed."""
    if not isinstance(meta_array, list) or len(meta_array) < 12:
        return {}
    return {
        "name": meta_array[0],
        "type": meta_array[1],
        "path": meta_array[2],
        "creation_time": meta_array[3],
        "id": meta_array[4],
        "owner_id": meta_array[5],
        "size": meta_array[6],
        "userMeta": meta_array[7],
        "autoMeta": meta_array[8],
        "user_permissions": meta_array[9],
        "global_permission": meta_array[10],
        "link_reference": meta_array[11],
    }

def _build_workspace_metadata(workspace_response: dict, resolved_path: str) -> dict:
    """Build normalized metadata shape for workspace files."""
    metadata = None
    data_field = workspace_response.get("data")
    if isinstance(data_field, dict):
        metadata = data_field
    elif isinstance(data_field, list):
        if data_field and isinstance(data_field[0], list) and data_field[0]:
            first = data_field[0][0]
            if isinstance(first, list):
                metadata = _workspace_meta_array_to_dict(first)
            elif isinstance(first, dict):
                metadata = first
    if not isinstance(metadata, dict):
        metadata = workspace_response if isinstance(workspace_response, dict) else {}

    user_meta = metadata.get("userMeta", {}) if isinstance(metadata.get("userMeta"), dict) else {}
    auto_meta = metadata.get("autoMeta", {}) if isinstance(metadata.get("autoMeta"), dict) else {}
    content_type = user_meta.get("content_type") or auto_meta.get("content_type") or "application/octet-stream"
    name = metadata.get("name") or os.path.basename(resolved_path)

    return {
        "source_type": "workspace",
        "source": "bvbrc-workspace",
        "identifier": metadata.get("id") or resolved_path,
        "name": name,
        "path": f"{metadata.get('path', '')}{name}" if metadata.get("path") else resolved_path,
        "size_bytes": metadata.get("size"),
        "content_type": content_type,
        "is_binary": auto_meta.get("is_binary"),
        "created_at": metadata.get("creation_time"),
        "updated_at": auto_meta.get("last_modified"),
        "checksum_sha256": user_meta.get("sha256") or auto_meta.get("sha256"),
        "session_id": None,
        "file_id": None,
        "workspace_path": resolved_path,
    }

def _sanitize_metadata_for_assistant(metadata: dict) -> dict:
    """Return a user-safe metadata subset for assistant-facing responses."""
    if not isinstance(metadata, dict):
        return metadata
    safe_keys = [
        "source_type",
        "source",
        "name",
        "size_bytes",
        "content_type",
        "is_binary",
        "created_at",
        "updated_at",
        "checksum_sha256",
    ]
    return {k: metadata.get(k) for k in safe_keys}

def _resolve_local_file_path(session_id: str, file_id: str, file_utilities_config: Optional[dict]) -> str:
    """Resolve local session file path using configured file_utilities settings."""
    config = file_utilities_config or {}
    base_path = config.get("session_base_path")
    if not base_path:
        raise ValueError("file_utilities.session_base_path must be configured")

    registered_path = _get_registered_file_path(session_id, file_id, config)
    if registered_path and os.path.exists(registered_path):
        return registered_path

    downloads_path = os.path.join(base_path, session_id, "downloads")

    extensions = ["", ".json", ".csv", ".tsv", ".txt"]
    for ext in extensions:
        candidate = os.path.join(downloads_path, f"{file_id}{ext}")
        if os.path.exists(candidate):
            return candidate

    return os.path.join(downloads_path, file_id)

def _detect_local_file_type(file_path: str) -> str:
    """Detect local file type for line-oriented reads."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".json":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return "json_array"
            if isinstance(data, dict):
                return "json_object"
        except Exception:
            pass
    if ext == ".csv":
        return "csv"
    if ext == ".tsv":
        return "tsv"
    return "text"

def _read_local_file_lines(file_path: str, start: int, end: Optional[int], limit: int) -> dict:
    """Read local file lines using Copilot tool semantics."""
    limit = min(limit, 10000)
    file_type = _detect_local_file_type(file_path)

    if file_type in ("json_array", "json_object"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if file_type == "json_object":
            data = [{"key": k, "value": v} for k, v in data.items()]
        total_lines = len(data)
        start_idx = max(0, start - 1)
        end_idx = min(end if end else total_lines, total_lines)
        end_idx = min(start_idx + limit, end_idx)
        return {
            "lines": data[start_idx:end_idx],
            "startLine": start_idx + 1,
            "endLine": end_idx,
            "totalLines": total_lines,
            "hasMore": end_idx < total_lines,
            "source": "bvbrc-workspace"
        }

    if file_type in ("csv", "tsv"):
        delimiter = "\t" if file_type == "tsv" else ","
        with open(file_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter=delimiter))
        total_lines = len(rows)
        start_idx = max(0, start - 1)
        end_idx = min(end if end else total_lines, total_lines)
        end_idx = min(start_idx + limit, end_idx)
        return {
            "lines": rows[start_idx:end_idx],
            "startLine": start_idx + 1,
            "endLine": end_idx,
            "totalLines": total_lines,
            "hasMore": end_idx < total_lines,
            "source": "bvbrc-workspace"
        }

    with open(file_path, "r", encoding="utf-8") as f:
        text_lines = f.readlines()
    total_lines = len(text_lines)
    start_idx = max(0, start - 1)
    end_idx = min(end if end else total_lines, total_lines)
    end_idx = min(start_idx + limit, end_idx)
    return {
        "lines": [line.rstrip("\n") for line in text_lines[start_idx:end_idx]],
        "startLine": start_idx + 1,
        "endLine": end_idx,
        "totalLines": total_lines,
        "hasMore": end_idx < total_lines,
        "source": "bvbrc-workspace"
    }

def _read_local_file_byte_range(file_path: str, start_byte: int, max_bytes: int) -> dict:
    """Read byte range from a local file and normalize output to workspace range shape."""
    if start_byte < 0:
        return {
            "error": True,
            "errorType": "INVALID_PARAMETERS",
            "message": "start_byte must be >= 0",
            "source": "bvbrc-workspace"
        }

    if max_bytes <= 0:
        return {
            "error": True,
            "errorType": "INVALID_PARAMETERS",
            "message": "max_bytes must be > 0",
            "source": "bvbrc-workspace"
        }

    max_bytes = min(max_bytes, 1024 * 1024)
    total_size = os.path.getsize(file_path)

    with open(file_path, "rb") as f:
        f.seek(start_byte)
        chunk = f.read(max_bytes)

    try:
        data = chunk.decode("utf-8")
    except UnicodeDecodeError:
        base64_content = base64.b64encode(chunk).decode("utf-8")
        data = f"<base64_encoded_data>{base64_content}</base64_encoded_data>"

    bytes_read = len(chunk)
    next_start = start_byte + bytes_read
    is_complete = next_start >= total_size

    return {
        "data": data,
        "start_byte": start_byte,
        "bytes_read": bytes_read,
        "total_size": total_size,
        "is_complete": is_complete,
        "requested_max_bytes": max_bytes,
        "next_start_byte": next_start if not is_complete else None,
        "source_type": "local",
        "source": "bvbrc-workspace"
    }

def register_workspace_tools(
    mcp: FastMCP,
    api: JsonRpcCaller,
    token_provider: TokenProvider,
    file_utilities_config: Optional[dict] = None
):
    """Register workspace tools with the FastMCP server"""
    
    @mcp.tool()
    async def workspace_browse_tool(
        token: Optional[str] = None,
        path: Optional[str] = None,
        name_contains: Optional[Union[str, List[str]]] = None,
        file_extensions: Optional[Union[str, List[str]]] = None,
        workspace_types: Optional[Union[str, List[str]]] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        num_results: Optional[int] = 50
    ) -> dict:
        """Find and list files in the user's personal workspace by name, type, or extension.

        Args:
            token: Authentication token (optional - will use default if not provided)
            path: Path to inspect/search. IMPORTANT PATH FORMAT:
                - Absolute paths MUST start with /<user_id>/home (e.g., /<user_id>/home or /<user_id>/home/subfolder)
                - Relative paths (e.g., "subfolder") are resolved relative to /{user_id}/home
                - DO NOT include /workspace prefix - paths should be /{user_id}/home or relative paths
                - If path is not provided or empty, defaults to user's home directory
                - Examples: "/<user_id>/home", "subfolder", "/<user_id>/home/Genome Groups"
            This tool always uses automatic mode selection:
                - Uses recursive search when filters are provided.
                - Uses one-level listing when no filters are provided.
            name_contains: Literal substrings that must appear in the filename (AND logic).
                ONLY use this for actual filename text the user specifies (e.g., "sample1", "ecoli").
                NEVER put file type categories here — use workspace_types or file_extensions instead.
                For example, if the user asks "find my reads files", do NOT set name_contains to "reads";
                instead set workspace_types to ["reads"]. Only set name_contains if the user explicitly
                mentions a filename or filename fragment they want to match.
            file_extensions: File extensions to match (OR logic). Example: ["fastq", "fq"] finds .fastq OR .fq files.
                IMPORTANT: Do NOT combine file_extensions with workspace_types for the same file category.
                Use ONE approach: either workspace_types OR file_extensions, not both together.
                For example, to find reads files, use workspace_types: ["reads"] alone — do NOT also add
                file_extensions: ["fastq", "fq"] because that creates an overly restrictive AND filter.
            workspace_types: Workspace object types to match (OR logic). This is the PREFERRED way to
                find files by category/type. Supported types:
                csv, diffexp_input_data, diffexp_input_metadata, doc, docx, embl,
                feature_dna_fasta, feature_protein_fasta, genbank_file, gff, gif, graph, jpg,
                json, nwk, pdf, phyloxml, png, pdb, ppt, pptx, reads, string, svg, tar_gz, tbi,
                tsv, txt, unspecified, vcf, vcf_gz, wig, xls, xlsx, xml.
                When the user asks for files of a particular type (e.g., "find my reads files",
                "show me contigs"), use ONLY workspace_types. Do not also set name_contains or
                file_extensions for the same concept.
            sort_by: Optional sort field. Valid options: creation_time, name, size, type.
            sort_order: Optional sort direction. Valid options: asc, desc.
            num_results: Maximum number of results to return. Defaults to 50.

        DO NOT USE THIS TOOL FOR:
            - Listing or searching genome groups (use list_genome_groups / get_genome_group)
            - Listing or searching feature groups (use list_feature_groups / get_feature_group)
            - File content retrieval (use read_file_bytes_tool)
            - Detailed single-file metadata inspection (use get_file_metadata)
        """
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace"
            }

        print(f"Inputs:\npath: {path}\nname_contains: {name_contains}\nfile_extensions: {file_extensions}\nworkspace_types: {workspace_types}\nsort_by: {sort_by}\nsort_order: {sort_order}\nnum_results: {num_results}")

        user_id = extract_userid_from_token(auth_token)
        sanitized_path = _sanitize_workspace_browse_path(path, user_id)
        resolved_path = resolve_relative_path(sanitized_path, user_id)

        effective_num_results = 50 if num_results is None else num_results
        normalized_name_contains = _normalize_string_or_list(name_contains)
        normalized_workspace_types = _normalize_string_or_list(workspace_types)
        normalized_extensions_raw = _normalize_string_or_list(file_extensions)
        normalized_extensions = [ext.lstrip(".") for ext in (normalized_extensions_raw or []) if ext]

        invalid_workspace_types = sorted(
            set(normalized_workspace_types or []) - set(SUPPORTED_WORKSPACE_TYPES)
        )
        if invalid_workspace_types:
            return {
                "error": "Invalid workspace_types value(s)",
                "errorType": "INVALID_PARAMETERS",
                "details": {
                    "invalid_workspace_types": invalid_workspace_types,
                    "supported_workspace_types": SUPPORTED_WORKSPACE_TYPES
                },
                "source": "bvbrc-workspace"
            }

        print(
            f"Browsing workspace path: {resolved_path}, user_id: {user_id}, original_path: {path}, "
            f"name_contains: {normalized_name_contains}, file_extensions: {normalized_extensions}, "
            f"workspace_types: {normalized_workspace_types}, sort_by: {sort_by}, sort_order: {sort_order}, "
            f"num_results: {effective_num_results}",
            file=sys.stderr
        )
        return await workspace_browse(
            api=api,
            token=auth_token,
            path=resolved_path,
            filename_search_terms=normalized_name_contains,
            file_extension=normalized_extensions,
            file_types=normalized_workspace_types,
            sort_by=sort_by,
            sort_order=sort_order,
            num_results=effective_num_results,
            tool_name="workspace_browse_tool"
        )

    # LEAVE HERE FOR NOW, DO NOT REMOVE
    #@mcp.tool(annotations={"readOnlyHint": True})
    async def get_file_metadata(
        token: Optional[str] = None,
        path: Optional[str] = None,
        session_id: Optional[str] = None,
        file_id: Optional[str] = None,
        include_internal_fields: bool = False
    ) -> dict:
        """Get normalized metadata for one file (workspace path or local session file).

        Resolution order:
        1) If session_id and file_id are provided, resolve local session file metadata.
        2) Else if path is provided and points to an existing local file under session_base_path, return local metadata.
        3) Else if path is provided, resolve as workspace path and return workspace metadata.

        DO NOT USE THIS TOOL FOR:
        - Listing directories or searching for files (use workspace_browse_tool)
        - Reading file content bytes/text (use read_file_bytes_tool/read_file_lines)

        Response behavior:
        - By default this tool returns user-safe metadata only.
        - Internal identifiers/paths (e.g. session_id, file_id, absolute path, workspace_path)
          are excluded unless include_internal_fields=True.
        """
        # Local session-file mode
        if session_id and file_id:
            try:
                record = _get_registered_file_record(session_id, file_id, file_utilities_config or {})
                file_path = _resolve_local_file_path(session_id, file_id, file_utilities_config)
                if not os.path.exists(file_path):
                    return {
                        "error": "Local file not found",
                        "errorType": "FILE_NOT_FOUND",
                        "details": {"session_id": session_id, "file_id": file_id},
                        "source": "bvbrc-workspace"
                    }
                metadata = _build_local_metadata(
                    file_path,
                    session_id=session_id,
                    file_id=file_id,
                    registry_record=record
                )
                return metadata if include_internal_fields else _sanitize_metadata_for_assistant(metadata)
            except Exception as e:
                return {
                    "error": f"Error getting local file metadata: {str(e)}",
                    "errorType": "PROCESSING_ERROR",
                    "source": "bvbrc-workspace"
                }

        if not path:
            return {
                "error": "Provide either (session_id and file_id) or path",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }

        # Local absolute-path mode (restricted to configured session_base_path)
        if os.path.isabs(path) and os.path.exists(path):
            base_path = (file_utilities_config or {}).get("session_base_path")
            if not base_path:
                return {
                    "error": "Local path metadata requires file_utilities.session_base_path to be configured",
                    "errorType": "INVALID_PARAMETERS",
                    "source": "bvbrc-workspace"
                }
            if not _is_within_base_path(path, base_path):
                return {
                    "error": "Local path is outside configured session base path",
                    "errorType": "INVALID_PARAMETERS",
                    "details": {"path": path},
                    "source": "bvbrc-workspace"
                }
            try:
                metadata = _build_local_metadata(path)
                return metadata if include_internal_fields else _sanitize_metadata_for_assistant(metadata)
            except Exception as e:
                return {
                    "error": f"Error getting local file metadata: {str(e)}",
                    "errorType": "PROCESSING_ERROR",
                    "source": "bvbrc-workspace"
                }

        # Workspace-path mode
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace"
            }

        user_id = extract_userid_from_token(auth_token)
        resolved_path = resolve_relative_path(path, user_id)
        print(f"Getting metadata for workspace path: {resolved_path}, user_id: {user_id}", file=sys.stderr)

        result = await workspace_get_file_metadata(api, resolved_path, auth_token)
        if "error" in result:
            return result
        metadata = _build_workspace_metadata(result, resolved_path)
        return metadata if include_internal_fields else _sanitize_metadata_for_assistant(metadata)

    # @mcp.tool()
    async def workspace_download_file_tool(token: Optional[str] = None, path: str = None, output_file: Optional[str] = None, return_data: bool = False) -> dict:
        """Download a file from the workspace.

        Args:
            token: Authentication token (optional - will use default if not provided)
            path: Path to the file to download (relative to user's home directory).
            output_file: Optional name and path of the file to save the downloaded content to.
                        If not provided and return_data is False, file data will be returned directly.
            return_data: If True, return the file data directly (base64 encoded for binary files, text for text files).
                        If False and output_file is provided, only write to file. If False and output_file is None,
                        returns file data (default behavior).
        """
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace"
            }

        # Extract user_id from token for path resolution and logging
        user_id = extract_userid_from_token(auth_token)
        resolved_path = resolve_relative_path(path, user_id)

        print(f"Downloading file from path: {resolved_path}, user_id: {user_id}, output_file: {output_file}, return_data: {return_data}")

        result = await workspace_download_file(api, resolved_path, auth_token, output_file, return_data)
        return result

    @mcp.tool(name="read_file_bytes_tool", annotations={"readOnlyHint": True})
    async def read_file_bytes_tool(
        token: Optional[str] = None,
        path: Optional[str] = None,
        session_id: Optional[str] = None,
        file_id: Optional[str] = None,
        start_byte: int = 0,
        max_bytes: int = 8192
    ) -> dict:
        """Read a byte window from local session files or workspace files.

        This is the canonical tool for file-byte access:
        - For quick preview, use defaults (start_byte=0, max_bytes=8192)
        - For paging, set start_byte to the next_start_byte value from the previous response
        - For local session files, provide file_id (session_id is auto-injected)
        - For workspace files, provide path

        IMPORTANT - Paging guidance:
        - The response includes is_complete (bool) and next_start_byte (int or null).
        - If is_complete is true, you have read the entire file. Do NOT call this tool
          again with the same path/file_id.
        - If is_complete is false, use next_start_byte as start_byte to read the next window.
        - total_size (when available) shows the full file size in bytes.

        Args:
            token: Authentication token (optional - will use default if not provided)
            path: Workspace path (e.g. /user@domain/home/file.txt) for reading files from
                the user's cloud workspace.
            session_id: Local Copilot session ID (auto-injected by system; do NOT set).
            file_id: Local Copilot file ID (use for reading local session files).
            start_byte: Zero-based starting byte offset (default: 0).
            max_bytes: Maximum bytes to read (default: 8192, max: 1048576).
        """
        try:
            # Local session file mode: only enter when file_id is explicitly provided.
            # Note: session_id is auto-injected by the system on every call, so we
            # cannot use its presence alone to decide the routing.
            if file_id:
                if not session_id:
                    return {
                        "error": True,
                        "errorType": "INVALID_PARAMETERS",
                        "message": "Provide both session_id and file_id for local file reads",
                        "source": "bvbrc-workspace"
                    }

                print(
                    f"Reading local file bytes from session file {file_id} in session {session_id}: "
                    f"start_byte={start_byte}, max_bytes={max_bytes}",
                    file=sys.stderr
                )
                file_path = _resolve_local_file_path(session_id, file_id, file_utilities_config)
                if not os.path.exists(file_path):
                    return {
                        "error": True,
                        "errorType": "FILE_NOT_FOUND",
                        "message": f"File {file_id} not found",
                        "details": {"fileId": file_id, "session_id": session_id},
                        "source": "bvbrc-workspace"
                    }

                result = _read_local_file_byte_range(file_path, start_byte, max_bytes)
                if result.get("error"):
                    return result
                result["session_id"] = session_id
                result["file_id"] = file_id
                return result

            if not path:
                return {
                    "error": True,
                    "errorType": "INVALID_PARAMETERS",
                    "message": "Provide either (session_id and file_id) or path",
                    "source": "bvbrc-workspace"
                }

            # Local absolute-path mode (restricted to configured session_base_path)
            if os.path.isabs(path) and os.path.exists(path):
                base_path = (file_utilities_config or {}).get("session_base_path")
                if not base_path:
                    return {
                        "error": True,
                        "errorType": "INVALID_PARAMETERS",
                        "message": "Local path reads require file_utilities.session_base_path to be configured",
                        "source": "bvbrc-workspace"
                    }
                if not _is_within_base_path(path, base_path):
                    return {
                        "error": True,
                        "errorType": "INVALID_PARAMETERS",
                        "message": "Local path is outside configured session base path",
                        "details": {"path": path},
                        "source": "bvbrc-workspace"
                    }

                print(
                    f"Reading local file bytes from path: {path}, start_byte={start_byte}, max_bytes={max_bytes}",
                    file=sys.stderr
                )
                return _read_local_file_byte_range(path, start_byte, max_bytes)

            auth_token = token_provider.get_token(token)
            if not auth_token:
                return {
                    "error": "No authentication token available",
                    "errorType": "AUTHENTICATION_ERROR",
                    "source": "bvbrc-workspace"
                }

            user_id = extract_userid_from_token(auth_token)
            resolved_path = resolve_relative_path(path, user_id)

            print(
                f"Reading workspace file bytes from path: {resolved_path}, user_id: {user_id}, "
                f"start_byte: {start_byte}, max_bytes: {max_bytes}"
            )

            result = await workspace_read_range(
                api=api,
                path=resolved_path,
                token=auth_token,
                start_byte=start_byte,
                max_bytes=max_bytes
            )
            if isinstance(result, dict):
                result.setdefault("source_type", "workspace")
                result.setdefault("workspace_path", resolved_path)
            return result
        except Exception as e:
            return {
                "error": True,
                "errorType": "PROCESSING_ERROR",
                "message": f"Error reading file bytes: {str(e)}",
                "source": "bvbrc-workspace"
            }

    @mcp.tool(annotations={"readOnlyHint": True})
    def read_file_lines(
        session_id: str,
        file_id: str,
        start: int = 1,
        end: Optional[int] = None,
        limit: int = 1000
    ) -> dict:
        """Read line ranges from a local Copilot session file.

        Use this for structured/text line access in session-local files.
        Do not use for workspace byte paging; use read_file_bytes_tool.
        """
        print(
            f"Reading lines from local file {file_id} in session {session_id}: start={start}, end={end}, limit={limit}",
            file=sys.stderr
        )
        try:
            if not session_id or not file_id:
                return {
                    "error": True,
                    "errorType": "INVALID_PARAMETERS",
                    "message": "Missing required parameters: session_id and file_id",
                    "source": "bvbrc-workspace"
                }

            file_path = _resolve_local_file_path(session_id, file_id, file_utilities_config)
            if not os.path.exists(file_path):
                return {
                    "error": True,
                    "errorType": "FILE_NOT_FOUND",
                    "message": f"File {file_id} not found",
                    "details": {"fileId": file_id, "session_id": session_id},
                    "source": "bvbrc-workspace"
                }

            return _read_local_file_lines(file_path, start, end, limit)
        except Exception as e:
            return {
                "error": True,
                "errorType": "PROCESSING_ERROR",
                "message": f"Error reading file lines: {str(e)}",
                "source": "bvbrc-workspace"
            }

    @mcp.tool()
    async def workspace_upload(token: Optional[str] = None, filename: str = None, upload_dir: str = None) -> dict:
        """Create an upload URL for a file in the workspace.

        Args:
            token: Authentication token (optional - will use default if not provided)
            filename: Name of the file to create upload URL for.
            upload_dir: Directory to upload the file to (relative to user's home directory, defaults to user's home directory).
        """
        if not filename:
            return {
                "error": "filename parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }

        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace"
            }

        # Extract user_id from token for path resolution and logging
        user_id = extract_userid_from_token(auth_token)
        if not upload_dir:
            upload_dir = get_user_home_path(user_id)
        else:
            # If upload_dir is provided and doesn't start with /, treat as relative to home
            if not upload_dir.startswith('/') and user_id:
                upload_dir = f"{get_user_home_path(user_id)}/{upload_dir}"

        print(f"Uploading file: {filename}, user_id: {user_id}, upload_dir: {upload_dir}")

        result = await workspace_upload_func(api, filename, upload_dir, auth_token)
        return result

    # ──────────────────────────────────────────────────────────────────
    # Group tools (create/get genome & feature groups) have been moved
    # to tools/group_tools.py.  Use the dedicated group tools:
    #   list_genome_groups, get_genome_group, create_genome_group
    #   list_feature_groups, get_feature_group, create_feature_group
    # ──────────────────────────────────────────────────────────────────