"""
BV-BRC Genome & Feature Group MCP Tools

Six first-class tools for managing genome and feature groups:

  list_genome_groups   – List all genome groups in the user's workspace
  list_feature_groups  – List all feature groups in the user's workspace
  get_genome_group     – Get genome IDs from a group by name
  get_feature_group    – Get feature IDs from a group by name
  create_genome_group  – Create a new genome group
  create_feature_group – Create a new feature group

Groups are identified by **name only**. The system automatically resolves
names to workspace paths using a two-tier strategy:
  1. Exact match in the default folder (e.g. /{user}/home/Genome Groups/)
  2. Fuzzy / case-insensitive match in the default folder

No workspace paths are exposed to the LLM.
"""

import sys
from fastmcp import FastMCP
from common.json_rpc import JsonRpcCaller
from common.token_provider import TokenProvider
from functions.group_functions import (
    list_groups,
    get_group_ids,
    create_group,
)
from typing import Optional, Union, List


def register_group_tools(
    mcp: FastMCP,
    api: JsonRpcCaller,
    token_provider: TokenProvider,
):
    """Register all genome/feature group tools with the FastMCP server."""

    # ------------------------------------------------------------------
    # LIST tools
    # ------------------------------------------------------------------

    @mcp.tool(name="list_genome_groups", annotations={"readOnlyHint": True})
    async def list_genome_groups(
        token: Optional[str] = None,
    ) -> dict:
        """List all genome groups in the user's workspace.

        Returns a list of genome group names, paths, and creation dates.
        Use this tool to discover which genome groups exist before
        retrieving their contents with get_genome_group.

        DO NOT USE workspace_browse_tool to find genome groups — use this tool instead.
        """
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace",
            }

        return await list_groups(api, "genome_group", auth_token, tool_name="list_genome_groups")

    @mcp.tool(name="list_feature_groups", annotations={"readOnlyHint": True})
    async def list_feature_groups(
        token: Optional[str] = None,
    ) -> dict:
        """List all feature groups in the user's workspace.

        Returns a list of feature group names, paths, and creation dates.
        Use this tool to discover which feature groups exist before
        retrieving their contents with get_feature_group.

        DO NOT USE workspace_browse_tool to find feature groups — use this tool instead.
        """
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace",
            }

        return await list_groups(api, "feature_group", auth_token, tool_name="list_feature_groups")

    # ------------------------------------------------------------------
    # GET tools
    # ------------------------------------------------------------------

    @mcp.tool(name="get_genome_group", annotations={"readOnlyHint": True})
    async def get_genome_group(
        genome_group_name: str = None,
        token: Optional[str] = None,
    ) -> dict:
        """Get genome IDs from a genome group by name.

        The group is looked up by name automatically — you do NOT need to
        provide a full workspace path. The system searches the user's
        default Genome Groups folder and will find the group even if the
        name casing doesn't match exactly.

        If the name is ambiguous (matches multiple groups), the tool
        returns a list of candidates so the user can clarify.

        Args:
            genome_group_name: Name of the genome group (e.g. "My E. coli genomes").
                               Do NOT provide a workspace path — just the name.
            token: Authentication token (auto-provided).

        Returns:
            Dictionary with genome_ids list and count,
            or an error/disambiguation response.
        """
        if not genome_group_name:
            return {
                "error": "genome_group_name parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace",
            }

        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace",
            }

        return await get_group_ids(api, genome_group_name, "genome_group", auth_token)

    @mcp.tool(name="get_feature_group", annotations={"readOnlyHint": True})
    async def get_feature_group(
        feature_group_name: str = None,
        token: Optional[str] = None,
    ) -> dict:
        """Get feature IDs from a feature group by name.

        The group is looked up by name automatically — you do NOT need to
        provide a full workspace path. The system searches the user's
        default Feature Groups folder and will find the group even if the
        name casing doesn't match exactly.

        If the name is ambiguous (matches multiple groups), the tool
        returns a list of candidates so the user can clarify.

        Args:
            feature_group_name: Name of the feature group (e.g. "AMR genes").
                                Do NOT provide a workspace path — just the name.
            token: Authentication token (auto-provided).

        Returns:
            Dictionary with feature_ids list and count,
            or an error/disambiguation response.
        """
        if not feature_group_name:
            return {
                "error": "feature_group_name parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace",
            }

        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace",
            }

        return await get_group_ids(api, feature_group_name, "feature_group", auth_token)

    # ------------------------------------------------------------------
    # CREATE tools
    # ------------------------------------------------------------------

    @mcp.tool(name="create_genome_group")
    async def create_genome_group(
        genome_group_name: str = None,
        genome_id_list: str = None,
        token: Optional[str] = None,
    ) -> dict:
        """Create a genome group in the user's workspace.

        The group is always created in the user's default Genome Groups
        folder. You only need to provide the name and the genome IDs.

        Args:
            genome_group_name: Name for the new genome group.
            genome_id_list: Comma-separated genome IDs to add.
                           Example: "83332.12,511145.12,386585.17"
            token: Authentication token (auto-provided).

        Returns:
            Dictionary with the created group's name, path, and count.
        """
        if not genome_group_name:
            return {
                "error": "genome_group_name parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace",
            }

        if not genome_id_list:
            return {
                "error": "genome_id_list parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace",
            }

        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace",
            }

        # Parse comma-separated string to list
        if isinstance(genome_id_list, str):
            parsed_ids = [gid.strip() for gid in genome_id_list.split(",") if gid.strip()]
        elif isinstance(genome_id_list, list):
            parsed_ids = [str(gid).strip() for gid in genome_id_list if gid]
        else:
            return {
                "error": f"genome_id_list must be a string or list, got {type(genome_id_list)}",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace",
            }

        if not parsed_ids:
            return {
                "error": "genome_id_list must contain at least one genome ID",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace",
            }

        print(f"create_genome_group: name={genome_group_name}, ids={len(parsed_ids)}", file=sys.stderr)
        return await create_group(api, genome_group_name, parsed_ids, "genome_group", auth_token)

    @mcp.tool(name="create_feature_group")
    async def create_feature_group(
        feature_group_name: str = None,
        feature_id_list: str = None,
        token: Optional[str] = None,
    ) -> dict:
        """Create a feature group in the user's workspace.

        The group is always created in the user's default Feature Groups
        folder. You only need to provide the name and the feature IDs.

        Args:
            feature_group_name: Name for the new feature group.
            feature_id_list: Comma-separated feature IDs to add.
                            Example: "fig|83332.12.peg.1,fig|83332.12.peg.2"
            token: Authentication token (auto-provided).

        Returns:
            Dictionary with the created group's name, path, and count.
        """
        if not feature_group_name:
            return {
                "error": "feature_group_name parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace",
            }

        if not feature_id_list:
            return {
                "error": "feature_id_list parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace",
            }

        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace",
            }

        # Parse comma-separated string to list
        if isinstance(feature_id_list, str):
            parsed_ids = [fid.strip() for fid in feature_id_list.split(",") if fid.strip()]
        elif isinstance(feature_id_list, list):
            parsed_ids = [str(fid).strip() for fid in feature_id_list if fid]
        else:
            return {
                "error": f"feature_id_list must be a string or list, got {type(feature_id_list)}",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace",
            }

        if not parsed_ids:
            return {
                "error": "feature_id_list must contain at least one feature ID",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace",
            }

        # Fix known LLM issue: consistently omits the final '.' in feature IDs
        # e.g. "PATRIC.123abc" should be "PATRIC.123.abc"
        fixed_ids = []
        for fid in parsed_ids:
            if len(fid) >= 4 and fid[-4] != ".":
                fid = fid[:-3] + "." + fid[-3:]
            fixed_ids.append(fid)

        print(f"create_feature_group: name={feature_group_name}, ids={len(fixed_ids)}", file=sys.stderr)
        return await create_group(api, feature_group_name, fixed_ids, "feature_group", auth_token)
