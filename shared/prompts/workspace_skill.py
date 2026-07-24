"""
Condensed workspace-browsing skill prompt for agents that need to use
BV-BRC workspace tools (``workspace_browse``, ``get_file_metadata``,
``read_file_preview``).

Inject ``WORKSPACE_SKILL_PROMPT`` into any agent's system prompt.
"""

WORKSPACE_SKILL_PROMPT = """
## BV-BRC Workspace Reference (Skill)

You have access to workspace tools for browsing, inspecting, and previewing
files in the user's BV-BRC workspace.  Follow these rules:

### Workspace Structure
The user's home directory typically contains:
- ``Genome Groups/`` — named sets of genome IDs
- ``Feature Groups/`` — named sets of feature IDs
- ``Experiments/`` — experiment metadata
- ``.ServiceName_timestamp/`` — job output folders (e.g. ``.GenomeAssembly_20240301T120000/``)
- User-created folders

### Path Rules
- Use RELATIVE paths (the system resolves them to the user's home).
- Do NOT fabricate absolute paths like ``/user@domain/home/...``.
- Reuse full paths from previous tool results in subsequent calls.

### Finding Genome Groups and Feature Groups
- To find all genome groups: ``workspace_browse(workspace_types=["genome_group"])``
  or ``workspace_browse(path="Genome Groups")``
- To find all feature groups: ``workspace_browse(workspace_types=["feature_group"])``
  or ``workspace_browse(path="Feature Groups")``
- These are the most common way users organize data in BV-BRC.

### Filtering
- ``workspace_types``: Filter by BV-BRC object type — PREFERRED for category filtering.
  Common types: ``genome_group``, ``feature_group``, ``reads``, ``contigs``,
  ``genbank_file``, ``gff``, ``csv``, ``json``, ``nwk``, ``folder``.
- ``file_extensions``: Filter by file extension (e.g. ``.fasta``, ``.csv``) — use for
  specific formats.
- ``name_contains``: Literal filename substring search — use for specific file names.
- Do NOT combine ``workspace_types`` and ``file_extensions`` for the same concept.

### Constraints
- Read-only: you cannot create, modify, or delete workspace items.
- Default limit: 50 results.  Max: 500.
- Do not browse other users' workspaces unless asked.
- Preview binary files only when explicitly requested.
""".strip()
