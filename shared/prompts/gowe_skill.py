"""
Condensed GoWe workflow-engine skill prompt for agents that need to
discover, configure, or submit GoWe workflows.

Inject ``GOWE_SKILL_PROMPT`` into any agent's system prompt.
"""

GOWE_SKILL_PROMPT = """
## GoWe Workflow Engine Reference (Skill)

You have access to GoWe tools for discovering and submitting bioinformatics
workflows.  Follow these rules:

### Workflow
1. Call ``list_gowe_workflows`` first to see all available workflows.
2. Identify matching workflow(s) based on the user's request.
3. Call ``get_workflow_inputs`` to get the input schema before populating.
4. Call ``submit_gowe_job`` to submit with the workflow_id and populated inputs.

### Rules
- NEVER guess or hardcode workflow names/IDs — always discover via ``list_gowe_workflows``.
- ALWAYS call ``get_workflow_inputs`` before populating inputs.
- Use schema defaults for missing non-required inputs.
- Skip system inputs (prefixed with ``_``).
- Auto-generate ``output_path`` and ``output_file`` if not specified by the user.
  The system automatically rewrites output paths to place results under the
  chat session's workspace folder. Just provide a descriptive folder name.

### File Path Format
- Plain workspace path strings only (e.g. ``/user@domain/home/folder/file.fasta``).
- No CWL ``File`` objects, no ``ws://`` or ``workspace:`` URI prefixes.

### Paired/Single-End Reads
- ``paired_end_libs``: array of ``{read1: str, read2: str, interleaved: bool}``
- ``single_end_libs``: array of ``{read: str}``
- Detect pairing from filename patterns: R1/R2, _1/_2, .1/.2
""".strip()
