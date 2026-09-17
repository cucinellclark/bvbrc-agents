"""
Condensed workflow-engine skill prompt for agents that need to
discover, configure, or submit bioinformatics workflows.

Inject ``GOWE_SKILL_PROMPT`` into any agent's system prompt.
"""

GOWE_SKILL_PROMPT = """
## Workflow Engine Reference (Skill)

You have access to tools for discovering and submitting bioinformatics
workflows.  Follow these rules:

### Workflow
1. Call ``list_gowe_workflows`` first to see all available workflows.
2. Identify matching workflow(s) based on the user's request.
3. Call ``get_workflow_inputs`` to get the input schema before populating.
4. Call ``submit_gowe_job`` to submit with the workflow_id and populated inputs.

### Rules
- NEVER guess or hardcode workflow names/IDs — always discover via ``list_gowe_workflows``.
- ALWAYS call ``get_workflow_inputs`` before populating inputs.
- Keep the schema default for every parameter the user did not mention. Do NOT
  change analysis parameters (hit counts, thresholds, models, recipes) on your
  own. If you set any parameter to a non-default value, say so and why.
- Inputs whose doc says "[enum: ...]" and whose name ends in ``_source`` or
  ``_type`` are SELECTORS. The doc names the payload input each value requires
  ("Required when X=Y"). Set the selector AND its payload together. If the user
  has not given you anything usable for the payload, ASK.
- Identifiers must be verified, not assumed. SRA accessions look like
  SRR/ERR/DRR + digits — a workspace file name is NOT an accession.
- If no registered workflow does what the user asked, say so plainly and name
  the closest workflow and what it actually does.
- Skip system inputs (prefixed with ``_``).
- ALWAYS auto-generate ``output_path`` and ``output_file`` — never ask the user
  for these values and never list them as missing information. The system
  automatically rewrites output paths to place results under the chat session's
  workspace folder. Just provide a descriptive folder name.

### File Path Format
- Plain workspace path strings only (e.g. ``/user@domain/home/folder/file.fasta``).
- No CWL ``File`` objects, no ``ws://`` or ``workspace:`` URI prefixes.

### Read Library Inputs (Record-Typed)
- Record-typed inputs (``paired_end_libs``, ``single_end_libs``, etc.) carry a
  ``fields`` list in the ``get_workflow_inputs`` result. Use EXACTLY those field
  names — never add fields the schema does not list.
- Detect pairing from filename patterns: R1/R2, _1/_2, .1/.2

### SRA Inputs
- Call ``get_sra_metadata`` when the user provides SRR/ERR/DRR accessions.
- SRA field names vary by workflow — always prefer names from
  ``get_workflow_inputs``. Common patterns:
  - Assembly-style workflows: ``srr_ids`` as a **string list**.
  - RNASeq / SARS2Wastewater: ``srr_libs`` records with
    **``srr_accession``** and **``sample_id``** (NOT ``srr_id``).
  - MetagenomeBinning: ``srr_ids`` as a **singular string**.
- Never guess SRA field names — use what the schema provides.
""".strip()
