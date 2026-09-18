"""
Condensed skill prompt for agents that create genome/feature groups or
submit workflows with SRA inputs.

Inject ``GROUPS_SRA_SKILL_PROMPT`` into any agent's system prompt.
"""

GROUPS_SRA_SKILL_PROMPT = """
## Groups and SRA Inputs Reference (Skill)

### Genome and Feature Groups

- Discover existing groups with ``list_groups`` or
  ``workspace_browse(workspace_types=["genome_group"])`` /
  ``workspace_browse(workspace_types=["feature_group"])``.
- Do NOT fabricate group paths like ``/user@domain/home/Genome Groups/Guessed Name``.
  Browse first, then copy the real path from the result.
- ``create_group`` creates a group, and via ``if_exists`` can also **append
  to** or **replace** an existing one (the same read-merge-overwrite the
  website does — there is no other update API):
  - "add / put these genomes in my group X", "extend group X" →
    ``if_exists="append"`` (duplicates are skipped; result reports
    ``added`` / ``already_present`` / ``count``).
  - "replace / overwrite group X" (explicit) → ``if_exists="replace"``.
  - Otherwise leave the default ``"error"``: if the name is taken the tool
    returns ``errorType: ALREADY_EXISTS`` and changes nothing. Then ask the
    user whether to append, replace, or use a new name — do NOT retry with a
    different query or a different name on your own.
- If ``create_group`` returns any error, report it. Never retry it with a
  changed query; a failure is never fixed by a different filter.
- In PLAN mode (see ``=== EXECUTION MODE ===``) ``create_group`` is refused
  with ``blocked_by_mode``. Do not retry it: report the query you would use,
  the matching count from ``search_data``/``facet_query``, and the intended
  group name, then tell the user to switch to Execute mode to create it.
  Never say the group was created.
- ``create_group`` parameters:
  - ``group_type``: ``"genome_group"`` or ``"feature_group"``
  - ``collection``: ``"genome"`` for genome groups, ``"genome_feature"`` for
    feature groups
  - ``query``: Solr query string (same syntax as ``search_data``)
  - ``limit``: Max IDs to include (default 500)
  - ``if_exists``: ``"error"`` (default) | ``"append"`` | ``"replace"``
- Before creating a group, use ``search_data`` with ``count_only=true`` to
  check how many records match. Tell the user the total count and the limit
  being applied.
- Always create **one** group unless the user explicitly asks for multiple.
  Never split "5 of A and 5 of B" into two separate groups to work around
  tool limitations.
- **Exact mixed subsets** (e.g. "5 of serovar A and 5 of serovar B in one
  group"): a single OR query with ``limit=N+M`` does NOT balance subsets —
  Solr returns whatever ranks first. Instead:
  1. ``search_data`` for subset A with the user's filters, ``limit`` = the
     requested count per subset.
  2. ``search_data`` for subset B, same approach.
  3. If a subset returns fewer IDs than requested, inform the user and ask
     whether to proceed with what is available or adjust.
  4. **One** ``create_group`` with
     ``query: genome_id:(id1 OR id2 OR ... OR idN)`` built from the
     combined IDs, and ``limit`` >= the total ID count.
  5. Reply with the group **name, workspace path, total count, and
     per-subset counts**.
- For homogeneous groups ("save these search results as a group"), the
  normal one-step flow is correct: count first, then ``create_group`` with
  the Solr query directly (no need to fetch IDs first).
- Do NOT re-run the same query under a ``_fixed`` / ``_adjusted`` name.
  That produces the same ranked set. If the user asks to "fix" or "adjust"
  a group, either ``if_exists="replace"`` with the corrected IDs (after the
  user agrees) or create a **new** group and say the old one is unchanged.
- When passing a group to a workflow, copy the full workspace path returned
  by ``create_group`` or ``workspace_browse`` — do not construct it.

### SRA Inputs

- Call ``get_sra_metadata`` when the user provides SRR/ERR/DRR accessions.
- SRA field names vary by workflow — always prefer field names from
  ``get_workflow_inputs`` over this cheat sheet. Common patterns:
  - **Most assembly-style workflows** (GenomeAssembly, ComprehensiveGenomeAnalysis,
    Variation, TaxonomicClassification, SARS2Assembly):
    ``srr_ids`` — a **string list** (e.g. ``["SRR123", "SRR456"]``).
  - **RNASeq / SARS2Wastewater-style workflows**:
    ``srr_libs`` — an **array of records**, each with ``srr_accession`` (string)
    and ``sample_id`` (string). **NOT** ``srr_id``.
  - **MetagenomeBinning**:
    ``srr_ids`` — a **singular string** (not a list).
- Never guess SRA field names. If the workflow schema uses a name you have
  not seen before, use that name exactly.
""".strip()
