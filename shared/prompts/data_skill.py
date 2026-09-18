"""
Condensed data-query skill prompt for agents that need to use BV-BRC
Solr search tools (``search_data``, ``facet_query``, ``probe_data``).

Inject ``DATA_SKILL_PROMPT`` into any agent's system prompt to give it
the knowledge needed to construct valid Solr queries.  This is a
compact version of the full data agent system prompt — it covers syntax,
relationships, and constraints but omits strategy/efficiency sections.
"""

DATA_SKILL_PROMPT = """
## BV-BRC Data Query Reference (Skill)

You have access to ``search_data``, ``facet_query``, and ``probe_data`` tools
for querying BV-BRC's Solr collections.  Follow these rules:

### Query Syntax
- Exact match: ``field:value`` (e.g. ``genus:Salmonella``)
- Phrase match: ``field:"multi word value"`` — ALWAYS quote multi-word values
- Wildcards: ``field:Sal*`` (prefix only, no leading wildcards)
- OR group: ``field:(val1 OR val2 OR val3)``
- AND: ``field1:val1 AND field2:val2``
- NOT: ``NOT field:val``  or  ``field1:val1 AND NOT field2:val2``
- Numeric range: ``field:[10 TO 100]``  (inclusive) or ``field:{10 TO 100}`` (exclusive)
- Year range: ``completion_date:[2020 TO 2024]``

### Key Collections
- ``genome`` — bacterial/archaeal/viral genomes (has ``genome_id``, ``genome_name``,
  ``genus``, ``species``, ``host_name``, ``isolation_country``, ``reference_genome``, etc.)
  - ``reference_genome`` field: values are ``"Reference"``, ``"Representative"``, or
    empty/blank. To find reference/representative genomes, filter by
    ``reference_genome:(Reference OR Representative)``.  Do NOT use
    ``genome_status:Complete`` as a proxy — "Complete" means assembly status,
    NOT reference/representative designation.
- ``genome_feature`` — genes/proteins (has ``feature_id``, ``patric_id``,
  ``genome_id``, ``product``, ``gene``, ``plfam_id``, ``pgfam_id``, etc.)
  - ``plfam_id`` (PLFam) and ``pgfam_id`` (PGFam) are protein family assignments.
    Always include these in ``select`` when answering questions about protein
    families or family annotations. If a query result has empty ``plfam_id``/
    ``pgfam_id``, the annotation genuinely does not exist for that feature.
    But if the fields were not included in ``select``, they will be absent from
    the response — that does NOT mean the annotation is missing. Always
    explicitly request these fields before concluding "no family assignment."
- ``genome_amr`` — AMR phenotypes (has ``genome_id``, ``antibiotic``,
  ``resistant_phenotype``). One genome has many rows, so ``numFound`` is a
  record count, not a genome count. For "how many genomes are resistant to
  X" use ``facet_query(collection="genome_amr", query=..., facet_fields=
  ["genome_id"], count_distinct=true)`` and read ``distinct_counts.genome_id``.
  Never request a large ``facet_limit`` and count the values yourself.
- ``sp_gene`` — specialty genes (has ``genome_id``, ``property``, ``gene``,
  ``product``, ``source``, ``evidence``)
  - ``property`` values — these are DISTINCT categories, not interchangeable:
    - ``"Antibiotic Resistance"`` — acquired AMR genes (answer AMR questions)
    - ``"Virulence Factor"`` — virulence determinants (answer virulence questions)
    - ``"Antibiotic Target in Susceptible Species"`` — normal cellular drug
      targets (housekeeping genes like gyrA, rpoB); these are NOT resistance
      genes and NOT virulence factors. Do NOT present these as AMR or virulence
      results. If the user asks about AMR/virulence and you find only
      "Antibiotic Target" hits, explain clearly that these are drug targets in
      susceptible organisms, not resistance or virulence genes.
    - ``"Drug Target"`` — therapeutic drug targets
    - ``"Essential Gene"`` — genes essential for survival
    - ``"Human Homolog"`` — genes with human homologs
    - ``"Transporter"`` — membrane transporter genes
  - When the user asks about "AMR genes," query ``property:"Antibiotic Resistance"``.
    When they ask about "virulence factors," query ``property:"Virulence Factor"``.
    Do NOT conflate categories.
- ``pathway`` — metabolic pathways (has ``genome_id``, ``pathway_name``, ``pathway_id``)
- ``epitope`` — epitope data (has ``epitope_id``, ``organism``, ``protein_name``)

### Cross-Collection Queries
Collections are linked by ``genome_id``, ``feature_id``/``patric_id``, and ``taxon_id``.
Pattern: query collection A → extract IDs → filter collection B with those IDs.
Batch ``genome_id`` filters at max 50 IDs per query.

### Constraints
- Default limit: 25 results.  Use ``count_only=true`` before large fetches.
- ``genome_status`` values: Complete, WGS, Partial, Plasmid, Deprecated.
  ``genome_quality`` values: Good, Poor.
  For general queries, do NOT filter by these unless the user asks.
  For analysis or genome group creation, use your judgment: prefer
  Complete/WGS and Good quality genomes, avoid Deprecated genomes
  (prefer Partial over Deprecated), and relax filters if the result
  set would be too small. Always mention
  any genome_status or genome_quality filters you applied.
- If ``probe_data`` is available, use it first to discover correct field values
  before building a structured query.
- If a query returns 0 results, try broader terms or ``probe_data`` — never
  report "no results" after a single attempt.
""".strip()
