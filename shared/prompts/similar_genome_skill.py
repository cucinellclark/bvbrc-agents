"""Condensed similar genome finder skill prompt.

Injected into all agents so any agent that receives a "find similar
genomes" query knows to call ``find_similar_genomes`` directly (not a
Solr query or GoWe workflow) and understands parameter tuning for
broad searches.
"""

SIMILAR_GENOME_SKILL_PROMPT = """\
## Similar Genome Finder Reference (Skill)

For "find similar genomes", "closest genomes", "genome distance", or
"similar genome finder" questions, call ``find_similar_genomes`` directly.
This is NOT a Solr query and NOT a GoWe workflow — it uses the BV-BRC
MinHash service and returns results immediately (no job submission needed).

### Inputs
Provide exactly ONE of:
- ``genome_id`` — a BV-BRC genome ID (e.g. ``83332.12``)
- ``fasta_file`` — a full workspace path to a FASTA/contigs file

### Parameter tuning
The defaults (``max_distance=0.01``, ``max_pvalue=0.01``) are very
restrictive (>99% sequence identity). When the user asks for many hits
(e.g., 100+), searches all public genomes (``scope="all"``), or queries
viral genomes, increase ``max_distance`` to **0.5** and ``max_pvalue``
to **0.1**. Otherwise the search may return zero results.

### Timeout
Broad searches (``scope="all"``, large ``max_hits``) can take up to two minutes.
Do not retry a timed-out search with the same parameters — narrow the scope or
reduce ``max_hits``, or tell the user the service is slow right now.

### Do NOT use for
- BLAST sequence similarity searches (use a BLAST workflow)
- Phylogenetic tree building (use a tree workflow)
- Genome annotation or assembly (use the appropriate workflow)
"""
