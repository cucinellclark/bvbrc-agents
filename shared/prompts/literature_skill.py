"""Condensed literature search skill prompt.

Injected into agents that may need to search published scientific
literature as part of broader work (planning, data).  The helpdesk
agent gets its own full ``## Literature Search`` section instead, which
reuses ``LITERATURE_CITATION_RULES`` so the two cannot drift apart.
"""

LITERATURE_CITATION_RULES = """\
### Searching
- Write focused queries: organism + gene/protein + phenomenon \
(e.g. "Klebsiella pneumoniae KPC carbapenemase plasmid spread"), not the \
user's whole sentence.
- If the first search misses part of the question, search again with \
different terms (synonyms, gene names, a narrower aspect) — at most 3–4 \
searches per question.
- Each result is a list of distinct ``papers`` with ``title``, ``doi`` / \
``url``, ``year``, ``journal``, ``authors`` (only when known) and \
``passages`` (the retrieved text). The corpus is ASM journals plus \
PubMed Central open-access articles — not all of PubMed.

### Citing
- Cite ONLY papers that appear in a ``search_literature`` result in this \
conversation. Give the title and a link (``url``, or the DOI); add authors, \
year and journal only when the result includes them. Never write author \
names, years, journals or DOIs that the result does not contain.
- Base each claim on the paper's ``passages``. If a statement comes from a \
paper's title alone, say so ("the title suggests …").
- Do not present well-known papers from your own memory as search results. \
If a key reference (e.g. the original description of a tool or standard) \
was not retrieved, you may name it in a separate line marked \
"Not in the search results — from general knowledge", with no DOI or \
author list.
- If the results do not answer the question, say that plainly, summarise \
what was found, and suggest a different or narrower search.
"""

LITERATURE_SKILL_PROMPT = f"""\
## Literature Search Reference (Skill)

You have access to ``search_literature`` for querying published scientific
papers. Use it when you need evidence from the research literature — e.g.
published studies about an organism, gene, protein, or disease.

### When to use
- The user asks about published research, papers, or evidence
- You need to ground a recommendation in published findings
- The user references a paper or asks "what does the research say about X"

### How to use
- ``search_literature(query="focused query", top_k=10)`` — ``top_k`` is
  the number of distinct papers (max 20)

{LITERATURE_CITATION_RULES}
### Do NOT use for
- BV-BRC platform documentation (use ``query_helpdesk`` instead)
- Searching BV-BRC data collections (use ``search_data`` instead)
"""
