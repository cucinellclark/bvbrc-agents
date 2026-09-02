"""Condensed literature search skill prompt.

Injected into agents that may need to search published scientific
literature as part of broader work (planning, data).  The helpdesk
agent gets its own full ``## Literature Search`` section instead.
"""

LITERATURE_SKILL_PROMPT = """\
## Literature Search Reference (Skill)

You have access to ``search_literature`` for querying published scientific
papers. Use it when you need evidence from the research literature — e.g.
published studies about an organism, gene, protein, or disease.

### When to use
- The user asks about published research, papers, or evidence
- You need to ground a recommendation in published findings
- The user references a paper or asks "what does the research say about X"

### How to use
- ``search_literature(query="natural language query", top_k=10)``
- Returns source passages with bibliographic metadata (titles, scores,
  document IDs)
- Present results with paper titles and key findings

### Do NOT use for
- BV-BRC platform documentation (use ``query_helpdesk`` instead)
- Searching BV-BRC data collections (use ``search_data`` instead)
"""
