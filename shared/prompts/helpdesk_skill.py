"""
Condensed helpdesk/documentation skill prompt for agents that need to
look up BV-BRC service parameters, usage guidance, or FAQ answers.

Inject ``HELPDESK_SKILL_PROMPT`` into any agent's system prompt.
This skill does NOT submit jobs — it only grounds explanations and
parameter choices in the helpdesk knowledge base.
"""

HELPDESK_SKILL_PROMPT = """
## BV-BRC Helpdesk Reference (Skill)

You have access to ``query_helpdesk``, ``list_services``, and
``get_service_schema`` tools for looking up BV-BRC documentation.

### When to Use
- If you are unsure how a BV-BRC service works, what parameters it
  accepts, or what valid values are, call ``query_helpdesk`` with a
  concise question before guessing.
- If you need the exact parameter schema for a service form field
  (e.g., valid recipe names, strategy options), call
  ``get_service_schema`` with the service name.
- If you need to confirm which services are available, call
  ``list_services``.

### Rules
- Do NOT invent service parameter names, valid values, or "how to"
  steps. Ground your answers in the retrieved documentation.
- These tools are read-only. They do NOT submit jobs or modify data.
- Keep queries concise — a short phrase describing what you need
  (e.g., "GenomeAssembly recipe options", "BLAST parameters").
- If the first query returns insufficient results, rephrase with
  different keywords before giving up.
- Stop querying as soon as you have enough information.
""".strip()
