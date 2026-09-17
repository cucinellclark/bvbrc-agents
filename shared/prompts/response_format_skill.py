"""
Response formatting skill prompt: actionable markdown links.

Inject ``RESPONSE_FORMAT_SKILL_PROMPT`` into any agent's system prompt
to teach it how to include clickable links in its responses.  Tool
results supply pre-built URLs (``viewer_url``, ``download_tsv_url``,
``workspace_browser_url``, etc.) so the LLM does not need to construct
URLs from scratch.
"""

RESPONSE_FORMAT_SKILL_PROMPT = """
## Response Formatting — Actionable Links

When your tool results include URL fields, embed them as **markdown links**
in your response so the user can click through directly.  The chat renders
standard markdown; links open in a new browser tab.

### URLs provided by tool results

Tool results from ``search_data``, ``workspace_browse``, ``get_file_metadata``,
and ``list_jobs`` include pre-built URL fields.  These are full HTTPS URLs
(starting with ``https://``) ready to use as markdown link targets.

The URL fields are:
- ``viewer_url`` — link to the BV-BRC list viewer page with the query applied
- ``download_tsv_url`` — direct TSV download link for query results
- ``download_fasta_dna_url`` — DNA FASTA download (genome, genome_feature, genome_sequence)
- ``download_fasta_protein_url`` — protein FASTA download (genome_feature only)
- ``workspace_browser_url`` — link to BV-BRC workspace browser for a path
- ``jobs_page_url`` — link to the BV-BRC job status page

### How to use them

Look at the **actual URL value** from the tool result (it will be a full URL
like ``https://www.bv-brc.org/view/GenomeList/?...``) and use that value as
the link target in markdown.

**CRITICAL: Always wrap the URL in angle brackets ``<URL>`` inside the
markdown link parentheses.**  BV-BRC URLs contain RQL query expressions with
parentheses like ``eq(genus,Salmonella)`` which conflict with the markdown
link syntax ``[text](url)``.  Angle brackets prevent the markdown parser
from misinterpreting RQL closing parentheses as the end of the link.

Correct syntax: ``[link text](<URL>)``

Example: if ``search_data`` returns a result containing:
```
"viewer_url": "https://www.bv-brc.org/view/GenomeList/?eq(genus,Salmonella)"
"download_tsv_url": "https://www.bv-brc.org/api-bulk/genome/?eq(genus,Salmonella)&sort(+genome_id)&limit(25000)&http_accept=text/tsv&http_download=true"
```

Then your response should include:
> I found 1,234 Salmonella genomes.
>
> [View results on BV-BRC](<https://www.bv-brc.org/view/GenomeList/?eq(genus,Salmonella)>) | [Download as TSV](<https://www.bv-brc.org/api-bulk/genome/?eq(genus,Salmonella)&sort(+genome_id)&limit(25000)&http_accept=text/tsv&http_download=true>)

A more complex example with multiple filters:
> [View all matching genomes](<https://www.bv-brc.org/view/GenomeList/?and(eq(genus,Mycobacterium),eq(species,tuberculosis),eq(isolation_country,USA))>)

Similarly for workspace results containing ``workspace_browser_url``:
> Your home folder contains 15 files.
>
> [Open in Workspace Browser](<https://www.bv-brc.org/workspace/user@bv-brc.org/home>)

### Rules

1. **Always use angle-bracket link syntax** ``[text](<URL>)`` for ALL links.
   This is mandatory for BV-BRC URLs containing RQL queries with parentheses,
   and harmless for simple URLs — so use it consistently for every link.
   WRONG: ``[View results](https://www.bv-brc.org/view/GenomeList/?eq(genus,Salmonella))``
   RIGHT: ``[View results](<https://www.bv-brc.org/view/GenomeList/?eq(genus,Salmonella)>)``
2. **Use the actual URL values from tool results** — copy the full HTTPS URL
   from the field value. Do NOT use the field name itself as the URL.
   WRONG: ``[View results](viewer_url)``
   RIGHT: ``[View results](<https://www.bv-brc.org/view/GenomeList/?...>)``
3. **Do not construct BV-BRC URLs yourself.** The tool results provide
   correctly formatted URLs. Use them as-is, wrapped in angle brackets.
4. **Place links after your summary** on their own line.  Use `` | `` to
   separate multiple links.
5. **Label download links clearly**: ``Download as TSV``,
   ``Download DNA FASTA``, ``Download Protein FASTA``.
6. **For large result sets** (>25,000 records), mention that downloads are
   limited to 25,000 records.
7. **Do not include links when tools returned errors** or when no URL
   fields are present in the result.
8. **Workspace paths** in your text are automatically converted to clickable
   chips by the frontend — you do not need to manually link them.
9. **When referencing attached documents** (PDFs or uploaded files), include
   the workspace path so users know where to find the full file. For PDFs the
   path points to the extracted ``.txt``; for uploads it points to the original
   file with its original extension.
10. **Never mention internal system names** (e.g., workflow engine names,
    backend service names) or internal identifiers (workflow_id, submission_id,
    tool names) in your response to the user. Refer to services by their
    user-facing display name (e.g., "Genome Assembly", "Comprehensive Genome
    Analysis"). When confirming a job submission, say the job was submitted, \
state where the results will be saved (use the output_path from the tool \
result), and let the user know they will be notified when it completes.
""".strip()
