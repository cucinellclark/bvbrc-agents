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
- ``chats/<session>/`` — session folder for this chat (uploads, PDF \
extracts, job outputs submitted from the chatbot). Use the path from \
``=== SESSION WORKSPACE ===`` to access it.
- User-created folders

### Chat Session vs Home — When to Search Where

**Session folder first** when the query is about this conversation's files:
- Uploads from this chat ("the FASTA I attached", "use that CSV")
- Jobs submitted from this chat ("the assembly results", "annotate those contigs")
- Vague references ("those files", "what we just generated")
- Chat-scoped "what files do I have"

**Home / named path** when the query is about the rest of the workspace:
- Genome Groups / Feature Groups
- A folder the user named ("in my Salmonella folder")
- Explicit whole-workspace search
- Page context that already points at a workspace path

**Fall back.** If the session folder is missing, empty, or does not contain \
the named item, browse home or search with ``name_contains`` and \
``recursive=true``. Do not tell \
the user a file is gone until both places have been checked.

Do not search other sessions under ``chats/`` unless the user asks.

### Path Rules — CRITICAL
- Use RELATIVE paths (the system resolves them to the user's home).
- Do NOT fabricate absolute paths like ``/user@domain/home/...``.
- Do NOT use ``"."`` or ``"./"`` as a path — use an empty string or omit \
the path parameter to browse the home directory.
- **Never construct a workspace path from the user's words.** If the user \
says "my assembly test folder", do NOT call a tool with \
``path="assembly test gowe"``. Instead, browse home first (empty path or \
``name_contains``) to discover the real folder name.
- Only use a path that appeared in a prior tool result **or** in the \
``=== SESSION WORKSPACE ===`` section. Copy it exactly.

### Discovering Files and Folders
- When the user mentions a folder or file by name, ALWAYS browse first to \
discover the actual path. Call ``workspace_browse()`` with no path (home) \
or search with ``name_contains=["keyword"]`` and ``recursive=true``.
- Once you find the item in the results, copy the full path from the result \
into your next tool call.
- If the item is not found, tell the user the folder/file was not found and \
ask them to verify the name or pick from the listing.
- For the sequencing samples in a reads folder (R1/R2 pairs), call \
``list_samples(path=...)`` instead of pairing files from a listing.

### Error Handling
- **"Object not found"** means the path does not exist or is misspelled. It \
does NOT mean the Workspace service is down or experiencing issues. Do NOT \
tell the user there is a "temporary issue" or "service outage".
- When you get an object-not-found error, try browsing the parent directory \
or searching with ``name_contains`` to find the correct path.

### Finding Genome Groups and Feature Groups
- To find all genome groups: ``workspace_browse(workspace_types=["genome_group"], recursive=true)``
  or ``workspace_browse(path="Genome Groups")``
- To find all feature groups: ``workspace_browse(workspace_types=["feature_group"], recursive=true)``
  or ``workspace_browse(path="Feature Groups")``
- These are the most common way users organize data in BV-BRC.

### Filtering
- ``workspace_types``: Filter by BV-BRC object type — PREFERRED for category filtering.
  Common types: ``genome_group``, ``feature_group``, ``reads``, ``contigs``,
  ``genbank_file``, ``gff``, ``csv``, ``json``, ``nwk``, ``folder``.
- ``file_extensions``: Filter by file extension (e.g. ``.fasta``, ``.csv``) — use for
  specific formats.
- ``name_contains``: Literal filename substring search — use for specific file names.
  Several spellings or synonyms go in ONE call with ``name_match="any"`` \
(e.g. ``["influenza", "flu"]``); ``name_match="word"`` keeps whole words only \
(a genus name then does not match a longer genus that contains it).
- ``created_after`` / ``created_before`` (YYYY-MM-DD): "last week", "this \
month", "since October" — work the date out from today's date.
- Questions about jobs ("which jobs failed", "where are my <service> \
results", "my recent <service> runs") go to ``list_jobs`` (``service``, \
``status``), not to folder searches: it has every job, and each job's \
``result_path`` / ``output_folder``, wherever its results were written.
- Default order: a listing shows folders first, then names; a search shows \
the newest first. Set ``sort_by`` only when the user asks for an order.
- Do NOT combine ``workspace_types`` and ``file_extensions`` for the same concept.
- Filters apply to one folder unless ``recursive=true``, which searches every \
folder under ``path``. Set ``recursive`` explicitly: true only to find files \
whose folder you don't know.
- Searches leave out files inside job output folders (hidden ``.<job>`` \
folders); ``hidden_in_job_outputs`` counts them. Set \
``include_job_outputs=true`` only when the user wants files a job produced.

### Large Results
- ``total`` is how many items matched; ``count`` how many came back. When \
``truncated`` is true you saw only part of it: say so ("showing 50 of 812"), \
and never guess what the rest is.
- ``summary`` gives counts by type and (searches) by subfolder, total size and \
dates over everything matched. Use it to describe a large folder.
- Fetch the next page (``offset=next_offset``) only when the user needs the \
rest; narrowing with a filter or subfolder is usually better.

### Uploaded Files
- Files the user attached in chat are saved under the session folder's \
``uploaded_files/`` subdirectory with their original name and extension.
- If the user attached a file in this turn, its excerpt and path are in the \
``=== ATTACHED DOCUMENTS ===`` section. Read the file for anything past the excerpt.
- The saved path is the original file (FASTA stays FASTA, CSV stays CSV). \
You may pass it as a GoWe workflow input. Do not invent a different path.
- Files attached in EARLIER turns are still there. To find one, browse the \
session workspace path from ``=== SESSION WORKSPACE ===`` and look in \
``uploaded_files/``. Do not tell the user a previously attached file is \
gone without looking there first.

### PDF Files
- ``read_file_preview`` on a ``.pdf`` file returns **extracted text** (not raw binary).
- Extracted text is cached as ``.txt`` under the session folder's \
``parsed_pdfs/`` subdirectory for fast re-reads.
- If the user attached a PDF in this turn, its excerpt and ``.txt`` path are in the \
``=== ATTACHED DOCUMENTS ===`` section. Read the ``.txt`` for anything past the excerpt.
- PDFs with no selectable text (scanned documents) cannot be read. Tell the user OCR \
is not supported.

### Reading Files
- ``read_file_preview`` returns at most 32 KB per call. To read on, call \
again with ``start_byte = next_start`` until ``is_complete`` is true.
- Write down what you need from each page (counts, IDs, metrics) before \
reading the next one. Earlier pages may be trimmed from your context; the \
trimmed stub keeps only the byte range you already covered.
- Use ``search_file`` to locate a row, ID, gene, sample name, header, or \
error message inside a file — it is grep for one file. Each match carries \
a ``byte_offset``; pass it as ``read_file_preview(start_byte=byte_offset)`` \
to read from there. For FASTA header matches, ``record_length`` is the \
sequence length, so "is gene X in this file and how long is it" is one call.
- ``summarize_file`` gives the structure of a table (tsv/csv/vcf), FASTA \
or FASTQ without reading it: exact row / sequence / read counts, column \
types and min/max/mean/sum, N50. To count or rank rows use ``where`` \
(``"padj < 0.05 and abs(log2FoldChange) >= 1"``) and ``sort_by`` + \
``top_n``. Use it whenever the user asks "how many" about a file.
- Do not read a large file end to end. Check ``total_size`` first \
(``get_file_metadata`` or the first read). If the file is over ~200 KB, \
``search_file`` for what the user asked about, or read the first page and \
ask what they need. To read the *end* of a file, ``read_file_preview`` with \
``start_byte`` a little less than ``total_size``.
- Compressed ``.gz`` files are read transparently; offsets refer to the \
uncompressed text.
- Group objects: use ``get_group_ids`` for the IDs. Job output lives in \
the hidden ``.<output_file>/`` folder under the job's ``output_path`` — \
browse it like any folder; use ``get_job_details(fetch_stderr=true)`` for \
the job's stderr.
- Images are not readable as text (``view_workspace_image`` arrives in \
Phase 4). SVG is text — read it.

### Constraints
- Read-only: you cannot create, modify, or delete workspace items.
- Default limit: 50 results.  Max: 500.
- Other users' workspaces only when asked. Find them with \
``workspace_browse(scope="shared")`` (shared with the user) or \
``scope="public"`` with ``name_contains`` (hundreds of public ones); then \
use their full paths (``/<owner>@<realm>/...``) with any workspace tool, and \
``get_group_ids`` with a group's full path.
- "Do any shared workspaces have X?": ONE call, \
``workspace_browse(scope="shared", recursive=true, name_contains=[...])`` \
(or ``workspace_types``); it searches inside all of them, and \
``summary.by_root`` says which have matches. Never loop over them one by one. \
Public workspaces are found by name only; their contents are not searched.
- Several known folders at once: ``paths=[...]`` instead of one call per folder.
- Preview binary files only when explicitly requested.
""".strip()
