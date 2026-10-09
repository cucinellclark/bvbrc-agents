"""
``summarize_file`` — structure and statistics of one workspace file without
loading it into the LLM context.

Streams the file through :mod:`shared.tools.file_stream` (Range GET, gzip
transparent, 25 MB scan cap) and returns a compact summary:

- **table** (csv / tsv / tab-separated tool output): columns, exact row
  count, per-column type, nulls, numeric min/max/mean/sum, top values.
  ``where`` filters rows ("padj < 0.05 and abs(log2FoldChange) >= 1") and
  the statistics are computed over the matching rows; ``sort_by`` +
  ``top_n`` return the top matching rows.  This is how the LLM counts rows —
  ``search_file`` stops at 200 matches and paging a 1 700-row table loses
  earlier pages to context trimming.
- **fasta**: sequence count, total length, min/median/max, N50, GC.
- **fastq**: read count, bases, length stats, quality encoding, pair hint.
- **text** (anything else): line count and first lines.

When the scan cap is hit, every count is renamed ``…_scanned`` so a partial
count cannot be reported as the file total.

Phase 3 of ``current_plans/workspace_file_reading.md`` (docs repo); gff,
json and html summarizers are not built yet.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from shared.tools.file_stream import SCAN_CAP_BYTES, iter_lines, open_workspace_stream
from shared.tools.workspace import (
    _extract_token,
    _extract_user_id,
    _is_object_not_found,
    _path_not_found_result,
    _resolve_path,
)

RESULT_CHAR_BUDGET = 11_000   # JSON of the result; loop cap is 12 000
MAX_COLUMNS_DETAILED = 40     # wider tables list the rest by name only
MAX_DISTINCT_TRACKED = 10_000
TOP_VALUES = 5
SAMPLE_ROWS = 3
MAX_TOP_N = 50
MAX_CELL_CHARS = 80
MAX_WHERE_CHARS = 500
FIRST_LINES = 15

_FASTA_EXTS = (".fa", ".fasta", ".fna", ".faa", ".ffn", ".frn")
_FASTQ_EXTS = (".fq", ".fastq")
_TABLE_EXTS = {".tsv": "\t", ".tab": "\t", ".csv": ",", ".vcf": "\t"}
_NULLS = {"", "na", "nan", "none", "null", "n/a", "-", "."}

FORMATS = ("table", "fasta", "fastq", "text")


# ---------------------------------------------------------------- helpers


def _strip_gz(name: str) -> str:
    return name[:-3] if name.lower().endswith(".gz") else name


def _ext(path: str) -> str:
    base = _strip_gz(path.lower()).rsplit("/", 1)[-1]
    return "." + base.rsplit(".", 1)[-1] if "." in base else ""


def _num(value: str) -> Optional[float]:
    v = value.strip()
    if v.lower() in _NULLS:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return f if f == f else None  # NaN → None


def _is_null(value: str) -> bool:
    return value.strip().lower() in _NULLS


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + "…"
    return value


def _round(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if x == int(x) and abs(x) < 1e15:
        return int(x)
    return float(f"{x:.6g}")


def _guess_format(path: str, first_lines: List[str]) -> str:
    ext = _ext(path)
    if ext in _FASTA_EXTS or path.lower().rstrip("/").endswith("contigs"):
        return "fasta"
    if ext in _FASTQ_EXTS:
        return "fastq"
    if ext in _TABLE_EXTS:
        return "table"
    content = [ln for ln in first_lines if ln.strip()]
    if not content:
        return "text"
    if content[0].startswith(">"):
        return "fasta"
    if content[0].startswith("@") and len(content) >= 3 and content[2].startswith("+"):
        return "fastq"
    data = [ln for ln in content if not ln.startswith("##")]
    if data and sum("\t" in ln for ln in data) >= 0.8 * len(data):
        return "table"
    if data and sum(ln.count(",") >= 1 for ln in data) >= 0.8 * len(data) and ext in ("", ".txt"):
        counts = {ln.count(",") for ln in data[:20]}
        if len(counts) == 1:
            return "table"
    return "text"


def _sniff_delimiter(path: str, lines: List[str]) -> str:
    ext = _ext(path)
    if ext in _TABLE_EXTS:
        return _TABLE_EXTS[ext]
    data = [ln for ln in lines if ln.strip() and not ln.startswith("##")]
    if data and sum("\t" in ln for ln in data) >= 0.8 * len(data):
        return "\t"
    try:
        return csv.Sniffer().sniff("\n".join(data[:20]), delimiters=",;\t|").delimiter
    except csv.Error:
        return "\t"


def _split(line: str, delim: str) -> List[str]:
    if delim == "\t":
        return line.split("\t")
    return next(csv.reader([line], delimiter=delim))


def _looks_like_header(first: List[str], rest: List[List[str]]) -> bool:
    """Header if some column is numeric in the data but not in the first row,
    or every first-row cell is non-numeric while data rows have numbers."""
    if not rest:
        return all(_num(c) is None for c in first)
    for i, cell in enumerate(first):
        col = [r[i] for r in rest if i < len(r) and not _is_null(r[i])]
        if col and _num(cell) is None and not _is_null(cell) and \
                sum(_num(v) is not None for v in col) >= 0.8 * len(col):
            return True
    if all(_num(c) is None for c in first):
        # all-text table: a header repeats nowhere below it
        firsts = {r[0] for r in rest if r}
        return first[0] not in firsts and len(set(first)) == len(first)
    return False


# ---------------------------------------------------------------- where


_COND = re.compile(
    r"""^\s*
    (?P<col>abs\(\s*(?:`[^`]+`|"[^"]+"|[^\s()]+)\s*\)|`[^`]+`|"[^"]+"|[^\s]+)
    \s+
    (?P<op><=|>=|!=|==|=|<|>|not\s+contains|contains|not\s+startswith|startswith|is\s+not\s+null|is\s+null)
    \s*
    (?P<val>.*?)\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


class WhereError(ValueError):
    pass


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "`\"'":
        return s[1:-1]
    return s


def _resolve_column(name: str, columns: List[str]) -> int:
    if name in columns:
        return columns.index(name)
    lower = [c.lower() for c in columns]
    if name.lower() in lower:
        return lower.index(name.lower())
    raise WhereError(f"unknown column {name!r}; columns are {columns[:MAX_COLUMNS_DETAILED]}")


def _column_ref(token: str, columns: List[str]) -> Tuple[int, bool]:
    """'abs(x)' → (index of x, True); 'x' → (index, False)."""
    m = re.match(r"^abs\(\s*(.+?)\s*\)$", token, re.IGNORECASE)
    if m:
        return _resolve_column(_unquote(m.group(1)), columns), True
    return _resolve_column(_unquote(token), columns), False


def compile_where(where: str, columns: List[str]) -> Tuple[Callable[[List[str]], Optional[bool]], List[int]]:
    """Compile 'a < 1 and abs(b) >= 2 and c contains x' into a predicate.

    The predicate returns True/False, or None when a numeric condition met a
    non-numeric cell (NA) — such rows do not match and are counted apart.
    Only AND is supported; call again for an OR.
    """
    if len(where) > MAX_WHERE_CHARS:
        raise WhereError(f"where is longer than {MAX_WHERE_CHARS} characters")
    if re.search(r"\s+or\s+", where, re.IGNORECASE):
        raise WhereError("only 'and' is supported; run one summarize_file per alternative")
    parts = re.split(r"\s+and\s+", where.strip(), flags=re.IGNORECASE)
    checks: List[Callable[[List[str]], Optional[bool]]] = []
    used: List[int] = []
    for part in parts:
        m = _COND.match(part)
        if not m:
            raise WhereError(
                f"cannot parse condition {part!r}; use e.g. 'padj < 0.05', "
                "'abs(log2FoldChange) >= 1', 'Gene contains rRNA', 'name startswith __'"
            )
        idx, use_abs = _column_ref(m.group("col"), columns)
        used.append(idx)
        op = re.sub(r"\s+", " ", m.group("op").lower())
        raw_val = _unquote(m.group("val"))

        if op in ("is null", "is not null"):
            want_null = op == "is null"
            checks.append(lambda row, i=idx, w=want_null: (_is_null(row[i]) if i < len(row) else True) == w)
            continue
        if op in ("contains", "not contains", "startswith", "not startswith"):
            needle = raw_val.lower()
            neg = op.startswith("not")
            if op.endswith("contains"):
                fn = lambda cell, n=needle: n in cell.lower()
            else:
                fn = lambda cell, n=needle: cell.lower().startswith(n)
            checks.append(lambda row, i=idx, f=fn, ng=neg: (f(row[i]) if i < len(row) else False) != ng)
            continue

        if op == "=":
            op = "=="
        target = _num(raw_val)
        if target is None:
            if op not in ("==", "!="):
                raise WhereError(f"{op} needs a number, got {raw_val!r}")
            want_eq = op == "=="
            checks.append(lambda row, i=idx, v=raw_val, w=want_eq:
                          ((row[i].strip() == v) if i < len(row) else False) == w)
            continue
        cmp = {
            "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
            "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
        }[op]

        def numeric(row, i=idx, t=target, c=cmp, a=use_abs):
            if i >= len(row):
                return None
            x = _num(row[i])
            if x is None:
                return None
            return c(abs(x) if a else x, t)

        checks.append(numeric)

    def predicate(row: List[str]) -> Optional[bool]:
        saw_na = False
        for chk in checks:
            r = chk(row)
            if r is None:
                saw_na = True
            elif not r:
                return False
        return None if saw_na else True

    return predicate, used


def _sort_key_fn(sort_by: str, columns: List[str]) -> Tuple[Callable[[List[str]], Optional[float]], bool, str]:
    desc = sort_by.startswith("-")
    token = sort_by[1:] if desc else sort_by
    idx, use_abs = _column_ref(token.strip(), columns)

    def key(row: List[str]) -> Optional[float]:
        if idx >= len(row):
            return None
        x = _num(row[idx])
        if x is None:
            return None
        return abs(x) if use_abs else x

    return key, desc, columns[idx]


# ---------------------------------------------------------------- table


class _ColStats:
    __slots__ = ("n", "nulls", "numeric", "sum", "min", "max", "values", "overflow")

    def __init__(self) -> None:
        self.n = 0
        self.nulls = 0
        self.numeric = 0
        self.sum = 0.0
        self.min: Optional[float] = None
        self.max: Optional[float] = None
        self.values: Counter = Counter()
        self.overflow = False

    def add(self, cell: str) -> None:
        self.n += 1
        if _is_null(cell):
            self.nulls += 1
            return
        x = _num(cell)
        if x is not None:
            self.numeric += 1
            self.sum += x
            self.min = x if self.min is None or x < self.min else self.min
            self.max = x if self.max is None or x > self.max else self.max
        if not self.overflow:
            self.values[cell] += 1
            if len(self.values) > MAX_DISTINCT_TRACKED:
                self.overflow = True

    def report(self, name: str) -> Dict[str, Any]:
        filled = self.n - self.nulls
        out: Dict[str, Any] = {"name": name}
        if filled and self.numeric >= 0.9 * filled:
            out["type"] = "numeric"
            out.update(min=_round(self.min), max=_round(self.max),
                       mean=_round(self.sum / self.numeric), sum=_round(self.sum))
            if self.numeric < filled:
                out["non_numeric"] = filled - self.numeric
        else:
            out["type"] = "text"
            if self.overflow:
                out["distinct"] = f">{MAX_DISTINCT_TRACKED}"
            else:
                out["distinct"] = len(self.values)
                if len(self.values) < filled:  # repeated values: show the commonest
                    out["top_values"] = [[_clip(v), c] for v, c in self.values.most_common(TOP_VALUES)]
        if self.nulls:
            out["nulls"] = self.nulls
        return out


async def _summarize_table(lines, path: str, first_lines: List[str], *, where, sort_by, top_n,
                           has_header, delimiter) -> Dict[str, Any]:
    delim = delimiter or _sniff_delimiter(path, first_lines)
    comment_lines = 0
    header: Optional[List[str]] = None
    pending: List[List[str]] = []     # rows read before the header decision
    header_decided = False
    columns: List[str] = []
    stats: List[_ColStats] = []
    total_rows = 0
    matched = 0
    na_skipped = 0
    samples: List[List[str]] = []
    top: List[Tuple[float, int, List[str]]] = []
    predicate = None
    sort_key = None
    sort_desc = False
    sort_col = None
    sort_missing = 0
    ragged = 0
    hash_header: Optional[List[str]] = None
    header_labels: Optional[List[str]] = None

    def setup(cols: List[str]) -> None:
        nonlocal columns, stats, predicate, sort_key, sort_desc, sort_col
        columns = [c.strip() or f"column_{i + 1}" for i, c in enumerate(cols)]
        seen: Counter = Counter()
        for i, c in enumerate(columns):  # make names unique
            seen[c] += 1
            if seen[c] > 1:
                columns[i] = f"{c}_{seen[c]}"
        stats = [_ColStats() for _ in columns]
        if where:
            predicate, _ = compile_where(where, columns)
        if sort_by:
            sort_key, sort_desc, sort_col = _sort_key_fn(sort_by, columns)

    def consume(row: List[str]) -> None:
        nonlocal total_rows, matched, na_skipped, ragged, sort_missing
        total_rows += 1
        if len(row) > len(columns):
            for i in range(len(columns), len(row)):
                columns.append(f"column_{i + 1}")
                st = _ColStats()
                st.n = st.nulls = total_rows - 1  # earlier rows lacked it
                stats.append(st)
        if len(row) != len(columns):
            ragged += 1
        if predicate is not None:
            r = predicate(row)
            if r is None:
                na_skipped += 1
                return
            if not r:
                return
        matched += 1
        for i, st in enumerate(stats):
            st.add(row[i] if i < len(row) else "")
        if len(samples) < (10 if predicate is not None else SAMPLE_ROWS):
            samples.append(row)
        if sort_key is not None:
            k = sort_key(row)
            if k is None:
                sort_missing += 1
                return
            key = -k if sort_desc else k
            top.append((key, total_rows, row))
            if len(top) > 4 * n_top:
                top.sort(key=lambda t: (t[0], t[1]))
                del top[n_top:]

    n_top = min(max(int(top_n or 10), 1), MAX_TOP_N)

    def decide() -> None:
        """Pick the header from the buffered rows, size the columns from the
        data (so where/sort_by can reach every column), then replay them."""
        nonlocal header, pending, header_decided, header_labels
        rows = pending
        if hash_header is not None:
            header = hash_header
        elif has_header is True and rows:
            header, rows = rows[0], rows[1:]
        elif has_header is None and rows and _looks_like_header(rows[0], rows[1:]):
            header, rows = rows[0], rows[1:]
        widths = Counter(len(r) for r in rows)
        width = widths.most_common(1)[0][0] if widths else len(header or [])
        if header is not None and width > len(header):
            # Header is short of the data (a label missing somewhere): names
            # cannot be placed safely, so columns are positional.
            header_labels = header
            setup([f"column_{i + 1}" for i in range(width)])
        else:
            setup(header or [f"column_{i + 1}" for i in range(max([width] + [len(r) for r in rows[:1]]))])
        header_decided = True
        pending = []
        for r in rows:
            consume(r)

    async for _, raw in lines:
        text = raw.decode("utf-8", errors="replace").rstrip("\r")
        if not text.strip():
            continue
        if text.startswith("##"):
            comment_lines += 1
            continue
        if text.startswith("#"):
            if not header_decided and hash_header is None and has_header is not False:
                hash_header = _split(text[1:], delim)  # '#CHROM', '#Template' style headers
            else:
                comment_lines += 1
            continue
        row = _split(text, delim)
        if not header_decided:
            pending.append(row)
            if len(pending) < 20:
                continue
            decide()
            continue
        consume(row)

    if not header_decided and (pending or hash_header is not None):  # fewer than 20 rows
        decide()

    out: Dict[str, Any] = {
        "format_detected": "table",
        "delimiter": {"\t": "tab", ",": "comma"}.get(delim, delim),
        "header": ("from file" if header is not None and header_labels is None
                   else "none — columns named column_N"),
        "row_count": total_rows,
        "column_count": len(columns),
    }
    if header_labels is not None:
        out["header"] = "MISALIGNED — columns named column_N"
        out["header_labels"] = [_clip(h) for h in header_labels]
        out["header_warning"] = (
            f"The header row has {len(header_labels)} labels but data rows have {len(columns)} "
            "fields, so a label is missing and names cannot be matched to columns by position. "
            "Map columns using sample_rows (and the file's own documentation), and say which "
            "column you used for each number."
        )
    if comment_lines:
        out["comment_lines_skipped"] = comment_lines
    if ragged:
        out["rows_with_other_column_count"] = ragged
    if where:
        out["where"] = where
        out["matched_rows"] = matched
        if na_skipped:
            out["rows_skipped_non_numeric"] = na_skipped
            out["note_non_numeric"] = (
                "rows whose tested cell was NA/non-numeric did not match and are not in matched_rows"
            )
    detailed = list(zip(columns, stats))[:MAX_COLUMNS_DETAILED]
    out["columns"] = [st.report(name) for name, st in detailed]
    if len(columns) > MAX_COLUMNS_DETAILED:
        out["more_columns"] = columns[MAX_COLUMNS_DETAILED:]
    out["stats_over"] = "matched rows" if where else "all rows"
    out["sample_rows"] = [[_clip(c) for c in r] for r in samples]
    if sort_key is not None:
        top.sort(key=lambda t: (t[0], t[1]))
        out["sort_by"] = sort_by
        out["top_rows"] = [dict(zip(columns, [_clip(c) for c in r])) for _, _, r in top[:n_top]]
        if sort_missing:
            out["rows_without_sort_value"] = sort_missing
    return out


# ---------------------------------------------------------------- sequences


def _n50(lengths: List[int]) -> int:
    half = sum(lengths) / 2
    acc = 0
    for L in sorted(lengths, reverse=True):
        acc += L
        if acc >= half:
            return L
    return 0


async def _summarize_fasta(lines, **_: Any) -> Dict[str, Any]:
    lengths: List[int] = []       # ungapped residues
    widths: List[int] = []        # with gap characters (alignment columns)
    headers: List[Tuple[str, int]] = []
    cur = -1
    comp: Counter = Counter()
    first_headers: List[str] = []
    async for _, raw in lines:
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        if text.startswith(">"):
            lengths.append(0)
            widths.append(0)
            cur = len(lengths) - 1
            h = _clip(text[1:])
            headers.append((h, cur))
            if len(first_headers) < 5:
                first_headers.append(h)
            continue
        if cur < 0:
            continue
        widths[cur] += len(text)
        lengths[cur] += len(text) - text.count("-") - text.count(".")
        if sum(comp.values()) < 5_000_000:
            comp.update(text.upper())
    out: Dict[str, Any] = {"format_detected": "fasta", "sequence_count": len(lengths)}
    if not lengths:
        return out
    total = sum(lengths)
    nuc = sum(comp[c] for c in "ACGTUN")
    seen = (sum(comp.values()) - comp["-"] - comp["."]) or 1
    alphabet = "dna" if nuc >= 0.9 * seen else "protein"
    out.update(
        total_length=total, min_length=min(lengths), median_length=int(statistics.median(lengths)),
        max_length=max(lengths), n50=_n50(lengths), alphabet=alphabet,
    )
    if alphabet == "dna":
        gc = comp["G"] + comp["C"]
        acgt = sum(comp[c] for c in "ACGT") or 1
        out["gc_percent"] = round(100 * gc / acgt, 2)
        out["n_count_sampled"] = comp["N"]
    if widths != lengths:
        out["aligned"] = True
        out["alignment_width"] = max(widths)
        out["length_note"] = ("lengths are ungapped residues per sequence; alignment_width is the "
                              "number of alignment columns, not a sequence length")
        if min(widths) != max(widths):
            out["width_range"] = [min(widths), max(widths)]
    out["first_headers"] = first_headers
    by_len = sorted(headers, key=lambda h: lengths[h[1]], reverse=True)[:3]
    out["longest"] = [{"header": h, "length": lengths[i]} for h, i in by_len]
    out.update(_contig_coverage(headers, lengths))
    return out


_COV_IN_HEADER = re.compile(r"(?:\bcoverage[ =:]|_cov_|\bcov[=:_ ]|\bdepth[=:])\s*([\d.]+)", re.IGNORECASE)


def _contig_coverage(headers: List[Tuple[str, int]], lengths: List[int]) -> Dict[str, Any]:
    """Assembler headers (Unicycler/SPAdes/BV-BRC 'length L coverage C') carry
    per-contig depth.  Plasmid candidates and collapsed repeats are the
    contigs far from the length-weighted median, so list those explicitly —
    the longest-first sample above never shows them."""
    covs: List[Tuple[str, int, float]] = []
    for h, i in headers:
        m = _COV_IN_HEADER.search(h)
        if not m:
            return {}
        try:
            covs.append((h.split()[0], lengths[i], float(m.group(1).rstrip("."))))
        except ValueError:
            return {}
    if not covs:
        return {}
    # length-weighted median depth = the depth of the bulk of the genome
    acc, half, median = 0, sum(c[1] for c in covs) / 2, covs[0][2]
    for name, L, c in sorted(covs, key=lambda c: c[2]):
        acc += L
        if acc >= half:
            median = c
            break
    rows = [{"name": n, "length": L, "coverage": c, "relative": round(c / median, 2) if median else None}
            for n, L, c in covs]
    out: Dict[str, Any] = {"coverage_weighted_median": median}
    if len(rows) <= 60:
        out["contigs"] = rows
    out["coverage_outliers"] = [
        r for r in rows if r["relative"] is not None and (r["relative"] < 0.5 or r["relative"] > 1.5)
    ][:40]
    out["coverage_note"] = (
        "relative = contig depth / length-weighted median depth. <0.5 or >1.5 with its own graph "
        "component: plasmid candidate; short contigs >3: collapsed repeats or high-copy plasmids."
    )
    return out


async def _summarize_fastq(lines, **_: Any) -> Dict[str, Any]:
    n = 0
    bases = 0
    min_len: Optional[int] = None
    max_len = 0
    qmin = 255
    first_headers: List[str] = []
    pair_marks: Counter = Counter()
    pos = 0
    async for _, raw in lines:
        text = raw.decode("utf-8", errors="replace").rstrip("\r")
        k = pos % 4
        pos += 1
        if k == 0:
            if len(first_headers) < 3:
                first_headers.append(_clip(text))
            if n < 1000:
                if re.search(r"/1$|\s1:[NY]:", text):
                    pair_marks["1"] += 1
                elif re.search(r"/2$|\s2:[NY]:", text):
                    pair_marks["2"] += 1
        elif k == 1:
            L = len(text.strip())
            n += 1
            bases += L
            min_len = L if min_len is None or L < min_len else min_len
            max_len = max(max_len, L)
        elif k == 3 and n <= 10_000 and text:
            qmin = min(qmin, min(text.encode()))
    out: Dict[str, Any] = {"format_detected": "fastq", "read_count": n, "total_bases": bases}
    if n:
        out.update(min_length=min_len, mean_length=round(bases / n, 1), max_length=max_len)
        out["quality_encoding"] = "phred33" if qmin < 59 else "phred64 (or high-quality phred33)"
        if pair_marks:
            out["pair_hint"] = "read 2 of a pair" if pair_marks["2"] > pair_marks["1"] else "read 1 of a pair"
        out["first_headers"] = first_headers
    return out


async def _summarize_text(lines, **_: Any) -> Dict[str, Any]:
    n = 0
    first: List[str] = []
    longest = 0
    async for _, raw in lines:
        n += 1
        text = raw.decode("utf-8", errors="replace").rstrip("\r")
        longest = max(longest, len(text))
        if len(first) < FIRST_LINES:
            first.append(_clip(text))
    return {"format_detected": "text", "line_count": n, "longest_line_chars": longest, "first_lines": first}


SUMMARIZERS = {
    "fasta": _summarize_fasta,
    "fastq": _summarize_fastq,
    "text": _summarize_text,
}

_COUNT_KEYS = ("row_count", "matched_rows", "rows_skipped_non_numeric", "sequence_count",
               "total_length", "read_count", "total_bases", "line_count")


# ---------------------------------------------------------------- tool


class _Replay:
    """Async line iterator that first replays buffered lines."""

    def __init__(self, buffered, rest):
        self.buffered = buffered
        self.rest = rest

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for item in self.buffered:
            yield item
        async for item in self.rest:
            yield item


def _fit(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the result under the loop cap by shedding detail, never counts."""
    def size() -> int:
        return len(json.dumps(result, indent=2, default=str))

    for key, shrink in (("top_rows", 10), ("sample_rows", 1), ("first_lines", 5), ("contigs", 0), ("top_rows", 3)):
        if size() <= RESULT_CHAR_BUDGET:
            break
        if isinstance(result.get(key), list) and len(result[key]) > shrink:
            result[key] = result[key][:shrink]
            result.setdefault("trimmed_for_size", []).append(key)
    cols = result.get("columns")
    while isinstance(cols, list) and len(cols) > 5 and size() > RESULT_CHAR_BUDGET:
        dropped = cols.pop()
        result.setdefault("more_columns", []).insert(0, dropped["name"])
    if size() > RESULT_CHAR_BUDGET and isinstance(cols, list):
        for c in cols:
            c.pop("top_values", None)
    return result


async def summarize_file(
    path: str,
    format: Optional[str] = None,
    where: Optional[str] = None,
    sort_by: Optional[str] = None,
    top_n: Optional[int] = None,
    has_header: Optional[bool] = None,
    delimiter: Optional[str] = None,
    config: Any = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Summarize one workspace file; see the module docstring."""
    token = _extract_token(headers)
    user_id = _extract_user_id(headers)
    if not token:
        return {
            "error": "Authentication required for workspace operations.",
            "errorType": "AUTHENTICATION_ERROR",
            "source": "bvbrc-workspace",
        }
    resolved_path = _resolve_path(path, user_id)
    fmt = (format or "").strip().lower() or None
    if fmt in ("csv", "tsv"):
        delimiter = delimiter or ("," if fmt == "csv" else "\t")
        fmt = "table"
    if fmt is not None and fmt not in FORMATS:
        return {"error": f"format must be one of {list(FORMATS)} (or csv/tsv)",
                "errorType": "INVALID_PARAMETERS"}
    if delimiter in ("tab", "\\t"):
        delimiter = "\t"
    elif delimiter == "comma":
        delimiter = ","
    if resolved_path.lower().endswith(".pdf"):
        return {
            "error": "summarize_file does not read PDFs.",
            "errorType": "UNSUPPORTED_FILE",
            "hint": "Use read_file_preview on the PDF; it extracts the text.",
            "workspace_path": resolved_path,
        }

    try:
        info, stream = await open_workspace_stream(resolved_path, token, config, start_byte=0, max_bytes=None)
    except Exception as e:
        if _is_object_not_found(e):
            return _path_not_found_result(resolved_path)
        return {"error": f"File open failed: {type(e).__name__}: {e}", "errorType": "API_ERROR",
                "path": resolved_path, "source": "bvbrc-workspace"}

    if info.detected_format and info.detected_format != "gzip":
        return {"error": "binary file", "errorType": "BINARY_FILE",
                "hint": f"Detected format: {info.detected_format}. summarize_file only reads text.",
                "workspace_path": resolved_path, "total_size": info.total_size}

    try:
        line_iter = iter_lines(stream).__aiter__()
        buffered: List[Tuple[int, bytes]] = []
        content_lines = 0
        async for item in line_iter:  # sniff on 40 lines past any '##' preamble
            buffered.append(item)
            if not item[1].startswith(b"##"):
                content_lines += 1
            if content_lines >= 40 or len(buffered) >= 2000:
                break
        first_lines = [raw.decode("utf-8", errors="replace") for _, raw in buffered]
        fmt = fmt or _guess_format(resolved_path, first_lines)
        lines = _Replay(buffered, line_iter)
        if (where or sort_by) and fmt != "table":
            return {"error": f"where/sort_by only apply to tables; this file was read as {fmt}",
                    "errorType": "INVALID_PARAMETERS", "workspace_path": resolved_path}
        if fmt == "table":
            result = await _summarize_table(
                lines, resolved_path, first_lines, where=where, sort_by=sort_by, top_n=top_n,
                has_header=has_header, delimiter=delimiter)
        else:
            result = await SUMMARIZERS[fmt](lines)
    except WhereError as e:
        return {"error": str(e), "errorType": "INVALID_PARAMETERS", "workspace_path": resolved_path}
    except Exception as e:
        if _is_object_not_found(e):
            return _path_not_found_result(resolved_path)
        return {"error": f"Summary failed: {type(e).__name__}: {e}", "errorType": "API_ERROR",
                "path": resolved_path, "source": "bvbrc-workspace"}
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()

    partial = bool(info.partial)
    if partial:
        for k in _COUNT_KEYS:
            if k in result:
                result[f"{k}_scanned"] = result.pop(k)
        result["partial_note"] = (
            f"Only the first {SCAN_CAP_BYTES // (1024 * 1024)} MB were scanned; every *_scanned "
            "count covers that part only. Do not report it as the file total."
        )
    result.update(
        workspace_path=resolved_path,
        total_size=info.total_size,
        scanned_bytes=info.bytes_consumed,
        partial=partial,
        gzip=info.gzip,
    )
    result = _fit(result)
    summary_keys = ("format_detected", "row_count", "row_count_scanned", "matched_rows",
                    "matched_rows_scanned", "where", "sequence_count", "read_count", "line_count")
    result["_summary"] = {"path": resolved_path, **{k: result[k] for k in summary_keys if k in result}}
    return result
