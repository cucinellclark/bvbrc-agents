"""Tests for shared/tools/file_summary.py (summarize_file).

Reuses the local Range-capable HTTP server from test_file_reading.py, so no
network is needed.  Table shapes mirror real BV-BRC outputs checked
2026-10-09: an RNASeq DESeq2 table (1 742 rows, padj NA), a counts matrix
with an unnamed first column and HTSeq ``__`` rows, a VCF with a long
``##`` preamble, a headerless Kraken report.
"""

from __future__ import annotations

import pytest

from tests.test_file_reading import (  # noqa: F401  (range_server is a fixture)
    MockConfig,
    _mock_headers,
    _patch_download_url,
    _write,
    range_server,
)

from shared.tools import file_summary
from shared.tools.file_summary import WhereError, compile_where, summarize_file


async def _summ(base_url, filename, **kw):
    kw.setdefault("config", MockConfig())
    kw.setdefault("headers", _mock_headers())
    with _patch_download_url(base_url, filename):
        return await summarize_file(path=f"/test@patricbrc.org/home/{filename}", **kw)


def _deseq(n=1742):
    rows = ["Gene_Name\tbaseMean\tlog2FoldChange\tlfcSE\tstat\tpvalue\tpadj"]
    for i in range(n):
        padj = "NA" if i % 50 == 0 else f"{(i % 100) / 100:.4f}"   # 0.00 … 0.99
        lfc = ((i % 9) - 4) * 0.5                                    # -2.0 … 2.0
        rows.append(f"fig|727.3012.peg.{i}\t{100 + i}\t{lfc}\t0.2\t1.0\t0.01\t{padj}")
    return "\n".join(rows) + "\n"


def _col(result, name):
    return next(c for c in result["columns"] if c["name"] == name)


@pytest.mark.asyncio
async def test_table_counts_every_row_and_types(range_server):
    tmpdir, base = range_server
    _write(tmpdir, "de.deseq2.tsv", _deseq())
    r = await _summ(base, "de.deseq2.tsv")
    assert "error" not in r, r
    assert r["format_detected"] == "table" and r["header"] == "from file"
    assert r["row_count"] == 1742 and r["column_count"] == 7
    assert _col(r, "Gene_Name")["type"] == "text"
    padj = _col(r, "padj")
    assert padj["type"] == "numeric" and padj["nulls"] == 35  # every 50th row is NA
    assert _col(r, "baseMean")["min"] == 100 and _col(r, "baseMean")["max"] == 1841
    assert r["_summary"]["row_count"] == 1742


@pytest.mark.asyncio
async def test_where_counts_matches_and_skips_na(range_server):
    tmpdir, base = range_server
    _write(tmpdir, "de2.deseq2.tsv", _deseq())
    r = await _summ(base, "de2.deseq2.tsv", where="padj < 0.05")
    expected = sum(1 for i in range(1742) if i % 50 != 0 and (i % 100) < 5)
    assert r["matched_rows"] == expected
    assert r["rows_skipped_non_numeric"] == 35
    assert r["stats_over"] == "matched rows"
    assert _col(r, "padj")["max"] < 0.05

    up = await _summ(base, "de2.deseq2.tsv", where="padj < 0.05 and log2FoldChange >= 1")
    both = await _summ(base, "de2.deseq2.tsv", where="padj < 0.05 and abs(log2FoldChange) >= 1")
    down = await _summ(base, "de2.deseq2.tsv", where="padj < 0.05 and log2FoldChange <= -1")
    assert up["matched_rows"] + down["matched_rows"] == both["matched_rows"] > 0


@pytest.mark.asyncio
async def test_sort_by_returns_top_rows(range_server):
    tmpdir, base = range_server
    _write(tmpdir, "de3.deseq2.tsv", _deseq(300))
    r = await _summ(base, "de3.deseq2.tsv", where="padj is not null", sort_by="-baseMean", top_n=3)
    assert [row["baseMean"] for row in r["top_rows"]] == ["399", "398", "397"]
    r = await _summ(base, "de3.deseq2.tsv", sort_by="abs(log2FoldChange)", top_n=2)
    assert all(abs(float(row["log2FoldChange"])) == 0 for row in r["top_rows"])


@pytest.mark.asyncio
async def test_counts_matrix_unnamed_column_and_htseq_rows(range_server):
    """Gene-assignment rate needs per-sample sums of gene rows vs __ rows."""
    tmpdir, base = range_server
    lines = ["\tctrl_rep1\tinf_rep1"]
    lines += [f"fig|727.3012.peg.{i}\t10\t2" for i in range(100)]
    lines += ["__no_feature\t500\t300", "__ambiguous\t50\t20", "__too_low_aQual\t900\t600",
              "__not_aligned\t0\t0", "__alignment_not_unique\t0\t0"]
    _write(tmpdir, "gene_counts_matrix.tsv", "\n".join(lines) + "\n")
    genes = await _summ(base, "gene_counts_matrix.tsv", where="column_1 not startswith __")
    unassigned = await _summ(base, "gene_counts_matrix.tsv", where="column_1 startswith __")
    assert genes["columns"][0]["name"] == "column_1"
    assert genes["matched_rows"] == 100 and unassigned["matched_rows"] == 5
    assert _col(genes, "ctrl_rep1")["sum"] == 1000
    assert _col(unassigned, "ctrl_rep1")["sum"] == 1450
    assert len(unassigned["sample_rows"]) == 5  # all matches shown when filtered


@pytest.mark.asyncio
async def test_vcf_long_preamble_hash_header(range_server):
    tmpdir, base = range_server
    pre = [f"##INFO=<ID=X{i},Number=1>" for i in range(60)]
    body = [f"NC_002516\t{1000 + i}\t.\tA\tG\t{50 + i}\t.\tDP=10" for i in range(68)]
    _write(tmpdir, "PE1.var.vcf", "\n".join(pre + ["#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"] + body) + "\n")
    r = await _summ(base, "PE1.var.vcf", where="QUAL >= 100")
    assert r["format_detected"] == "table" and r["row_count"] == 68
    assert r["comment_lines_skipped"] == 60
    assert [c["name"] for c in r["columns"]][:2] == ["CHROM", "POS"]
    assert r["matched_rows"] == 18


@pytest.mark.asyncio
async def test_headerless_kraken_report_and_categorical_counts(range_server):
    tmpdir, base = range_server
    rows = [" 90.00\t9000\t100\tG\t561\t  Escherichia",
            " 80.00\t8000\t8000\tS\t562\t    Escherichia coli",
            "  5.00\t500\t500\tS\t28901\t    Salmonella enterica",
            "  1.00\t100\t100\tS\t1280\t    Staphylococcus aureus"] * 6
    _write(tmpdir, "k2_report.txt", "\n".join(rows) + "\n")
    r = await _summ(base, "k2_report.txt", where="column_4 == S", sort_by="-column_2", top_n=2)
    assert r["header"].startswith("none")
    assert r["row_count"] == 24 and r["matched_rows"] == 18
    assert r["top_rows"][0]["column_6"].strip() == "Escherichia coli"
    rank = _col(await _summ(base, "k2_report.txt"), "column_4")
    assert rank["top_values"] == [["S", 18], ["G", 6]]


@pytest.mark.asyncio
async def test_csv_with_quoted_commas(range_server):
    tmpdir, base = range_server
    _write(tmpdir, "amr.csv", 'Antibiotic,Class,MIC\n"ampicillin, high dose",beta-lactam,8\nciprofloxacin,quinolone,0.5\n')
    r = await _summ(base, "amr.csv")
    assert r["row_count"] == 2 and r["column_count"] == 3
    assert _col(r, "MIC")["type"] == "numeric"


@pytest.mark.asyncio
async def test_fasta_stats(range_server):
    tmpdir, base = range_server
    seqs = {"c1 length 900 coverage 40.5": "ACGT" * 225, "c2": "GGCC" * 50, "c3": "AT" * 50}
    text = "".join(f">{h}\n" + "\n".join(s[i:i + 60] for i in range(0, len(s), 60)) + "\n"
                   for h, s in seqs.items())
    _write(tmpdir, "x_contigs.fasta", text)
    r = await _summ(base, "x_contigs.fasta")
    assert r["sequence_count"] == 3 and r["total_length"] == 1200
    assert (r["min_length"], r["median_length"], r["max_length"]) == (100, 200, 900)
    assert r["n50"] == 900 and r["alphabet"] == "dna"
    assert r["longest"][0] == {"header": "c1 length 900 coverage 40.5", "length": 900}
    assert r["gc_percent"] == round(100 * (450 + 200) / 1200, 2)


@pytest.mark.asyncio
async def test_fastq_gz(range_server):
    tmpdir, base = range_server
    recs = "".join(f"@r{i} 1:N:0:1\n{'A' * (50 + i % 3)}\n+\n#{'I' * (49 + i % 3)}\n" for i in range(30))
    _write(tmpdir, "reads_R1.fastq.gz", recs, gz=True)
    r = await _summ(base, "reads_R1.fastq.gz")
    assert r["gzip"] is True and r["format_detected"] == "fastq"
    assert r["read_count"] == 30 and r["min_length"] == 50 and r["max_length"] == 52
    assert r["quality_encoding"] == "phred33" and r["pair_hint"] == "read 1 of a pair"


@pytest.mark.asyncio
async def test_partial_scan_renames_counts(range_server, monkeypatch):
    tmpdir, base = range_server
    _write(tmpdir, "big.tsv", "a\tb\n" + "".join(f"x{i}\t{i}\n" for i in range(5000)))
    monkeypatch.setattr("shared.tools.file_stream.SCAN_CAP_BYTES", 4096)
    r = await _summ(base, "big.tsv")
    assert r["partial"] is True
    assert "row_count" not in r and 0 < r["row_count_scanned"] < 5000
    assert "Do not report it as the file total" in r["partial_note"]


@pytest.mark.asyncio
async def test_wide_table_fits_budget(range_server):
    tmpdir, base = range_server
    header = "\t".join(f"sample_with_a_long_name_{i}" for i in range(120))
    rows = ["\t".join(f"label_{r}_{c}_padding_text" for c in range(120)) for r in range(30)]
    _write(tmpdir, "wide.tsv", header + "\n" + "\n".join(rows) + "\n")
    r = await _summ(base, "wide.tsv", sort_by="sample_with_a_long_name_0")
    import json
    assert len(json.dumps(r, indent=2)) <= file_summary.RESULT_CHAR_BUDGET + 600
    assert r["row_count"] == 30 and r["column_count"] == 120


@pytest.mark.asyncio
async def test_errors(range_server):
    tmpdir, base = range_server
    _write(tmpdir, "t.tsv", "a\tb\n1\t2\n")
    r = await _summ(base, "t.tsv", where="nope < 1")
    assert r["errorType"] == "INVALID_PARAMETERS" and "unknown column" in r["error"]
    r = await _summ(base, "t.tsv", where="a < 1 or b > 2")
    assert "only 'and'" in r["error"]
    _write(tmpdir, "s.fasta", ">x\nACGT\n")
    r = await _summ(base, "s.fasta", where="a < 1")
    assert r["errorType"] == "INVALID_PARAMETERS"
    r = await summarize_file(path="/x/y.pdf", config=MockConfig(), headers=_mock_headers())
    assert r["errorType"] == "UNSUPPORTED_FILE"
    r = await summarize_file(path="/x/y.tsv", config=MockConfig(), headers={})
    assert r["errorType"] == "AUTHENTICATION_ERROR"


def test_compile_where_parsing():
    cols = ["Gene", "% Identity", "padj"]
    pred, used = compile_where("`% Identity` >= 90 and Gene contains bla", cols)
    assert used == [1, 0]
    assert pred(["blaKPC-2", "99.5", "0.01"]) is True
    assert pred(["gyrA", "99.5", "0.01"]) is False
    assert pred(["blaTEM", "NA", "0.01"]) is None  # NA → neither match nor non-match
    pred, _ = compile_where("padj is null", cols)
    assert pred(["g", "1", "NA"]) is True
    with pytest.raises(WhereError):
        compile_where("Gene > abc", cols)


@pytest.mark.asyncio
async def test_fasta_contig_coverage_outliers(range_server):
    """Assembler headers carry depth; the low-depth plasmid candidate must be
    listed even though it is not among the longest contigs (GA2 regression)."""
    tmpdir, base = range_server
    spec = [(1, 800_000, 40.5), (2, 576_000, 47.2), (3, 376_000, 57.1), (13, 93_817, 18.9), (38, 1_666, 372.2)]
    text = "".join(f">S_contig_{n} length {L} coverage {c} normalized_cov 1.0\n" + "ACGT" * (L // 4) + "\n"
                   for n, L, c in spec)
    _write(tmpdir, "cov_contigs.fasta", text)
    r = await _summ(base, "cov_contigs.fasta")
    assert r["coverage_weighted_median"] == 47.2
    assert len(r["contigs"]) == 5
    assert [o["name"] for o in r["coverage_outliers"]] == ["S_contig_13", "S_contig_38"]
    assert r["coverage_outliers"][0]["relative"] == 0.4


@pytest.mark.asyncio
async def test_fasta_without_coverage_headers_has_no_coverage_block(range_server):
    tmpdir, base = range_server
    _write(tmpdir, "plain.fasta", ">a\nACGT\n>b desc\nACGTAC\n")
    r = await _summ(base, "plain.fasta")
    assert "coverage_outliers" not in r and r["sequence_count"] == 2


@pytest.mark.asyncio
async def test_fasta_alignment_reports_ungapped_lengths(range_server):
    """GeneTree regression: '279 aa uniform length' was the alignment width."""
    tmpdir, base = range_server
    _write(tmpdir, "x_aligned.fa", ">a\nMKV--LLA-\n>b\nMKVQQLLAW\n>c\n----LL---\n")
    r = await _summ(base, "x_aligned.fa")
    assert r["aligned"] is True and r["alignment_width"] == 9
    assert (r["min_length"], r["max_length"]) == (2, 9)
    assert r["alphabet"] == "protein"
