"""Service output knowledge for the analysis agent.

Per service: which small files hold the numbers (read these first), which
large files never to scan, the checks and typical ranges to judge results
against, and sensible next steps.

Built 2026-10-08 from 13 real completed jobs whose folders were surveyed and
whose baseline analyses were graded against the files
(``analysis_lit_eval/`` in the docs repo, plan
``current_plans/analysis_and_literature.md``). Services marked
``verified: False`` still carry the old, unchecked templates.

Path templates use:
  ${output_folder}        = ${params.output_path}/.${params.output_file}
  ${params.output_path}   - the workspace output directory
  ${params.output_file}   - the output file prefix

Ranges are general guides from the literature and common practice, not
BV-BRC-defined cutoffs, unless a line says otherwise. Values marked
"(unconfirmed)" should not be stated as BV-BRC policy.
"""

from __future__ import annotations

from typing import Any


OF = "${output_folder}"
NAME = "${params.output_file}"

# AMR guidance shared by every service that runs BV-BRC annotation.
_AMR_RULES = [
    "AMR: lead with specialty-amrfinder.txt (acquired genes and point mutations), grouped by "
    "drug class; then amr-sir.txt / amr-mic.txt as PREDICTED phenotypes. Flag predictions the "
    "genes do not explain (e.g. carbapenem R with no carbapenemase, fluoroquinolone R with no "
    "gyrA/parC mutation).",
    "'Antibiotic target in susceptible species' hits (gyrA, rpoB, parC, PBPs, …) are normal "
    "housekeeping genes, NOT resistance genes. Never list them as AMR genes.",
    "Present AMR as a per-drug-class table: gene or mutation | predicted S/R | agree? Read the S/R "
    "value from amr-sir.txt exactly. A resistance gene with an S prediction is a conflict too "
    "(e.g. aadA1 with streptomycin S).",
    "Acquired genes (bla, sul, tet, aad/aph/aac on mobile elements, dfr, floR…) differ from intrinsic "
    "chromosomal genes and efflux pumps present in every member of the species — do not present the "
    "intrinsic ones as the isolate's resistance profile. Biocide/heavy-metal genes (qacEdelta1, …) are "
    "not antibiotic classes.",
    "SIR predictions AND AMRFinder point-mutation calls depend on the submitted species: if the "
    "identity check fails, treat both as unreliable (a 'mutation' may be the native residue of the "
    "real species).",
    "In specialty-rgi.txt, hits covering <~60% of the reference are fragments.",
]

_IDENTITY_RULES = [
    "IDENTITY CHECK (do this first): compare the submitted organism (job parameters "
    "scientific_name / taxonomy_id) with the 'Group:' line in genome_quality_details.txt, GC% "
    "and genome size. Typical GC / size: E. coli 50.5–50.8% / 4.6–5.5 Mb; Salmonella enterica "
    "52.0–52.2% / 4.6–5.1 Mb; Klebsiella pneumoniae ~57% / 5.2–5.8 Mb; Staphylococcus aureus "
    "~32.8% / 2.8 Mb; Pseudomonas aeruginosa ~66% / 6.3–7 Mb; Mycobacterium tuberculosis "
    "~65.6% / 4.4 Mb. If the GENUS or species clearly disagrees, say so FIRST — the name may be a mislabel.",
    "The 'Group:' line names the reference group used to estimate completeness, not an identity "
    "call: a different serovar or strain in it (e.g. Heidelberg for an Infantis genome) is normal "
    "and is not a conflict.",
    "PLfam/CDS ratio below ~0.9 while PGfam/CDS is ~0.98 suggests the genus-specific families "
    "are for the wrong genus (another sign of a mislabel).",
]

_ANNOTATION_QUALITY = [
    "Genome quality: report completeness, contamination, coarse and fine consistency and "
    "'Good Seed' from genome_quality_details.txt / quality.json. Commonly quoted BV-BRC 'good "
    "genome' cutoffs are completeness ≥80%, contamination ≤10%, fine consistency ≥87% "
    "(unconfirmed — do not state them as official BV-BRC policy).",
    "Hypothetical proteins 20–35% of CDS is typical for well-studied bacteria. CDS ratio ≈1.0.",
    "Short-read assemblies collapse rRNA operons: a low rRNA count is expected, not a quality "
    "failure. 'Chromosomes 0 / plasmids 0' on a draft means 'not classified', not 'none'.",
    "tRNA/rRNA counts are in quality.json (feature_summary), not GenomeReport.html.",
]

_ASSEMBLY_CHECKS = [
    "Read details/<name>_run_details.json first: QUAST metrics, contig filtering and Pilon "
    "rounds in one small file. The AssemblyReport.html starts with ~30 KB of SVG.",
    "Typical Illumina bacterial assembly: contigs <100 good, 100–300 fair, >500 fragmented or "
    "contaminated; N50 >100 kb good, 20–100 kb fair, <20 kb poor; total length within ~±10% "
    "of the species' expected size; depth <20× low, 30–100× good (the service downsamples to "
    "target_depth); N's per 100 kbp 0 for Unicycler; Pilon changes few and converging to 0.",
    "Contig headers carry length, coverage and normalized_cov: ~1.0 = chromosome. A contig at "
    "<0.5× or >1.5× normalised depth, especially one that is its own Unicycler graph component "
    "(see the Unicycler log's final Component/Status table), is a plasmid candidate; short "
    "contigs at >3× are collapsed repeats or high-copy plasmids. Do not call a contig a "
    "plasmid from its length alone.",
]

SERVICE_KNOWLEDGE: dict[str, dict[str, Any]] = {
    "GenomeAssembly2": {
        "verified": True,
        "read_first": {
            f"{OF}/details/{NAME}_run_details.json": "QUAST metrics, contig filtering, Pilon rounds, read stats",
            f"{OF}/details/{NAME}_unicycler.log": "Unicycler component table (circular/complete vs incomplete); read the end",
            f"{OF}/{NAME}_contigs.fasta": "search_file '>' for headers: length, coverage, normalized_cov per contig",
            f"{OF}/{NAME}_AssemblyReport.html": "overview report for the user (mostly SVG; link it, do not page through it)",
        },
        "avoid": {
            f"{OF}/details/{NAME}_assembly_graph.gfa": "multi-MB graph",
        },
        "checks": _ASSEMBLY_CHECKS,
        "next_steps": [
            "Annotate (GenomeAnnotation) or run Comprehensive Genome Analysis for identity, completeness and AMR.",
            "Similar Genome Finder for nearest relatives.",
            "Long-read or hybrid assembly to close plasmids/replicons.",
        ],
    },
    "GenomeAnnotation": {
        "verified": True,
        "read_first": {
            f"{OF}/genome_quality_details.txt": "Good Seed, completeness, contamination, 'Group:' (reference group used — identity check), role consistency",
            f"{OF}/quality.json": "search_file for \"feature_summary\" and \"genome_length\" (CDS, tRNA, rRNA, GC, PLfam/PGfam ratios); counts sit ~13 KB in",
            f"{OF}/specialty-amrfinder.txt": "acquired AMR genes and point mutations (small; always read)",
            f"{OF}/amr-sir.txt": "predicted S/I/R per antibiotic",
            f"{OF}/amr-mic.txt": "predicted MICs",
            f"{OF}/GenomeReport.html": "summary for the user; problematic roles",
        },
        "avoid": {
            f"{OF}/{NAME}.genome": "18 MB",
            f"{OF}/{NAME}.gb / .embl / .xls / .txt": "multi-MB feature dumps",
            f"{OF}/load_files/": "database load files",
            f"{OF}/specialty-blast-alignments.txt": "multi-MB",
        },
        "checks": _IDENTITY_RULES + _ANNOTATION_QUALITY + _AMR_RULES,
        "next_steps": [
            "Confirm resistance with laboratory AST.",
            "MLST/serotyping; Similar Genome Finder or a CodonTree against references of the expected species.",
            "If the identity check fails, re-annotate with the correct taxon.",
        ],
    },
    "ComprehensiveGenomeAnalysis": {
        "verified": True,
        "read_first": {
            f"{OF}/.annotation/genome_quality_details.txt": "quality + 'Group:' line (identity check)",
            f"{OF}/.annotation/quality.json": "feature_summary, GC, PLfam/PGfam ratios",
            f"{OF}/.annotation/specialty-amrfinder.txt": "acquired AMR genes and point mutations",
            f"{OF}/.annotation/amr-sir.txt": "predicted S/I/R",
            f"{OF}/.assembly/details/": "*_run_details.json: assembly metrics",
            f"{OF}/codontree.genesPerGenome.txt": "genes used for the tree",
            f"{OF}/codontree.nex": "tree with organism names: nearest neighbours of the new genome (identity check)",
            f"{OF}/FullGenomeReport.html": "1.8 MB, mostly SVG; useful text in the first ~16 KB",
        },
        "avoid": {f"{OF}/annotated.genome": "18 MB"},
        "checks": _IDENTITY_RULES + [
            "The CodonTree in CGA uses few genes (often ~5): placement is coarse, but neighbours of "
            "a different genus than submitted are a strong mislabel signal.",
        ] + _ASSEMBLY_CHECKS[1:2] + _ANNOTATION_QUALITY + _AMR_RULES,
        "next_steps": [
            "If the identity check fails: Similar Genome Finder / ANI, then re-annotate with the correct taxon.",
            "Confirm resistance with laboratory AST.",
        ],
    },
    "TaxonomicClassification": {
        "verified": True,
        "read_first": {
            f"{OF}/JobFailed.html": "failure report, if present (check the job's output_check first)",
            f"{OF}/summary_table.csv": "per sample: raw, classified, unclassified, host (chordate), bacterial, viral, fungal reads",
            f"{OF}/<sample>/kraken_output/<sample>_k2_report.txt": "Kraken report: col1 % of reads in clade, col2 clade count, col3 reads assigned DIRECTLY to that taxon, col4 rank",
            f"{OF}/<sample>/bracken_output/<sample>_bracken_report.txt": "species abundance after redistribution — use this for species %",
            f"{OF}/alpha_diversity.csv": "Shannon, Simpson, Berger-Parker, Fisher per sample",
            f"{OF}/beta_diversity.csv": "Bray-Curtis between samples",
        },
        "avoid": {"*.html": "Krona/Sankey/MultiQC reports are 0.2–5 MB of JS; link them instead"},
        "checks": [
            "Check sample names in the outputs match the job's srr_libs / reads; files from another run can share the folder.",
            "Do not mix Kraken clade counts, direct counts and Bracken estimates. A species row's clade % "
            "is not its abundance when most reads stop at genus: if the genus DIRECT count is large "
            "(>30% of reads), species are unresolved — report Bracken species % and say so.",
            "Classification rate depends on the database, not read quality: isolate/culture >95%; human "
            "gut 50–85%; environmental can be <50%.",
            "Taxa under ~0.01% or <10–50 reads are at k-mer misclassification noise; do not call them contaminants.",
            "Shannon <0.5 with Berger-Parker >0.9 = one dominant organism (culture/isolate); gut "
            "communities are typically Shannon 2.5–4.5. Bray-Curtis <0.2 very similar, >0.7 very different.",
        ],
        "next_steps": [
            "Single dominant organism: assemble and annotate it.",
            "For SRA failures: check the accession exists and is public.",
        ],
    },
    "MetagenomeBinning": {
        "verified": True,
        "read_first": {
            f"{OF}/BinningReport.html": "per-bin table, good/bad bin counts and criteria (small)",
            f"{OF}/bins.stats.txt": "contigs kept, low-coverage, too small, placed/unplaced",
            f"{OF}/coverage.stats.txt": "TOTAL assembly: contigs found, letters (bp), coverage histogram",
            f"{OF}/params.txt": "assembler and read layout (paired/single)",
            f"{OF}/.bin.<N>.<taxid>/quality.json": "per-bin completeness, contamination, consistency, N50, CDS, tRNA, rRNA",
            f"{OF}/.bin.<N>.<taxid>/genome_quality_details.txt": "per-bin quality + Group line",
            f"{OF}/.bin.<N>.<taxid>/specialty-amrfinder.txt": "per-bin AMR genes",
            f"{OF}/.bin.<N>.<taxid>/amr-sir.txt": "per-bin predicted S/R",
        },
        "avoid": {f"{OF}/contigs.fasta": "full assembly", f"{OF}/unplaced.fasta": "MB of contigs (size only)"},
        "checks": [
            "Report the share of assembled bp that ended up in bins (bin sizes / coverage.stats.txt letters); "
            "a large unbinned share means the sample is NOT a near-pure culture.",
            "Fill each MIMAG criterion only from the value that holds it (rRNA and tRNA counts are in the "
            "bin's quality.json feature_summary); a role name like '16sRrnaNMeth' is a methyltransferase, "
            "not a 16S rRNA gene. Write 'not checked' for anything you did not read.",
            "'contigKept' in bins.stats.txt is contigs kept after filtering, not the assembly size.",
            "MIMAG tiers: high >90% complete, <5% contamination, 23S/16S/5S rRNA and ≥18 tRNA; "
            "medium ≥50% / <10%; low <50%. BV-BRC 'good bin' is commonly quoted as completeness ≥80%, "
            "fine consistency ≥87%, contamination ≤10% (unconfirmed).",
            "Bin depth ≥10× for a reliable draft; N50 >20–50 kb is a good draft.",
        ] + _AMR_RULES[:2],
        "next_steps": ["Annotate good bins individually; compare with references via Similar Genome Finder."],
    },
    "FastqUtils": {
        "verified": True,
        "read_first": {
            f"{OF}/<reads>.fastq_trimming_report.txt": "Trim Galore/Cutadapt: reads/pairs processed, removed, bp trimmed, adapter length histogram",
            f"{OF}/<reads>_ptrim_fastqc.html": "FastQC: Basic Statistics in the first ~3 KB of text, then module PASS/WARN/FAIL",
        },
        "avoid": {f"{OF}/*.fq.gz": "the reads themselves"},
        "checks": [
            "Retention: use pairs/reads removed and FastQC Total Sequences, not bp written. >95% retained "
            "is typical; flag <90%. Bases trimmed <5% typical.",
            "Cutadapt 'Reads with adapters' counts 1–3 bp chance matches at stringency 1; use the removed-length "
            "histogram (≥5 bp) or FastQC Adapter Content for real adapter content.",
            "Expected FastQC flags: per-base sequence content FAIL in the first ~10–15 bp for Nextera/Tn5 and "
            "random-primed RNA-seq; length distribution WARN after trimming. A per-sequence GC second peak "
            "suggests contamination. Duplication <20–30% fine for genomic DNA; higher expected for RNA-seq/amplicons.",
            "Report total yield (bp) and the depth it implies for the downstream use: 30–100× is adequate "
            "for a bacterial assembly; do not call it low.",
            "Retention = pairs written / pairs processed after the length filter, to two decimals (e.g. "
            "99.87%), not 100%.",
        ],
        "next_steps": ["Proceed to assembly/mapping with the trimmed reads."],
    },
    "RNASeq": {
        "verified": True,
        "read_first": {
            f"{OF}/bvbrc_rnaseq_report.html": "alignment rates per sample (small)",
            f"{OF}/gene_counts_matrix.tsv": "the LAST rows (__no_feature, __ambiguous, __too_low_aQual, __not_aligned) give the gene-assignment rate",
            f"{OF}/<contrast>.deseq2.tsv": "Gene_Name, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj — count over the WHOLE file",
            f"{OF}/sample_metadata.tsv": "samples and conditions",
            f"{OF}/<cond>/<sample>/<sample>.align_stdout": "unique vs multi-mapped reads",
        },
        "avoid": {f"{OF}/multiqc_report.html": "2.7 MB; link it"},
        "checks": [
            "Never count genes or DE genes from one page. A DESeq2 table is ~1,700+ rows and several pages; "
            "search_file stops at 200 matches, so it cannot count rows. If you did not read the whole "
            "table, give no total — say 'from the first N rows only'.",
            "Gene-assignment rate = sum of gene counts / (that sum + all __ rows) in gene_counts_matrix.tsv. "
            "__too_low_aQual counts reads with low mapping quality — mostly MULTI-mapped reads (often rRNA), "
            "so a large value is a problem to flag, not 'typical'. alignment_not_unique=0 does not mean no "
            "multi-mapping; take that from <sample>.align_stdout.",
            "Alignment to a bacterial reference >90% good. Multi-mapped >20–30% usually means rRNA was not depleted.",
            "Reads assigned to genes >60–70% good for rRNA-depleted bacterial libraries; flag <40% — a high "
            "alignment rate does not mean the reads were counted.",
            "Report DE counts at padj<0.05 (and with |log2FC|≥1) as up/down. >~30% of genes DE is a large "
            "shift: check normalisation and library sizes. n=2 per condition is low power. Flag library-size imbalance.",
            "Do not describe plots you have not seen.",
        ],
        "next_steps": ["Pathway/function enrichment of the DE genes; more replicates if n≤2."],
    },
    "MetagenomicReadMapping": {
        "verified": True,
        "read_first": {
            f"{OF}/kma.res": "per template: Template_Identity, Template_Coverage, Query_Identity, Query_Coverage, Depth, q_value, p_value",
            f"{OF}/kma.mapstat": "fragmentCount per template, total fragments, KMA command (-ID)",
            f"{OF}/MetagenomicReadMappingReport.html": "same columns as kma.res, for the user",
        },
        "avoid": {f"{OF}/kma.frag.gz": "per-read assignments"},
        "checks": [
            "Map every value to its column header — Query_Coverage sits next to Depth.",
            "Confident presence: template identity ≥90–95% and template coverage ≥90%. Identity <80%, "
            "coverage <80% or depth <5× is low confidence / possible cross-mapping.",
            "Multi-copy rRNA templates run 4–7× deeper than single-copy genes.",
            "CARD rRNA-mutation templates (23S, 16S) and housekeeping targets (gyrA, parC, rpoB, PBPs) show "
            "only that the gene is present; resistance needs specific SNPs (check kma.aln or run Variation).",
            "Look up gene functions rather than guessing them from the name (e.g. patA/patB are fluoroquinolone efflux).",
            "The total in kma.mapstat's header is the INPUT fragment count, not fragments mapped. "
            "Fraction on targets = sum of per-template fragmentCount / that total (often ~1% for "
            "shotgun metagenomes — normal, not a sign of a targeted sample).",
        ],
        "next_steps": ["Check resistance SNPs in kma.aln for target genes; assemble for gene context."],
    },
    "Homology": {
        "verified": True,
        "read_first": {
            f"{OF}/blast_out.txt": "tabular hits (~3 KB)",
            f"{OF}/blast_headers.txt": "column names",
            f"{OF}/blast_out.metadata.json": "query/database parameters incl. max hits",
        },
        "avoid": {f"{OF}/blast_out.archive": "raw archive"},
        "checks": [
            "If the hit count equals blast_max_hits the list is truncated — say so.",
            "Per hit: percent identity, query coverage ((qend−qstart+1)/qlen), e-value, bitscore. Ortholog-level "
            "≥95% identity and ≥90% coverage; homolog 40–90%; twilight 25–35% needs coverage ≥70% and e<1e-10.",
            "Group hits by genome (fig|<genome_id>…): several hits in one genome are paralogs, not different "
            "species. Look up genome names with search_data(genome); never infer organisms from ID prefixes.",
            "When the database was a genome group (db_genome_group in the job parameters), call "
            "get_group_ids(group_name=<last path segment>, group_type='genome') and report which genomes "
            "had no hit or no full-length ortholog. Count hits per genome exactly.",
        ],
        "next_steps": ["Raise blast_max_hits if truncated; build a GeneTree of the hits to separate orthologs from paralogs."],
    },
    "CodonTree": {
        "verified": True,
        "read_first": {
            f"{OF}/detail_files/{NAME}.analysisStats": "genes used, alignment length, genome count",
            f"{OF}/detail_files/{NAME}.genesPerGenome.txt": "genes per genome (incomplete genomes/MAGs have far fewer)",
            f"{OF}/detail_files/{NAME}.homologAlignmentStats.txt": "per-gene alignment stats",
            f"{OF}/detail_files/{NAME}_treeWithGenomeNames.nwk": "tree with organism names and support",
            f"{OF}/detail_files/RAxML_info.{NAME}": "search_file 'identical' and 'Final ML' for warnings and likelihood",
            f"{OF}/{NAME}_report.html": "report for the user",
        },
        "avoid": {f"{OF}/detail_files/{NAME}.afa": "concatenated alignment (MB)"},
        "checks": [
            "Genes used vs single-copy genes available (report both; few genes = coarse tree).",
            "Bootstrap bands: ≥95 strong, 70–94 moderate, 50–69 weak, <50 unsupported. Report the share of weak nodes.",
            "Many branches at RAxML's 1e-6 floor = near-identical genomes, topology unresolved there.",
            "Tip branches >100× the median = outlier genome (contamination, MAG, misidentification).",
            "RAxML 'identical sequences' warnings: the order of identical genomes is arbitrary.",
            "Name organisms only from the tree-with-names file or a genome lookup, never from ID prefixes.",
        ],
        "next_steps": ["Increase number_of_genes; drop outlier/incomplete genomes and re-run."],
    },
    "GeneTree": {
        "verified": True,
        "read_first": {
            f"{OF}/{NAME}_raxml_log.txt": "sequences, columns, patterns, gaps %, model, identical-sequence warnings",
            f"{OF}/{NAME}_raxml_rell_tree.nwk": "tree with RELL support (<6 KB)",
            f"{OF}/{NAME}_aligned.fa": "alignment: ungapped length per sequence",
            f"{OF}/{NAME}_gene_tree_report.html": "report for the user",
        },
        "avoid": {},
        "checks": [
            "Sequence IDs look like PATRIC.<genome_id>.<contig>.CDS…; look genomes up with search_data(genome) — never name genera from IDs.",
            "Ungapped length outliers (<50% of the median) are fragments and inflate gap %. Gap % >30% suggests fragments or mixed domains.",
            "Internal branches >1 substitution/site, or genomes contributing >1 sequence, indicate paralogs or a mixed family — "
            "separate orthologs before reading the tree as species history.",
            "RELL support runs slightly higher than standard bootstrap; use the CodonTree bands cautiously.",
            "Different genome ids with the same prefix (e.g. 487514.3 and 487514.18) are different genomes, "
            "not paralogs. RAxML may mention an '.reduced' alignment; it is not saved in the output folder.",
        ],
        "next_steps": ["Re-run with trim_threshold / gap_threshold set; remove fragments and paralogs."],
    },
    "Variation": {
        "verified": True,
        "read_first": {
            f"{OF}/summary.txt": "reads, mapping %, coverage mean/median, ≤10× and zero-coverage fractions, variant counts",
            f"{OF}/all.var.tsv": "variants with Type, snpEff_impact, gene, Var_frac, Var_cov (~15 KB)",
            f"{OF}/Text_Files_Circular_Viewer/all.High_var_circular_view_input.txt": "HIGH-impact variants",
        },
        "avoid": {f"{OF}/*.bam": "alignments (hundreds of MB)", f"{OF}/*.consensus.fa": "consensus genome"},
        "checks": [
            "Properly mapped ≥95%. Median coverage ≥30×; mean/median >2 means skewed coverage — report the median.",
            ">5% of bases at ≤10× warns; >5% zero coverage suggests a divergent strain or deletions.",
            "High-quality variant count vs relatedness: <100 SNPs to the reference = same lineage.",
            "Report counts by Type and snpEff_impact and list HIGH-impact genes. Var_frac clusters at 0.3–0.7 "
            "suggest a mixed population or duplicated region (tentative). Var_cov <10 is low support.",
        ],
        "next_steps": ["Check HIGH-impact genes against phenotype; choose a closer reference if divergence is high."],
    },
    "ComparativeSystems": {
        "verified": True,
        "read_first": {
            f"{OF}/report.txt": "genomes with pathway/subsystem/protein-family data (186 B)",
            f"{OF}/{NAME}_subsystems_variant_mtx.tsv": "per subsystem × genome: active / likely / inactive (~46 KB; two pages)",
            f"{OF}/{NAME}_subsystems_tables.json": "first 16 KB only: 'overview' block with per-superclass counts",
        },
        "avoid": {
            f"{OF}/{NAME}_pathways.tsv": "5 MB per-feature dump — never scan or search per genome",
            f"{OF}/{NAME}_subsystems.tsv": "5 MB per-feature dump",
            f"{OF}/{NAME}_pathways_tables.json": "11 MB viewer data",
            f"{OF}/{NAME}_proteinfams_tables.json": "2+ MB viewer data",
        },
        "checks": [
            "Report per-genome active/likely/inactive subsystem totals exactly (count the status column), "
            "and name the subsystems that differ. Only claim a difference after checking that row's "
            "per-genome statuses actually differ.",
            "'Inactive' means not all roles were found, not proof of absence. '-- gjo' variant subsystems mirror a base subsystem.",
            "For pathway counts per genome, one facet_query on the pathway collection by genome_id is far cheaper than the files.",
        ],
        "next_steps": ["Open the job in BV-BRC's comparative systems viewer (the *_tables.json files feed it)."],
    },
    # ------------------------------------------------------------------
    # Not yet verified against real job folders (viral services were out
    # of scope for the 2026-10-08 baseline). Templates are the old ones.
    # ------------------------------------------------------------------
    "WholeGenomeSNPAnalysis": {
        "verified": False,
        "read_first": {f"{OF}/WholeGenomeSNP_Report.html": "report"},
        "avoid": {}, "checks": [], "next_steps": [],
    },
    "ViralAssembly": {
        "verified": False,
        "read_first": {
            "${params.output_path}/" + NAME + ".contigs.fasta": "contigs",
            "${params.output_path}/" + NAME + ".assembly_report.txt": "assembly report",
        },
        "avoid": {}, "checks": [], "next_steps": [],
    },
    "SARS2Assembly": {
        "verified": False,
        "read_first": {
            "${params.output_path}/" + NAME + ".consensus.fasta": "consensus",
            "${params.output_path}/" + NAME + ".analysis_report.txt": "analysis report",
        },
        "avoid": {}, "checks": [], "next_steps": [],
    },
}

# Rules that apply to every job. Injected into the system prompt.
GENERAL_RULES = [
    "VERIFY THE RUN FIRST. get_job_details returns an output_check per job. If its verdict is not "
    "'ok', say so before anything else: 'failed' — report the failure (read the job_failed_report) "
    "and do not analyse leftover files; 'outputs_from_other_run' / 'stale_files' — the folder holds "
    "files from another run with the same output name. BV-BRC lists failed jobs as 'completed', so "
    "never infer success from the status. When you browse instead, job_result entries carry "
    "job.success (1 = succeeded, 0 = failed).",
    "Check that sample names / accessions in the outputs match the job parameters.",
    "Read the small summary files listed by get_expected_outputs first; never scan or page "
    "through multi-MB tables or HTML reports. Make few tool calls per step.",
    "Never extrapolate counts from one page. Compare bytes read with total_size; page to the end "
    "or use search_file with max_matches and next_start, or state 'from the first N rows only'.",
    "Map every number to its column header before reporting it. Copy exact values; do not write "
    "'~' or '+' for numbers the file gives exactly, and make sure counts you list add up.",
    "Report only values you read. For any checklist (MIMAG tiers, quality criteria) write 'not "
    "checked' for items whose file you did not read.",
    "Gene functions and locus tags: quote them from the file (product/annotation columns) or look "
    "them up; never paraphrase a function from memory or build a locus tag from a peg number.",
    "Name organisms only from file contents or a search_data(genome) lookup — never from genome "
    "ID prefixes or your own guesses.",
    "Links: a workspace file or folder opens at https://www.bv-brc.org/workspace followed by its full "
    "path (e.g. https://www.bv-brc.org/workspace/user@patricbrc.org/home/run/.out/report.html); use "
    "workspace_browser_url when a tool result gives one. Write real URLs, never a placeholder, and "
    "never raw shock_api URLs (they need a token and fail in a browser).",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_expected_outputs(service_name: str) -> dict[str, Any]:
    """Return what to read, what to avoid and how to judge one service's outputs.

    Keys: service_name, known, verified, read_first {template: what it holds},
    avoid {template: why}, checks [...], next_steps [...], general_rules [...].
    Templates use ${output_folder} = ${params.output_path}/.${params.output_file}.
    """
    entry = SERVICE_KNOWLEDGE.get(service_name)
    if entry is None:
        return {
            "service_name": service_name,
            "known": False,
            "message": (
                "No stored knowledge for this service. Browse the output folder, read the "
                "smallest summary/report files first, and avoid multi-MB files."
            ),
            "general_rules": GENERAL_RULES,
        }
    return {
        "service_name": service_name,
        "known": True,
        "verified": entry["verified"],
        "path_note": "${output_folder} = ${params.output_path}/.${params.output_file}",
        "read_first": entry["read_first"],
        "avoid": entry["avoid"],
        "checks": entry["checks"],
        "next_steps": entry["next_steps"],
    }


def resolve_output_path(template: str, output_path: str, output_file: str) -> str:
    """Resolve a template using actual output_path and output_file values."""
    resolved = template.replace("${output_folder}", f"{output_path.rstrip('/')}/.{output_file}")
    resolved = resolved.replace("${params.output_path}", output_path)
    resolved = resolved.replace("${params.output_file}", output_file)
    return resolved


def build_service_knowledge_text() -> str:
    """Short system-prompt block: general rules + the services we know.

    Per-service detail is fetched with get_expected_outputs so the system
    prompt stays small.
    """
    lines = ["=== SERVICE OUTPUT KNOWLEDGE ===", "", "Rules for every job:"]
    lines += [f"- {r}" for r in GENERAL_RULES]
    lines.append("")
    known = sorted(k for k, v in SERVICE_KNOWLEDGE.items() if v["verified"])
    unverified = sorted(k for k, v in SERVICE_KNOWLEDGE.items() if not v["verified"])
    lines.append(
        "Call get_expected_outputs(service_name) before reading files. It returns which small "
        "files to read first, which large files to avoid, the checks and typical ranges to "
        "judge the results against, and next steps. Detailed knowledge: " + ", ".join(known) + "."
    )
    if unverified:
        lines.append("File templates only (unverified): " + ", ".join(unverified) + ".")
    lines.append(
        "For other services, browse the output folder and read the smallest summary files first."
    )
    return "\n".join(lines)
