"""Service output patterns and metric extraction hints.

Embeds the knowledge from OldAgents/mcp-server/bvbrc-mcp-server/config/service_outputs.json
plus metric extraction hints from the analysis agent plan. This gives the LLM
explicit knowledge of which files to look for per service and what metrics
to extract, avoiding blind directory browsing.

The output patterns use template variables:
  ${params.output_path}  - the workspace output directory
  ${params.output_file}  - the output file prefix
These are resolved at runtime using values from workflow_context steps.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Service output file patterns (from service_outputs.json)
# ---------------------------------------------------------------------------

SERVICE_OUTPUT_PATTERNS: dict[str, dict[str, str]] = {
    "GenomeAssembly2": {
        "contigs_fasta": "${params.output_path}/.${params.output_file}/${params.output_file}_contigs.fasta",
        "assembly_report": "${params.output_path}/.${params.output_file}/${params.output_file}_AssemblyReport.html",
    },
    "GenomeAnnotation": {
        "genome_file": "${params.output_path}/.${params.output_file}/${params.output_file}.genome",
        "genome_report": "${params.output_path}/.${params.output_file}/GenomeReport.html",
    },
    "ComprehensiveGenomeAnalysis": {
        "genome_report": "${params.output_path}/.${params.output_file}/FullGenomeReport.html",
        "genome_file": "${params.output_path}/.${params.output_file}/annotated.genome",
    },
    "Homology": {
        "blast_output": "${params.output_path}/.${params.output_file}/blast_out.txt",
        "blast_json": "${params.output_path}/.${params.output_file}/blast_out.json",
    },
    "CodonTree": {
        "tree_file": "${params.output_path}/.${params.output_file}/${params.output_file}_tree.nwk",
        "tree_report": "${params.output_path}/.${params.output_file}/.${params.output_file}_report.html",
    },
    "GeneTree": {
        "tree_file": "${params.output_path}/.${params.output_file}/${params.output_file}_raxml_rell_tree.nwk",
        "alignment_file": "${params.output_path}/.${params.output_file}/${params.output_file}_aligned.fa",
        "tree_report": "${params.output_path}/.${params.output_file}/${params.output_file}_tree_report.html",
    },
    "Variation": {
        "variation_table": "${params.output_path}/.${params.output_file}/all.var.tsv",
    },
    "RNASeq": {
        "multiqc_report": "${params.output_path}/.${params.output_file}/multiqc_report.html",
    },
    "TaxonomicClassification": {
        "classification_report": "${params.output_path}/.${params.output_file}/Taxonomic-Classification-Service-BVBRC_multiqc_report.html",
    },
    "MetagenomeBinning": {
        "binning_report": "${params.output_path}/.${params.output_file}/BinningReport.html",
    },
    "MetagenomicReadMapping": {
        "mapping_report": "${params.output_path}/.${params.output_file}/MetagenomicReadMappingReport.html",
    },
    "ComparativeSystems": {
        "pathways_table": "${params.output_path}/.${params.output_file}/${params.output_file}_pathways_tables.json",
        "subsystems_table": "${params.output_path}/.${params.output_file}/${params.output_file}_subsystems_tables.json",
        "proteinfams_table": "${params.output_path}/.${params.output_file}/${params.output_file}_proteinfams_tables.json",
    },
    "WholeGenomeSNPAnalysis": {
        "report": "${params.output_path}/.${params.output_file}/WholeGenomeSNP_Report.html",
    },
    "ViralAssembly": {
        "contigs_fasta": "${params.output_path}/${params.output_file}.contigs.fasta",
        "assembly_report": "${params.output_path}/${params.output_file}.assembly_report.txt",
    },
    "SARS2Assembly": {
        "consensus_fasta": "${params.output_path}/${params.output_file}.consensus.fasta",
        "analysis_report": "${params.output_path}/${params.output_file}.analysis_report.txt",
    },
    "FastqUtils": {
        "processed_reads": "${params.output_path}/${params.output_file}_processed",
        "qc_report": "${params.output_path}/${params.output_file}.qc_report.txt",
    },
}


# ---------------------------------------------------------------------------
# Metric extraction hints per service
# ---------------------------------------------------------------------------

SERVICE_METRIC_HINTS: dict[str, list[dict[str, str]]] = {
    "GenomeAssembly2": [
        {"metric_name": "N50", "unit": "bp", "hint": "Look in assembly report or contigs FASTA header"},
        {"metric_name": "Total contigs", "unit": "count", "hint": "Count sequences in contigs FASTA or read from report"},
        {"metric_name": "Total length", "unit": "bp", "hint": "Sum of contig lengths from report"},
        {"metric_name": "Largest contig", "unit": "bp", "hint": "Largest contig length from report"},
        {"metric_name": "GC%", "unit": "%", "hint": "GC content percentage from report"},
    ],
    "GenomeAnnotation": [
        {"metric_name": "CDS count", "unit": "count", "hint": "Number of coding sequences from GenomeReport"},
        {"metric_name": "Gene count", "unit": "count", "hint": "Total genes from GenomeReport"},
        {"metric_name": "tRNA count", "unit": "count", "hint": "Transfer RNA genes from GenomeReport"},
        {"metric_name": "rRNA count", "unit": "count", "hint": "Ribosomal RNA genes from GenomeReport"},
        {"metric_name": "Genome size", "unit": "bp", "hint": "Total genome size from GenomeReport"},
    ],
    "ComprehensiveGenomeAnalysis": [
        {"metric_name": "N50", "unit": "bp", "hint": "Assembly N50 from FullGenomeReport"},
        {"metric_name": "CDS count", "unit": "count", "hint": "Coding sequences from FullGenomeReport"},
        {"metric_name": "Gene count", "unit": "count", "hint": "Total genes from FullGenomeReport"},
        {"metric_name": "Genome size", "unit": "bp", "hint": "Total genome size from FullGenomeReport"},
    ],
    "Homology": [
        {"metric_name": "Number of hits", "unit": "count", "hint": "Total BLAST hits from blast_out"},
        {"metric_name": "Top hit organism", "unit": "", "hint": "Best-matching organism name"},
        {"metric_name": "Best E-value", "unit": "", "hint": "Lowest E-value from results"},
        {"metric_name": "Percent identity range", "unit": "%", "hint": "Min-max percent identity across hits"},
    ],
    "CodonTree": [
        {"metric_name": "Number of taxa", "unit": "count", "hint": "Leaf count from Newick tree"},
        {"metric_name": "Number of genes used", "unit": "count", "hint": "Genes used for tree construction from report"},
        {"metric_name": "Tree format", "unit": "", "hint": "Newick or other format"},
    ],
    "GeneTree": [
        {"metric_name": "Number of taxa", "unit": "count", "hint": "Leaf count from Newick tree"},
        {"metric_name": "Alignment length", "unit": "bp", "hint": "Length of multiple sequence alignment"},
        {"metric_name": "Model used", "unit": "", "hint": "Substitution model from RAxML"},
    ],
    "Variation": [
        {"metric_name": "Total variants", "unit": "count", "hint": "Total rows in all.var.tsv"},
        {"metric_name": "SNP count", "unit": "count", "hint": "SNP-type variants from TSV"},
        {"metric_name": "Indel count", "unit": "count", "hint": "Indel-type variants from TSV"},
    ],
    "RNASeq": [
        {"metric_name": "Read mapping rate", "unit": "%", "hint": "Mapping percentage from multiqc report"},
        {"metric_name": "Total reads", "unit": "count", "hint": "Total read count from multiqc report"},
        {"metric_name": "DE gene count", "unit": "count", "hint": "Differentially expressed genes"},
    ],
    "TaxonomicClassification": [
        {"metric_name": "Classification rate", "unit": "%", "hint": "Percentage of reads classified"},
        {"metric_name": "Top taxa", "unit": "", "hint": "Most abundant taxonomic groups"},
    ],
    "MetagenomeBinning": [
        {"metric_name": "Number of bins", "unit": "count", "hint": "Total bins from BinningReport"},
        {"metric_name": "Completeness", "unit": "%", "hint": "Bin completeness from report"},
        {"metric_name": "Contamination", "unit": "%", "hint": "Bin contamination from report"},
    ],
    "ViralAssembly": [
        {"metric_name": "Contig count", "unit": "count", "hint": "Number of assembled contigs"},
        {"metric_name": "Coverage", "unit": "x", "hint": "Read coverage depth"},
        {"metric_name": "Reference match", "unit": "", "hint": "Best matching reference genome"},
    ],
    "FastqUtils": [
        {"metric_name": "Reads before", "unit": "count", "hint": "Input read count"},
        {"metric_name": "Reads after", "unit": "count", "hint": "Output read count after processing"},
        {"metric_name": "Trimmed %", "unit": "%", "hint": "Percentage of reads trimmed"},
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_expected_outputs(service_name: str) -> dict[str, Any]:
    """Return the expected output file patterns and metric hints for a service.

    Args:
        service_name: The BV-BRC service app name (e.g. "GenomeAssembly2").

    Returns:
        Dict with keys:
          - "service_name": the service name
          - "output_patterns": dict mapping output_key -> file path template
          - "metric_hints": list of metric extraction hints
          - "known": bool indicating whether the service is in the knowledge base
    """
    patterns = SERVICE_OUTPUT_PATTERNS.get(service_name, {})
    hints = SERVICE_METRIC_HINTS.get(service_name, [])

    return {
        "service_name": service_name,
        "output_patterns": patterns,
        "metric_hints": hints,
        "known": bool(patterns),
    }


def resolve_output_path(
    template: str,
    output_path: str,
    output_file: str,
) -> str:
    """Resolve a template path using actual output_path and output_file values.

    Args:
        template: Path template with ${params.output_path} and ${params.output_file}.
        output_path: The actual workspace output directory.
        output_file: The actual output file prefix.

    Returns:
        Resolved path string.
    """
    resolved = template.replace("${params.output_path}", output_path)
    resolved = resolved.replace("${params.output_file}", output_file)
    return resolved


def build_service_knowledge_text() -> str:
    """Build a text block describing all known service output patterns.

    This is injected into the system prompt so the LLM knows what to look for.
    """
    lines = ["=== SERVICE OUTPUT KNOWLEDGE ===", ""]
    lines.append(
        "Below are the known output file patterns and key metrics for each "
        "BV-BRC service. Use get_expected_outputs(service_name) to get the "
        "specific patterns for a service, then resolve the path templates "
        "using output_path and output_file from the workflow context."
    )
    lines.append("")

    for service_name in sorted(SERVICE_OUTPUT_PATTERNS.keys()):
        patterns = SERVICE_OUTPUT_PATTERNS[service_name]
        hints = SERVICE_METRIC_HINTS.get(service_name, [])

        lines.append(f"**{service_name}**")
        lines.append(f"  Output files: {', '.join(patterns.keys())}")
        if hints:
            metric_names = [h["metric_name"] for h in hints]
            lines.append(f"  Key metrics: {', '.join(metric_names)}")
        lines.append("")

    lines.append(
        "For services not listed above, browse the output directory to "
        "discover and characterize the output files dynamically."
    )

    return "\n".join(lines)
