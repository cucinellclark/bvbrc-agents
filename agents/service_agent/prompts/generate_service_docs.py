#!/usr/bin/env python3
"""
Generate service_reference.py from service_required_params.json and
service_mapping.json.

Usage:
    python -m service_agent.prompts.generate_service_docs

This reads the config files from the MCP server and generates a Python
file containing the SERVICE_REFERENCE string used in the system prompt.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _load_config(filename: str, config_dir: str) -> dict:
    """Load a JSON config file."""
    path = os.path.join(config_dir, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Service categories
SERVICE_CATEGORIES = {
    "Genomics": [
        "genome_assembly", "genome_annotation", "comprehensive_genome_analysis",
        "similar_genome_finder", "genome_alignment",
    ],
    "Phylogenomics": [
        "bacterial_genome_tree", "gene_tree", "core_genome_mlst",
        "whole_genome_snp",
    ],
    "Metagenomics": [
        "taxonomic_classification", "metagenomic_binning",
        "metagenomic_read_mapping", "metacats",
    ],
    "Transcriptomics": [
        "rnaseq", "expression_import",
    ],
    "Proteomics": [
        "proteome_comparison", "docking",
    ],
    "Variation Analysis": [
        "variation", "msa_snp_analysis",
    ],
    "Sequence Analysis": [
        "blast", "primer_design", "fastqutils",
    ],
    "Transposon Analysis": [
        "tnseq",
    ],
    "Viral Analysis": [
        "viral_assembly", "sars_genome_analysis", "sars_wastewater_analysis",
        "influenza_ha_subtype_conversion", "subspecies_classification",
    ],
    "Comparative Analysis": [
        "comparative_systems",
    ],
    "Submission": [
        "sequence_submission",
    ],
    "Utility": [
        "date",
    ],
}

# Short descriptions
SERVICE_DESCRIPTIONS = {
    "genome_assembly": "Assemble genomes from sequencing reads (Illumina, PacBio, Nanopore)",
    "genome_annotation": "Annotate assembled genomes with gene predictions and functional annotations",
    "comprehensive_genome_analysis": "Full pipeline: assembly, annotation, and quality analysis",
    "similar_genome_finder": "Find similar genomes in BV-BRC using Mash/MinHash",
    "genome_alignment": "Align multiple genomes using Mauve",
    "bacterial_genome_tree": "Build phylogenetic trees from bacterial genomes using codon trees",
    "gene_tree": "Build gene-level phylogenetic trees (RAxML, PhyML, FastTree)",
    "core_genome_mlst": "Core genome MLST analysis using chewBBACA",
    "whole_genome_snp": "Whole genome SNP analysis for phylogenomics",
    "taxonomic_classification": "Classify metagenomic reads taxonomically (Kraken2)",
    "metagenomic_binning": "Bin metagenomic assemblies into individual genomes",
    "metagenomic_read_mapping": "Map metagenomic reads to gene sets (VFDB, CARD)",
    "metacats": "Metagenomic comparative analysis (MetaCATS)",
    "rnaseq": "RNA-Seq analysis with differential expression",
    "expression_import": "Import gene expression datasets",
    "proteome_comparison": "Compare proteomes across genomes",
    "docking": "Molecular docking analysis",
    "variation": "SNP/variant calling from sequencing reads",
    "msa_snp_analysis": "Multiple sequence alignment and SNP analysis",
    "blast": "BLAST sequence homology search",
    "primer_design": "Design PCR primers",
    "fastqutils": "FASTQ quality control and preprocessing",
    "tnseq": "Transposon insertion sequencing analysis",
    "viral_assembly": "Assemble viral genomes from sequencing reads",
    "sars_genome_analysis": "SARS-CoV-2 genome assembly and analysis",
    "sars_wastewater_analysis": "SARS-CoV-2 wastewater surveillance analysis",
    "influenza_ha_subtype_conversion": "Influenza HA subtype conversion/classification",
    "subspecies_classification": "Viral subspecies/clade classification",
    "comparative_systems": "Compare pathways, subsystems, and protein families across genomes",
    "sequence_submission": "Submit sequences to public databases",
    "date": "Date/time utility service",
}


def generate_reference(config_dir: str) -> str:
    """Generate the SERVICE_REFERENCE string from config files."""
    params = _load_config('service_required_params.json', config_dir)
    mapping_data = _load_config('service_mapping.json', config_dir)
    mapping = mapping_data.get('friendly_to_api', mapping_data)

    lines = []
    lines.append("=== BV-BRC SERVICE REFERENCE ===")
    lines.append("")
    lines.append(f"Total services: {len(mapping)}")
    lines.append("")

    # Group by category
    for category, services in SERVICE_CATEGORIES.items():
        # Only include services that exist in the mapping
        available = [s for s in services if s in mapping]
        if not available:
            continue

        lines.append(f"--- {category} ---")
        lines.append("")

        for svc_name in available:
            api_name = mapping.get(svc_name, svc_name)
            desc = SERVICE_DESCRIPTIONS.get(svc_name, "")
            config = params.get(svc_name, {})

            lines.append(f"  {svc_name} (API: {api_name})")
            if desc:
                lines.append(f"    Description: {desc}")

            # Required params
            required = config.get("required_params", [])
            if required:
                lines.append(f"    Required: {', '.join(required)}")

            # Required one of
            req_one = config.get("required_one_of", [])
            if req_one:
                lines.append(f"    Required (at least one): {' | '.join(req_one)}")

            # Defaults
            defaults = config.get("defaults", {})
            if defaults:
                default_strs = [f"{k}={v}" for k, v in defaults.items()]
                lines.append(f"    Defaults: {', '.join(default_strs)}")

            # Enum params
            enums = config.get("enum_params", {})
            if enums:
                for param, values in enums.items():
                    val_strs = [str(v) for v in values]
                    lines.append(f"    Enum {param}: {', '.join(val_strs)}")

            # Conditional required
            cond = config.get("conditional_required", [])
            if cond:
                for c in cond:
                    when = c.get("when", {})
                    req = c.get("require", [])
                    req_one_of = c.get("require_one_of", [])
                    when_str = " AND ".join(f"{k}={v}" for k, v in when.items())
                    if req:
                        lines.append(f"    When {when_str}: requires {', '.join(req)}")
                    if req_one_of:
                        lines.append(f"    When {when_str}: requires one of {' | '.join(req_one_of)}")

            lines.append("")

    return "\n".join(lines)


def main():
    # Find config directory
    script_dir = Path(__file__).resolve().parent
    # Walk up to find mcp_server/config
    config_dir = script_dir.parent.parent.parent / "mcp_server" / "config"

    if not config_dir.exists():
        print(f"Config directory not found: {config_dir}", file=sys.stderr)
        sys.exit(1)

    reference = generate_reference(str(config_dir))

    # Write to service_reference.py
    output_path = script_dir / "service_reference.py"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('"""Auto-generated BV-BRC service reference for system prompt.\n\n')
        f.write('Generated by: python -m service_agent.prompts.generate_service_docs\n')
        f.write('Source: bvbrc-mcp-server/config/service_required_params.json\n')
        f.write('        bvbrc-mcp-server/config/service_mapping.json\n')
        f.write('"""\n\n')
        f.write('SERVICE_REFERENCE = """\\\n')
        f.write(reference)
        f.write('"""\n')

    print(f"Generated {output_path}")
    print(f"Reference length: {len(reference)} chars")


if __name__ == "__main__":
    main()
