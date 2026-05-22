"""Lightweight service catalog for Phase 1 (Decomposition).

Phase 1 only needs service names, categories, and short descriptions --
NOT full parameter schemas. This keeps the context window small during
decomposition so the LLM can focus on planning structure.
"""

SERVICE_CATALOG = """\
=== BV-BRC SERVICE CATALOG ===

Below are all available BV-BRC services organized by category.
Use these names when creating workflow plans.

--- Genomics ---
  genome_assembly: Assemble genomes from sequencing reads (Illumina, PacBio, Nanopore)
  genome_annotation: Annotate assembled genomes with gene predictions and functional annotations
  comprehensive_genome_analysis: Full pipeline: assembly, annotation, and quality analysis
  similar_genome_finder: Find genomes in BV-BRC most similar to your sequence

--- Phylogenomics ---
  bacterial_genome_tree: Build phylogenetic tree from bacterial genome IDs
  gene_tree: Build gene-level phylogenetic tree
  whole_genome_snp: SNP analysis across whole genomes

--- Comparative Genomics ---
  comparative_systems: Compare metabolic pathways and subsystems across genomes
  proteome_comparison: Compare proteomes across genomes
  genome_alignment: Align genomes for structural comparison

--- Sequence Analysis ---
  blast: BLAST sequence similarity search
  msa_snp_analysis: Multiple sequence alignment and SNP analysis
  primer_design: Design PCR primers for target sequences

--- Metagenomics ---
  metagenomic_binning: Bin metagenomic reads into genome bins
  metagenomic_read_mapping: Map metagenomic reads to reference genomes
  taxonomic_classification: Classify reads by taxonomy

--- Transcriptomics ---
  rnaseq: RNA-Seq differential expression analysis
  expression_import: Import expression data

--- Variation Analysis ---
  variation: SNP/variant calling from reads against a reference genome

--- Specialized Viral ---
  viral_assembly: Assemble viral genomes from reads
  sars_genome_analysis: SARS-CoV-2 genome analysis pipeline
  sars_wastewater_analysis: SARS-CoV-2 wastewater analysis
  influenza_ha_subtype_conversion: Influenza HA subtype conversion

--- Other ---
  core_genome_mlst: Core genome MLST typing
  subspecies_classification: Subspecies classification
  tnseq: TnSeq transposon insertion analysis
  docking: Molecular docking simulation
  fastqutils: FASTQ file utilities (trim, filter, subsample)
  metacats: Metagenomic classification and typing
  sequence_submission: Submit sequences to public databases
  date: Date estimation for phylogenetic trees
"""
