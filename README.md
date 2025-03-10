# Configuration Guide for mdCATH Processor

## Configuration Overview

The mdCATH processor is configured through a YAML file that controls all aspects of the pipeline. This document provides detailed information about configuration options and best practices.

## Basic Configuration Setup

```bash
# Generate example configuration file
python -m mdcath.config

# Copy example configuration to active configuration
cp config.yaml.example config.yaml

# Edit configuration with your preferred editor
nano config.yaml  # or vim, emacs, etc.
```

## Configuration File Structure

The configuration file is organized into the following sections:

- `input`: Define data sources and selection criteria
- `output`: Specify where results should be saved
- `processing`: Control how data is processed
- `performance`: Optimize resource usage
- `logging`: Configure log detail levels

## Full Configuration Reference

Below is a complete configuration template with descriptions for all options:

```yaml
# Configuration file for mdCATH processor pipeline

input:
  # Base directory containing mdCATH HDF5 files 
  mdcath_dir: "/mnt/datasets/MD_CATH/data/"
  
  # List of domain IDs to process (empty list means process all available domains)
  domain_ids: []  
  
  # List of simulation temperatures to process
  temperatures: [320, 348, 379, 413, 450]

output:
  # Base directory for all outputs (user path is expanded automatically)
  base_dir: "~/drFelix/data/"
  
  # Directory for extracted PDB frames (relative to base_dir)
  pdb_frames_dir: "interim/aposteriori_extracted_pdb_frames_files"
  
  # Directory for RMSF data (relative to base_dir)
  rmsf_dir: "interim/per-residue-rmsf"
  
  # Directory for analysis results (relative to base_dir)
  summary_dir: "processed/mdcath_summary"
  
  # Path to log file (relative to base_dir)
  log_file: "pipeline.log"

processing:
  # Controls how trajectory frames are selected
  frame_selection:
    # Method for selecting representative frames:
    # - regular: Select frames at regular intervals
    # - rmsd: Select frames based on RMSD clustering
    # - gyration: Select frames at different gyration radii
    # - random: Select frames randomly
    method: "regular"
    
    # Number of frames to extract per temperature
    num_frames: 10
    
    # Clustering method for RMSD-based selection
    cluster_method: "kmeans"
  
  # Options for cleaning PDB files
  cleaning:
    # Replace chain identifier "0" with "A"
    replace_chain_0_with_A: true
    
    # Fix atom numbering (start from 1 and increment)
    fix_atom_numbering: true
    
    # Remove hydrogen atoms from PDB files
    remove_hydrogens: false
  
  # Data validation settings
  validation:
    # Check for missing residues in PDB files
    check_missing_residues: true
    
    # Verify RMSF values are correctly mapped to residues
    verify_rmsf_mapping: true
    
    # Compare results with reference data (if available)
    compare_to_reference: false
    
    # Path to reference data (if compare_to_reference is true)
    reference_path: ""

performance:
  # Number of CPU cores to use for processing
  # 0 means auto-detect (use all available cores)
  num_cores: 0
  
  # Number of files to process in one batch
  batch_size: 10
  
  # Memory limit in GB for processing
  memory_limit_gb: 16

logging:
  # Overall logging level
  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
  level: "INFO"
  
  # Logging level for console output
  console_level: "INFO"
  
  # Logging level for log file
  file_level: "DEBUG"
  
  # Whether to show progress bars in the console
  show_progress_bars: true
```

## Configuration Examples

### Minimal Configuration

For a simple setup with default options, you only need to specify the input and output paths:

```yaml
input:
  mdcath_dir: "/mnt/datasets/MD_CATH/data/"

output:
  base_dir: "~/drFelix/data/"
```

### High Performance Configuration

For processing large datasets on a high-performance system:

```yaml
input:
  mdcath_dir: "/mnt/datasets/MD_CATH/data/"
  
output:
  base_dir: "~/drFelix/results/mdcath_large_scale/"

performance:
  num_cores: 32  # Specify the exact number of cores to use
  batch_size: 50  # Process more domains in parallel
  memory_limit_gb: 128  # Increase memory limit for large systems

processing:
  frame_selection:
    method: "rmsd"  # More sophisticated frame selection
    num_frames: 20  # Extract more frames per temperature
```

### Minimal Output Configuration

For preliminary testing or when storage space is limited:

```yaml
input:
  mdcath_dir: "/mnt/datasets/MD_CATH/data/"
  domain_ids: ["1a1zA00", "1a3dA00", "1a0aA00"]  # Process only 3 domains
  temperatures: [320]  # Process only one temperature
  
output:
  base_dir: "~/drFelix/data/test_run/"

processing:
  frame_selection:
    num_frames: 5  # Extract fewer frames
```

## Tips and Best Practices

1. **Start small**: Begin with a few domains to verify everything works before scaling up.

2. **Paths**: Use absolute paths for `mdcath_dir` to avoid ambiguity. Relative paths in `output` are processed relative to `base_dir`.

3. **Memory management**: If encountering memory issues, reduce `batch_size` and ensure `memory_limit_gb` is set appropriately for your system.

4. **CPU cores**: Setting `num_cores` to 0 (auto-detect) is usually good, but specify a lower number if you want to leave resources for other tasks.

5. **Frame selection**: The `regular` method is fastest, while `rmsd` provides more representative frames but takes longer.

6. **Logging**: Set `file_level` to "DEBUG" for detailed logs when troubleshooting, but keep `console_level` at "INFO" to reduce terminal output.

7. **Domain selection**: When testing, explicitly list domains in `domain_ids` rather than processing all domains.

8. **Validation**: Keep validation options enabled (`check_missing_residues`, `verify_rmsf_mapping`) to ensure data quality.

## Command-Line Overrides

Many configuration options can be overridden at runtime via command-line arguments:

```bash
# Override domains and temperatures
mdcath extract-rmsf --config config.yaml --domains 1a1zA00 1a3dA00 --temps 320 348

# Override frame selection method and count
mdcath extract-frames --config config.yaml --method rmsd --num-frames 15

# These command-line arguments take precedence over config file settings
```