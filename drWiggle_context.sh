#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

OUTPUT_FILE="drWiggle_context.txt"
PROJECT_ROOT="." # Current directory

# Define paths based on the project structure relative to PROJECT_ROOT
# SOURCE_DIR is where the actual Python package source code lives
SOURCE_DIR="${PROJECT_ROOT}/drwiggle"
SETUP_FILE="${PROJECT_ROOT}/setup.py"
README_FILE="${PROJECT_ROOT}/README.md"
DEFAULT_CONFIG_FILE="${SOURCE_DIR}/default_config.yaml"

# --- Verification ---
echo "Verifying project structure from $(pwd)..."
CHECKS_PASSED=true
if [[ ! -d "$SOURCE_DIR" ]]; then echo "  - Error: Expected source directory not found: $SOURCE_DIR"; CHECKS_PASSED=false; fi
if [[ ! -f "$SETUP_FILE" ]]; then echo "  - Error: Expected setup file not found: $SETUP_FILE"; CHECKS_PASSED=false; fi
if [[ ! -f "$README_FILE" ]]; then echo "  - Error: Expected README file not found: $README_FILE"; CHECKS_PASSED=false; fi
if [[ ! -f "$DEFAULT_CONFIG_FILE" ]]; then echo "  - Error: Expected default config file not found: $DEFAULT_CONFIG_FILE"; CHECKS_PASSED=false; fi

if [[ "$CHECKS_PASSED" == "false" ]]; then
  echo "Error: Please ensure you are running this script from the root 'drwiggle_project' directory and all required files/dirs exist."
  exit 1
fi
echo "Project structure verified."
# --- End Verification ---

# --- Function to append file content ---
append_file_content() {
  local file_path="$1"
  local output_target="$2"
  local display_path="${file_path#${PROJECT_ROOT}/}" # Path relative to project root for display

  if [[ -f "$file_path" ]]; then
    echo "" >> "$output_target"
    echo "--- File: $display_path ---" >> "$output_target"
    echo "---------------------------------------" >> "$output_target"
    # Using sed to add a newline just in case the file doesn't end with one
    sed -e '$a\' "$file_path" >> "$output_target"
    echo "" >> "$output_target"
    echo "--- End File: $display_path ---" >> "$output_target"
  else
    echo "" >> "$output_target"
    echo "--- Warning: File not found during append: $display_path ---" >> "$output_target"
  fi
}


# --- Start Context File ---
echo "Generating project context file: $OUTPUT_FILE"
# Overwrite the file initially
cat << EOF > "$OUTPUT_FILE"
### drWiggle Project Context ###

**Project:** drWiggle - Protein Flexibility Classification Framework

**Description:**
A Python framework designed to classify protein residue flexibility based on Root Mean Square Fluctuation (RMSF) data and structural features. It transforms continuous RMSF values into discrete flexibility categories using methods like K-means or quantiles. The framework includes trainable classification models (Random Forest, Neural Network) with hyperparameter optimization capabilities. It integrates PDB feature extraction (B-factor, SS, ACC, dihedrals - requires DSSP for some) and offers visualization outputs (PyMOL scripts, plots). A key feature is the ability to analyze and compare flexibility across different temperatures. The entire workflow is managed via a command-line interface (CLI) and configurable through YAML files.

**Key Components:**
- **Configuration:** YAML-based (\`default_config.yaml\`, \`config.py\`)
- **Data Handling:** Loading (\`loader.py\`), Processing (\`processor.py\`), Binning (\`binning.py\`)
- **Models:** Base class (\`base.py\`), RandomForest (\`random_forest.py\`), NeuralNetwork (\`neural_network.py\`) with HPO (Optuna/RandomizedSearch)
- **Pipeline:** Orchestrator (\`pipeline.py\`)
- **CLI:** User interface (\`cli.py\` using Click)
- **Utilities:** Metrics (\`metrics.py\`), PDB Tools (\`pdb_tools.py\` using BioPython), Visualization (\`visualization.py\` using Matplotlib/Seaborn), Helpers (\`helpers.py\`)
- **Temperature Analysis:** Comparison logic (\`comparison.py\`)
- **Packaging:** \`setup.py\` for installation.

**Purpose of this File:**
This file concatenates the core source code and configuration files to provide context for understanding the project's implementation.

**Input Data shape from temperature_320_train.csv**
domain_id,resid,resname,rmsf_320,protein_size,normalized_resid,core_exterior,relative_accessibility,dssp,phi,psi,resname_encoded,core_exterior_encoded,secondary_structure_encoded,phi_norm,psi_norm,esm_rmsf,voxel_rmsf
1aabA00,1,GLY,1.0563009,83,0.0,exterior,1.0,C,360.0,61.8,8,1,2,2.0,0.3433333333333333,0.6741838,0.818848
1aabA00,2,LYS,0.950946,83,0.0121951219512195,exterior,0.9219512195121952,C,-148.1,55.8,14,1,2,-0.8227777777777777,0.31,0.84813493,0.97168
1aabA00,3,GLY,0.83723533,83,0.024390243902439,exterior,0.8214285714285714,C,-160.8,81.8,8,1,2,-0.8933333333333334,0.4544444444444444,0.84601504,0.545898
1aabA00,4,ASP,0.675135,83,0.0365853658536585,exterior,0.6012269938650306,C,-70.3,157.5,4,1,2,-0.3905555555555555,0.875,0.81490785,0.505371

=========================================
EOF

# --- Add Core Project Files ---
echo "" >> "$OUTPUT_FILE"
echo "--- Core Project Files ---" >> "$OUTPUT_FILE"

# Append core files using the function
append_file_content "$SETUP_FILE" "$OUTPUT_FILE"
append_file_content "$README_FILE" "$OUTPUT_FILE"
append_file_content "$DEFAULT_CONFIG_FILE" "$OUTPUT_FILE"


# --- Add Python Source Files from drwiggle/drwiggle ---
echo "" >> "$OUTPUT_FILE"
echo "=========================================" >> "$OUTPUT_FILE"
echo "      Python Source Code Files           " >> "$OUTPUT_FILE"
echo "=========================================" >> "$OUTPUT_FILE"

# Use find to get all .py files, exclude __init__.py, sort them, and process
find "$SOURCE_DIR" -type f -name '*.py' -not -name '__init__.py' | sort | while IFS= read -r FILE; do
    append_file_content "$FILE" "$OUTPUT_FILE"
done

# --- End Context File ---
echo "" >> "$OUTPUT_FILE"
echo "=========================================" >> "$OUTPUT_FILE"
echo "         End of Context File             " >> "$OUTPUT_FILE"
echo "=========================================" >> "$OUTPUT_FILE"

echo "Context file '$OUTPUT_FILE' created successfully in $(pwd)."

exit 0