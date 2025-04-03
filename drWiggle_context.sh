#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

OUTPUT_FILE="drWiggle_context.txt"
# Assume the script is run from the 'drwiggle_project' directory
PROJECT_ROOT="."
PACKAGE_DIR="$PROJECT_ROOT/drwiggle"
SOURCE_DIR="$PACKAGE_DIR/drwiggle"

# --- Verification ---
# Check if essential directories/files exist from where the script is run
if [[ ! -d "$SOURCE_DIR" || ! -f "$PACKAGE_DIR/setup.py" || ! -f "$PACKAGE_DIR/README.md" ]]; then
  echo "Error: Please run this script from the root 'drwiggle_project' directory."
  echo "       Ensure 'drwiggle/drwiggle/', 'drwiggle/setup.py', and 'drwiggle/README.md' exist."
  exit 1
fi

# --- Start Context File ---
echo "Generating project context file: $OUTPUT_FILE"
# Overwrite the file initially
cat << EOF > "$OUTPUT_FILE"
### drWiggle Project Context ###

**Project:** drWiggle - Protein Flexibility Classification Framework

**Description:**
A Python framework designed to classify protein residue flexibility based on Root Mean Square Fluctuation (RMSF) data and structural features. It transforms continuous RMSF values into discrete flexibility categories using methods like K-means or quantiles. The framework includes trainable classification models (Random Forest, Neural Network) with hyperparameter optimization capabilities. It integrates PDB feature extraction (B-factor, SS, ACC, dihedrals - requires DSSP for some) and offers visualization outputs (PyMOL scripts, plots). A key feature is the ability to analyze and compare flexibility across different temperatures. The entire workflow is managed via a command-line interface (CLI) and configurable through YAML files.

**Key Components:**
- **Configuration:** YAML-based (`default_config.yaml`, `config.py`)
- **Data Handling:** Loading (`loader.py`), Processing (`processor.py`), Binning (`binning.py`)
- **Models:** Base class (`base.py`), RandomForest (`random_forest.py`), NeuralNetwork (`neural_network.py`) with HPO (Optuna/RandomizedSearch)
- **Pipeline:** Orchestrator (`pipeline.py`)
- **CLI:** User interface (`cli.py` using Click)
- **Utilities:** Metrics (`metrics.py`), PDB Tools (`pdb_tools.py` using BioPython), Visualization (`visualization.py` using Matplotlib/Seaborn), Helpers (`helpers.py`)
- **Temperature Analysis:** Comparison logic (`comparison.py`)
- **Packaging:** `setup.py` for installation.

**Purpose of this File:**
This file concatenates the core source code and configuration files to provide context for understanding the project's implementation.

=========================================
EOF

# --- Add Core Project Files ---
CORE_FILES=(
  "$PACKAGE_DIR/setup.py"
  "$PACKAGE_DIR/README.md"
  "$SOURCE_DIR/default_config.yaml"
)

echo "" >> "$OUTPUT_FILE"
echo "--- Core Project Files ---" >> "$OUTPUT_FILE"

for FILE_PATH in "${CORE_FILES[@]}"; do
  RELATIVE_PATH="${FILE_PATH#$PROJECT_ROOT/}" # Get path relative to project root
  if [ -f "$FILE_PATH" ]; then
    echo "" >> "$OUTPUT_FILE"
    echo "--- File: $RELATIVE_PATH ---" >> "$OUTPUT_FILE"
    echo "---------------------------------------" >> "$OUTPUT_FILE"
    # Using sed to add a newline just in case the file doesn't end with one
    sed -e '$a\' "$FILE_PATH" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "--- End File: $RELATIVE_PATH ---" >> "$OUTPUT_FILE"
  else
    echo "" >> "$OUTPUT_FILE"
    echo "--- Warning: Core file not found: $RELATIVE_PATH ---" >> "$OUTPUT_FILE"
  fi
done

# --- Add Python Source Files from drwiggle/drwiggle ---
echo "" >> "$OUTPUT_FILE"
echo "=========================================" >> "$OUTPUT_FILE"
echo "      Python Source Code Files           " >> "$OUTPUT_FILE"
echo "=========================================" >> "$OUTPUT_FILE"

# Use find to get all .py files, exclude __init__.py, sort them, and process
find "$SOURCE_DIR" -type f -name '*.py' -not -name '__init__.py' | sort | while IFS= read -r FILE; do
    # Get relative path for display
    RELATIVE_PATH="${FILE#$PROJECT_ROOT/}"
    echo "" >> "$OUTPUT_FILE"
    echo "--- File: $RELATIVE_PATH ---" >> "$OUTPUT_FILE"
    echo "---------------------------------------" >> "$OUTPUT_FILE"
    # Using sed to add a newline just in case the file doesn't end with one
    sed -e '$a\' "$FILE" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "--- End File: $RELATIVE_PATH ---" >> "$OUTPUT_FILE"
done

# --- End Context File ---
echo "" >> "$OUTPUT_FILE"
echo "=========================================" >> "$OUTPUT_FILE"
echo "         End of Context File             " >> "$OUTPUT_FILE"
echo "=========================================" >> "$OUTPUT_FILE"

echo "Context file '$OUTPUT_FILE' created successfully in $(pwd)."

exit 0