# drWiggle 🧬🔬🤖 <0xF0><0x9F><0xAA><0xB6>

**Protein Flexibility Classification Framework**

[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)]() <!-- Placeholder: Replace with actual status badge if CI/CD is set up -->
[![Code Coverage](https://img.shields.io/badge/Coverage-N/A-lightgrey)]() <!-- Placeholder: Replace with actual coverage badge -->
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) <!-- Make sure you have a LICENSE file -->
[![PyPI version](https://img.shields.io/badge/PyPI-v1.0.0-orange)]() <!-- Update version as needed -->

**drWiggle** is a Python framework designed to classify protein residue flexibility based on structural features and Root Mean Square Fluctuation (RMSF) data derived from molecular dynamics simulations or other sources. It transforms continuous RMSF values into discrete flexibility categories and provides an end-to-end pipeline for data processing, model training (Random Forest, Neural Networks with HPO), evaluation, PDB feature integration, visualization (including structural mapping), and temperature-dependent analysis.

## Key Features ✨

*   **RMSF Binning 📊:** Converts continuous RMSF into a configurable number of discrete classes (default: 5, Very Rigid to Very Flexible) using K-means or Quantile-based strategies.
*   **Feature Engineering 🛠️:** Extracts relevant features including sequence context (windowing), amino acid properties, PDB structural properties (B-factor, SS, ACC, Dihedrals), and normalized values.
*   **Machine Learning Models 🧠:** Includes Random Forest and Feed-Forward Neural Network classifiers implemented using Scikit-learn and PyTorch.
*   **Hyperparameter Optimization (HPO) 📈:** Integrated support via Scikit-learn's `RandomizedSearchCV` (for RF) and `Optuna` (for NN) to find optimal model parameters.
*   **PDB Integration <0xF0><0x9F><0xAA><0xBD>:** Fetches PDB files, extracts features directly using BioPython, and requires external DSSP installation for secondary structure and accessibility calculations.
*   **Temperature Analysis 🔥❄️:** Facilitates comparison of classifications and model performance across different temperatures (requires results from multiple runs). Calculates transition matrices.
*   **Command-Line Interface (CLI) 💻:** User-friendly CLI powered by Click for easy pipeline execution (`train`, `evaluate`, `predict`, `process-pdb`, `compare-temperatures`, `analyze-distribution`).
*   **Configuration ⚙️:** Pipeline behavior controlled via YAML configuration files (`default_config.yaml`) with overrides via environment variables (`DRWIGGLE_*`) or CLI parameters.
*   **Visualization 🎨:** Generates plots for RMSF distributions, confusion matrices, feature importance, class distributions, and metric comparisons. Creates PyMOL scripts (`.pml`) and colored PDB files (via B-factor) to map flexibility onto protein structures.
*   **Modularity & Readability ✅:** Code structured into logical modules (data, models, utils, etc.) following Python best practices with type hinting and logging.

## Project Workflow Diagram 🐍 → 📊

This diagram illustrates the data flow and key processes within the `drWiggle` framework:

```mermaid
graph TD
    subgraph Inputs
        A[Input Data (.csv)];
        B[PDB ID/File];
        C[Configuration (YAML/CLI/ENV)];
    end

    subgraph Processing & Training
        D(Load & Process Data);
        E(Parse PDB);
        F(Extract PDB Features);
        G(Calculate Bin Boundaries);
        H(Apply Binning);
        I(Split Data);
        J(Train Models + HPO);
    end

    subgraph Evaluation & Prediction
        K(Evaluate Models);
        L(Predict on New Data);
        M(Compare Temperatures);
        N(Process Single PDB);
    end

    subgraph Outputs
        O[Trained Models (.joblib)];
        P[Binner Object (.joblib)];
        Q[Predictions (.csv)];
        R[Evaluation Reports (.csv/.json)];
        S[Plots (.png)];
        T[Colored PDB / PyMOL Script (.pdb/.pml)];
        U[Temperature Comparison Results];
    end

    %% Connections
    C --> D;
    C --> E;
    C --> F;
    C --> G;
    C --> H;
    C --> I;
    C --> J;
    C --> K;
    C --> L;
    C --> M;
    C --> N;

    A --> D;
    B --> E;
    E --> F;
    F --> D;

    D -- RMSF Values --> G;
    G --> P;
    P --> H;
    D --> H;

    H --> I;
    I -- Train Set --> J;
    I -- Validation Set --> J;
    I -- Test Set --> K;

    J --> O;

    O --> K;
    O --> L;
    O --> N;
    P --> K;  % Need binner for evaluation consistency
    P --> N;  % Need binner for PDB processing if based on bins

    K -- Metrics --> R;
    K -- Confusion Matrix Data --> S;

    L -- Input Features --> O;
    L --> Q;

    N -- PDB Structure & Features --> L;
    N -- Predictions --> T;
    N -- Model --> O;

    M -- Needs Multiple Run Outputs --> U;
    U -- Metrics & Transitions --> R;
    U -- Plots --> S;
```

## Inputs & Outputs 📥📤

This table summarizes the main inputs required and outputs generated by the framework:

| Category      | Item                              | Format / Type          | Description                                                                 | Related Command(s)                     |
| :------------ | :-------------------------------- | :--------------------- | :-------------------------------------------------------------------------- | :------------------------------------- |
| **Inputs**    | Configuration File                | YAML                   | Controls all pipeline parameters (paths, binning, features, models, etc.).  | *All*                                  |
|               | Training/Input Feature Data       | CSV                    | Contains residue features and target RMSF values (for training/binning).    | `train`, `predict`, `analyze-dist`     |
|               | Prediction Input Data             | CSV                    | Contains required features for model prediction.                            | `predict`                              |
|               | PDB ID / File                     | String / Path          | Protein structure for feature extraction and visualization.                 | `process-pdb`                          |
|               | CLI Options / ENV Vars            | String / Variables     | Overrides settings from configuration files.                                | *All*                                  |
|               | Saved Model (for prediction)      | `.joblib` / `.pt`      | Pre-trained model file.                                                     | `predict`, `evaluate`, `process-pdb`   |
|               | Saved Binner (for prediction)     | `.joblib`              | Pre-calculated binner file (needed if prediction involves non-binned RMSF). | `predict` (implicitly), `evaluate`     |
| **Outputs**   | Trained Model                     | `.joblib` / `.pt`      | Serialized model object (RF, NN state dict + scaler).                       | `train`                                |
|               | Binner Object                     | `.joblib`              | Serialized binner object containing calculated boundaries.                  | `train`                                |
|               | Predictions                       | CSV                    | Predicted flexibility classes (and optionally probabilities) per residue.   | `predict`, `evaluate`                  |
|               | Evaluation Summary                | CSV                    | Table comparing metrics (accuracy, F1, etc.) across evaluated models.       | `evaluate`                             |
|               | Detailed Reports                  | JSON / CSV             | Classification report (per-class metrics), Confusion Matrix (counts).     | `evaluate`                             |
|               | Plots                             | PNG                    | Visualizations (RMSF dist, CM, feat importance, metric trends, etc.).     | `train`, `evaluate`, `analyze-dist`... |
|               | Colored PDB / PyMOL Script        | PDB / PML              | Structure files colored by predicted flexibility for visualization software.  | `process-pdb`                          |
|               | Temperature Comparison Results    | CSV / PNG              | Combined metrics, transition matrices, comparison plots.                   | `compare-temperatures`                 |
|               | Extracted PDB Features            | CSV                    | Intermediate file containing features extracted by `process-pdb`.           | `process-pdb`                          |
|               | Log File                          | `.log` (Optional)      | Detailed execution logs (if file handler is configured).                    | *All*                                  |

## Project Structure 🌳

```
drwiggle_project/      # Top-level directory for your project instance
├── data/              # Input data files (e.g., CSVs with RMSF)
├── models/            # Saved trained models (.joblib) and binners (.joblib)
├── output/            # Generated outputs (plots, reports, predictions)
├── pdb_cache/         # Cached downloaded PDB files
└── drwiggle/          # The installable package code lives here
    ├── drwiggle/      # Source code package
    │   ├── __init__.py
    │   ├── cli.py     # Command Line Interface logic (Click)
    │   ├── config.py  # Configuration loading and helpers
    │   ├── pipeline.py # Main workflow orchestration
    │   ├── default_config.yaml # Default settings
    │   ├── data/      # Data loading, processing, binning
    │   │   ├── __init__.py
    │   │   ├── binning.py
    │   │   ├── loader.py
    │   │   └── processor.py
    │   ├── models/    # ML model implementations
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── neural_network.py
    │   │   └── random_forest.py
    │   ├── utils/     # Utility functions
    │   │   ├── __init__.py
    │   │   ├── helpers.py
    │   │   ├── metrics.py
    │   │   ├── pdb_tools.py
    │   │   └── visualization.py
    │   └── temperature/ # Temperature comparison logic
    │       ├── __init__.py
    │       └── comparison.py
    ├── setup.py       # Installation script
    ├── README.md      # This file
    └── tests/         # Unit and integration tests (placeholders)
        └── ...
```

## Installation 🚀

1.  **Clone the repository or create the structure:** Ensure you have the `drwiggle_project` directory containing the inner `drwiggle` package directory as shown above.

2.  **Navigate to the `drwiggle` package directory:**
    ```bash
    cd path/to/drwiggle_project/drwiggle
    ```

3.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # On Linux/macOS
    source venv/bin/activate
    # On Windows
    .\venv\Scripts\activate
    ```

4.  **Upgrade pip and Install the package:**
    ```bash
    python -m pip install --upgrade pip setuptools wheel
    pip install -e .
    ```
    *(The `-e .` installs the package in "editable" mode, linking directly to your source code, which is great for development.)*

5.  **(Optional but Recommended) Install DSSP:** For calculating secondary structure and solvent accessibility from PDB files (`process-pdb` command), install the `dssp` executable (e.g., via `apt`, `yum`, `conda`, `brew`, or from source) and ensure it's in your system's PATH. Alternatively, specify the full path in your configuration file (`pdb.dssp_path`).

## Configuration ⚙️

The pipeline's behavior is controlled primarily by `drwiggle/drwiggle/default_config.yaml`. You can customize runs by overriding these defaults using the following methods (highest precedence first):

1.  **Command-line parameters:** Specific options like `--temperature VALUE` or generic overrides like `--param dataset.split.test_size=0.25`.
2.  **Environment variables:** Prefix keys with `DRWIGGLE_`, use `__` for nesting (e.g., `export DRWIGGLE_BINNING__METHOD=quantile`).
3.  **Custom YAML file:** Pass a user-defined YAML file using `drwiggle --config my_config.yaml ...`.
4.  **Default config:** The settings in `drwiggle/drwiggle/default_config.yaml`.

**Important Note on Paths:** Paths defined in the configuration (`paths` section) are typically *relative to the directory where you execute the `drwiggle` command*. The recommended practice is to run `drwiggle` commands from the main `drwiggle_project` directory.

## Usage Examples 🧑‍💻

Ensure your data (e.g., `temperature_320_train.csv`) is placed in the `drwiggle_project/data` directory. Run commands from the `drwiggle_project` directory.

*   **Train models:**
    ```bash
    # Train default models (all enabled in config) for temperature 320K
    # Uses data matching pattern in config (e.g., data/temperature_320_*.csv)
    # Outputs go to output/, models saved to models/
    drwiggle train --temperature 320

    # Train only Random Forest using a specific config, overriding binning method
    drwiggle train --model random_forest --config drwiggle/my_config.yaml --binning quantile --temperature 348
    ```

*   **Evaluate models:**
    ```bash
    # Evaluate models previously trained for 320K on the test split
    drwiggle evaluate --temperature 320

    # Evaluate specific models on an external test dataset
    drwiggle evaluate --model neural_network --input /path/to/external_test_features.csv --temperature 320
    ```

*   **Predict on new data:**
    ```bash
    # Predict using the saved random_forest model for 379K
    # Input file needs necessary features
    drwiggle predict --input data/my_new_protein_features.csv --temperature 379 --model random_forest --output output/my_new_protein_predictions.csv --probabilities
    ```

*   **Process a PDB file:**
    ```bash
    # Download 1AKE, extract features, predict flexibility at 320K, generate pymol script
    # Assumes a model for 320K exists in models/ and DSSP is installed
    drwiggle process-pdb --pdb 1AKE --temperature 320 --output-prefix output/pdb_vis/1AKE_flex --model random_forest

    # Process a local PDB file
    drwiggle process-pdb --pdb /path/to/my_protein.pdb --temperature 320 --output-prefix output/pdb_vis/my_protein_flex
    ```

*   **Compare temperatures:**
    ```bash
    # Analyze how classifications change across temperatures defined in config
    # Assumes train/evaluate has been run for multiple temperatures storing results in output/
    drwiggle compare-temperatures --output-dir output/temp_comparison --model random_forest
    ```

*   **Analyze RMSF distribution:**
    ```bash
    # Plot the RMSF distribution from a data file and show bin boundaries (requires saved binner)
    drwiggle analyze-distribution --input data/temperature_320_train.csv --temperature 320
    ```

➡️ Use `drwiggle <command> --help` for details on specific commands and options.

## Contributing 🤝

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request on the project repository. Adhering to code style (e.g., Black, Flake8) and adding tests for new features is appreciated.

## License 📜

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (assuming you add one).
```

**Key additions and changes:**

1.  **Title/Subtitle:** Clearly defined.
2.  **Emojis:** Added relevant emojis to headings and features.
3.  **Workflow Diagram:** Included a `mermaid` graph illustrating the process flow. This diagram is rendered automatically on platforms like GitHub.
4.  **Inputs & Outputs Table:** Created a detailed table clarifying expected inputs and generated outputs, along with the relevant CLI commands.
5.  **Project Structure:** Included the `tree`-like structure for clarity.
6.  **Installation:** Added step to upgrade pip/setuptools/wheel. Emphasized editable install. Added detail about DSSP.
7.  **Configuration:** Clarified the override precedence and the important note about relative paths.
8.  **Usage:** Provided more varied and descriptive examples.
9.  **Contributing/License:** Standard sections included. Added placeholder badges at the top (update these if you set up CI/CD, code coverage, etc.).