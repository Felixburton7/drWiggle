# --- File: drwiggle/utils/pdb_tools.py ---
# Corrected with debug print statements around BioPython import

import logging
import os
import re
import warnings
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

# Local imports (keep these first if possible, though sometimes order matters less)
# Defer config import if it causes circular issues, but seems okay here
from drwiggle.config import get_pdb_config, get_pdb_feature_config
from drwiggle.utils.helpers import ensure_dir

logger = logging.getLogger(__name__)

# --- DEBUG ---
print("DEBUG: Attempting to import BioPython in pdb_tools.py...")
# --- END DEBUG ---

# Biopython imports
try:
    from Bio.PDB import PDBParser, PDBIO, Select, Polypeptide
    from Bio.PDB.DSSP import DSSP
    from Bio.PDB.exceptions import PDBException
    from Bio.PDB.PDBList import PDBList
    from Bio.SeqUtils import seq1 # To convert 3-letter AA code to 1-letter
    _biopython_available = True
    # --- DEBUG ---
    print("DEBUG: BioPython imported successfully in pdb_tools.py.")
    # --- END DEBUG ---
except ImportError:
    # --- DEBUG ---
    print("DEBUG: ImportError caught in pdb_tools.py.")
    # --- END DEBUG ---
    # Log the warning using the logger setup by the main application
    # Check if a logger exists before trying to use it during initial import phase
    try:
        logging.getLogger(__name__).warning("BioPython not found. PDB processing features will be unavailable. Install with `pip install biopython`.")
    except Exception: # Catch potential logging setup issues during import
        print("Warning: BioPython not found (logging not fully configured yet). Install with `pip install biopython`.")

    # Define dummy classes/functions to avoid errors if module is imported but BP not installed
    class PDBParser: pass
    class PDBIO: pass
    class Select: pass
    class Polypeptide: pass
    class DSSP: pass
    class PDBException(Exception): pass
    class PDBList: pass
    def seq1(res): return 'X'
    _biopython_available = False


# --- PDB Parsing and Feature Extraction ---

def fetch_pdb(pdb_id: str, cache_dir: str) -> Optional[str]:
    """
    Downloads a PDB file if not already cached.

    Args:
        pdb_id: The 4-character PDB ID.
        cache_dir: The directory to store/retrieve PDB files.

    Returns:
        The path to the cached PDB file (format .pdb), or None if download fails.
    """
    if not _biopython_available:
         # Use logger now as it should be configured when function is called
         logger.error("BioPython PDBList not available for fetching PDB files.")
         return None

    ensure_dir(cache_dir)
    pdb_list = PDBList(pdb=cache_dir, obsolete_pdb=cache_dir, verbose=False)
    # Explicitly request pdb format, adjust filename handling
    try:
        # retrieve_pdb_file returns the path it *would* have if downloaded/cached
        expected_path = pdb_list.retrieve_pdb_file(pdb_id, pdir=cache_dir, file_format='pdb')

        # Check if the file actually exists after retrieve_pdb_file call
        if os.path.exists(expected_path):
            logger.info(f"PDB file for {pdb_id} found/downloaded at: {expected_path}")
            return expected_path
        else:
             # Sometimes PDBList doesn't error but fails to download
             logger.error(f"Failed to retrieve PDB file for {pdb_id} (expected path: {expected_path}). Check ID and network.")
             return None

    except Exception as e:
        logger.error(f"Error retrieving PDB file for {pdb_id}: {e}", exc_info=True)
        return None

def parse_pdb(pdb_path_or_id: str, pdb_config: Dict[str, Any]) -> Optional[Any]:
    """
    Parses a PDB file using BioPython's PDBParser. Handles fetching if ID is given.

    Args:
        pdb_path_or_id: Path to the PDB file or a 4-character PDB ID.
        pdb_config: PDB configuration dictionary (must contain 'pdb_cache_dir').

    Returns:
        Bio.PDB Model object (the first model found), or None if parsing/fetching fails.
    """
    if not _biopython_available:
         logger.error("BioPython PDBParser not available for parsing PDB files.")
         return None

    pdb_id_pattern = re.compile(r"^[a-zA-Z0-9]{4}$")
    pdb_path = None
    structure_id = "structure" # Default ID for the structure object

    if os.path.isfile(pdb_path_or_id):
        pdb_path = os.path.abspath(pdb_path_or_id)
        structure_id = os.path.splitext(os.path.basename(pdb_path))[0] # Use filename stem as ID
        logger.info(f"Parsing local PDB file: {pdb_path}")
    elif pdb_id_pattern.match(pdb_path_or_id):
        pdb_id = pdb_path_or_id.upper()
        structure_id = pdb_id # Use PDB ID as structure ID
        cache_dir = pdb_config.get('pdb_cache_dir')
        if not cache_dir:
             logger.error("pdb_cache_dir not specified in config. Cannot fetch PDB ID.")
             return None
        logger.info(f"Attempting to fetch PDB ID: {pdb_id} using cache: {cache_dir}")
        pdb_path = fetch_pdb(pdb_id, cache_dir)
        if not pdb_path: return None # Fetch failed
    else:
        logger.error(f"Invalid PDB input: '{pdb_path_or_id}'. Must be a valid file path or 4-character PDB ID.")
        return None

    parser = PDBParser(QUIET=True, STRUCTURE_BUILDER=Polypeptide.PolypeptideBuilder()) # Use builder for phi/psi
    try:
        structure = parser.get_structure(structure_id, pdb_path)
        logger.info(f"Successfully parsed PDB structure '{structure.id}'. Models: {len(structure)}")
        if len(structure) > 1:
             logger.warning(f"PDB file contains multiple models ({len(structure)}). Using only the first model (ID: {structure[0].id}).")
        if len(structure) == 0:
            logger.error(f"No models found in PDB structure '{structure.id}'. Cannot proceed.")
            return None
        return structure[0] # Return only the first model
    except PDBException as e:
        logger.error(f"Bio.PDB parsing error for {pdb_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing PDB file {pdb_path}: {e}", exc_info=True)
        return None


def extract_pdb_features(
    structure_model: Any, # Should be a Bio.PDB Model object
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Extracts features like B-factor, SS, ACC, Dihedrals from a Bio.PDB Model.

    Args:
        structure_model: The Bio.PDB Model object (typically structure[0]).
        config: The main configuration dictionary.

    Returns:
        DataFrame containing extracted features per residue.
    """
    if not _biopython_available:
        logger.error("BioPython not available, cannot extract PDB features.")
        return pd.DataFrame()

    pdb_config = get_pdb_config(config)
    feature_flags = get_pdb_feature_config(config)
    model_id = structure_model.id
    pdb_structure_id = structure_model.get_parent().id # Get the overall structure ID
    data = []

    # --- Run DSSP if needed ---
    dssp_results = None
    dssp_path = pdb_config.get('dssp_path') # Path to executable
    needs_dssp = feature_flags.get('secondary_structure') or feature_flags.get('solvent_accessibility')

    if needs_dssp:
        # DSSP requires a file path. Save the model temporarily.
        # Using a unique temp name based on structure and model ID
        temp_pdb_dir = os.path.join(pdb_config.get("pdb_cache_dir", "."), "temp") # Save in cache/temp
        ensure_dir(temp_pdb_dir)
        temp_pdb_path = os.path.join(temp_pdb_dir, f"_temp_{pdb_structure_id}_model_{model_id}.pdb")

        io = PDBIO()
        io.set_structure(structure_model)
        io.save(temp_pdb_path)
        logger.debug(f"Temporarily saved model {model_id} to {temp_pdb_path} for DSSP.")

        try:
            logger.info(f"Running DSSP (using path: {dssp_path or 'system PATH'})...")
            # Pass model object AND file path to DSSP constructor
            dssp_results = DSSP(structure_model, temp_pdb_path, dssp=dssp_path)
            logger.info(f"DSSP calculation successful for {len(dssp_results)} residues.")
        except FileNotFoundError as e:
             # Check if dssp_path was specified or if it failed from PATH search
             search_location = f"specified path '{dssp_path}'" if dssp_path else "system PATH"
             logger.error(f"DSSP executable not found at {search_location}. Cannot calculate SS/ACC. Error: {e}")
             logger.error("Please install DSSP (e.g., `sudo apt install dssp` or `conda install dssp`) "
                          "and ensure it's in your PATH, or set 'pdb.dssp_path' in config.")
             dssp_results = None # Ensure it's None if failed
        except PDBException as e: # Catch DSSP execution errors (e.g., invalid PDB for DSSP)
             logger.error(f"DSSP calculation failed for {temp_pdb_path}: {e}")
             dssp_results = None
        except Exception as e: # Catch other unexpected errors
             logger.error(f"Unexpected error running DSSP: {e}", exc_info=True)
             dssp_results = None
        finally:
             # Clean up temporary PDB file
             if os.path.exists(temp_pdb_path):
                  try: os.remove(temp_pdb_path)
                  except OSError: logger.warning(f"Could not remove temporary PDB file: {temp_pdb_path}")

    # --- Iterate through residues and extract features ---
    logger.info(f"Extracting features for Model ID: {model_id} of Structure: {pdb_structure_id}")
    residue_counter = 0
    # Note: Pre-calculating phi/psi list was removed for robustness, calculating per residue below

    for chain in structure_model:
        chain_id = chain.id
        for residue in chain.get_residues():
            res_id_tuple = residue.get_id() # tuple: (hetflag, resid, icode)
            resname = residue.get_resname()

            # Skip HETATMs and non-standard residues
            if res_id_tuple[0] != ' ':
                continue
            try:
                is_standard_aa = Polypeptide.is_aa(residue, standard=True)
            except Exception:
                is_standard_aa = False
            if not is_standard_aa:
                continue

            res_seq_id = res_id_tuple[1] # Residue sequence number
            res_icode = res_id_tuple[2].strip() # Insertion code

            residue_features = {
                "domain_id": pdb_structure_id,
                "chain_id": chain_id,
                "resid": res_seq_id,
                **({"icode": res_icode} if res_icode else {}),
                "resname": resname,
            }
            residue_counter += 1

            # Extract B-factor
            if feature_flags.get('b_factor'):
                ca_atom = residue.get("CA")
                if ca_atom:
                    bfactors = [ca_atom.get_bfactor()]
                else:
                     backbone_atoms = ['N', 'CA', 'C', 'O']
                     bfactors = [atom.get_bfactor() for atom_name, atom in residue.items() if atom_name in backbone_atoms]
                     if not bfactors:
                          bfactors = [atom.get_bfactor() for atom in residue if atom.element != 'H']
                residue_features['b_factor'] = np.mean(bfactors) if bfactors else 0.0

            # Extract DSSP features
            ss = '-'
            rsa = np.nan
            if dssp_results:
                 dssp_key = (chain_id, res_id_tuple)
                 if dssp_key in dssp_results:
                      dssp_data = dssp_results[dssp_key]
                      if feature_flags.get('secondary_structure'):
                           ss = dssp_data[2]
                      if feature_flags.get('solvent_accessibility'):
                           rsa = dssp_data[3]
                 else:
                      if not hasattr(extract_pdb_features, "_dssp_missing_logged"):
                           logger.warning(f"Residue {chain_id}:{res_id_tuple} not found in DSSP results. DSSP might skip residues. Subsequent warnings suppressed.")
                           extract_pdb_features._dssp_missing_logged = True

            residue_features['dssp'] = ss
            residue_features['relative_accessibility'] = rsa if not pd.isna(rsa) else None

            # Extract Dihedral angles
            if feature_flags.get('dihedral_angles'):
                 phi = None
                 psi = None
                 try:
                      phi = Polypeptide.calc_phi(residue)
                      psi = Polypeptide.calc_psi(residue)
                 except Exception as e:
                      logger.debug(f"Could not calculate phi/psi for {chain_id}:{res_id_tuple}: {e}")

                 residue_features['phi'] = np.degrees(phi) if phi is not None else None
                 residue_features['psi'] = np.degrees(psi) if psi is not None else None

            data.append(residue_features)

    # Reset DSSP logging flag
    if hasattr(extract_pdb_features, "_dssp_missing_logged"):
        del extract_pdb_features._dssp_missing_logged

    df = pd.DataFrame(data)
    logger.info(f"Extracted features for {residue_counter} standard residues across {len(structure_model)} chains.")

    # --- Post-processing ---
    if feature_flags.get('core_exterior_encoded'):
         logger.warning("Feature 'core_exterior_encoded' requested, but calculation logic is not implemented. Column will be missing or filled with UNK.")
         if 'relative_accessibility' in df.columns and not df['relative_accessibility'].isnull().all():
             threshold = 0.20
             df['core_exterior'] = df['relative_accessibility'].apply(lambda x: 'SURFACE' if pd.notna(x) and x >= threshold else 'CORE')
             logger.info(f"Assigned 'core_exterior' based on RSA threshold ({threshold}).")
         else:
              df['core_exterior'] = 'UNK'

    # Ensure required columns exist
    expected_cols = ['domain_id', 'chain_id', 'resid']
    if feature_flags.get('b_factor'): expected_cols.append('b_factor')
    if feature_flags.get('secondary_structure'): expected_cols.append('dssp')
    if feature_flags.get('solvent_accessibility'): expected_cols.append('relative_accessibility')
    if feature_flags.get('dihedral_angles'): expected_cols.extend(['phi', 'psi'])
    if feature_flags.get('core_exterior_encoded'): expected_cols.append('core_exterior')

    for col in expected_cols:
        if col not in df.columns:
            logger.warning(f"Expected feature column '{col}' not found after extraction. Adding column with NaN/defaults.")
            default_val = 0.0 if col == 'b_factor' else ('-' if col == 'dssp' else ('UNK' if col == 'core_exterior' else np.nan))
            df[col] = default_val

    return df

# --- PDB Visualization/Output ---

class ColorByFlexibilitySelect(Select):
    """Bio.PDB Selector to set B-factor based on predicted flexibility class."""
    def __init__(self, predictions_map: Dict[Tuple[str, int], int], default_b: float = 20.0):
        """
        Args:
            predictions_map: Dictionary mapping (chain_id, resid) to predicted_class.
            default_b: B-factor value for residues not in the predictions map.
        """
        self.predictions = predictions_map
        self.default_b = default_b
        self.class_to_bfactor = {
            0: 10.0,  # Very Rigid
            1: 25.0,  # Rigid
            2: 40.0,  # Moderately Flexible
            3: 60.0,  # Flexible
            4: 80.0,  # Very Flexible
        }
        max_class = max(self.class_to_bfactor.keys())
        max_b = self.class_to_bfactor[max_class]
        for i in range(max_class + 1, 10):
            self.class_to_bfactor[i] = max_b + (i - max_class) * 15.0

        logger.debug(f"B-factor mapping for coloring: {self.class_to_bfactor}")

    def accept_atom(self, atom) -> int:
        """Accepts the atom and sets its B-factor based on prediction."""
        residue = atom.get_parent()
        chain = residue.get_parent()
        res_id_tuple = residue.get_id()

        # Default to keeping original B-factor unless it's a standard AA we have a prediction for
        bfactor_to_set = atom.get_bfactor() # Start with original

        if res_id_tuple[0] == ' ':
            try: # Protect against non-standard residues causing errors in is_aa
                if Polypeptide.is_aa(residue.get_resname(), standard=True):
                    chain_id = chain.id
                    res_seq_id = res_id_tuple[1]
                    pred_key = (chain_id, res_seq_id)
                    predicted_class = self.predictions.get(pred_key)

                    if predicted_class is not None:
                         bfactor_to_set = self.class_to_bfactor.get(predicted_class, self.default_b)
                    else:
                         bfactor_to_set = self.default_b # Apply default if not found
                         if not hasattr(ColorByFlexibilitySelect, "_missing_logged"):
                              logger.warning(f"Residue {chain_id}:{res_seq_id} not in prediction map. Setting B-factor to default {self.default_b}. Subsequent warnings suppressed.")
                              ColorByFlexibilitySelect._missing_logged = True
            except Exception as e:
                logger.debug(f"Skipping B-factor modification for residue {chain.id}:{res_id_tuple} due to error: {e}")

        # Set the B-factor (original or modified)
        atom.set_bfactor(float(bfactor_to_set))
        return 1 # Keep the atom


def color_pdb_by_flexibility(
    structure_model: Any, # Bio.PDB Model object
    predictions_df: pd.DataFrame, # Must contain 'chain_id', 'resid', 'predicted_class'
    output_pdb_path: str
):
    """
    Creates a new PDB file where the B-factor column reflects the predicted flexibility class.

    Args:
        structure_model: The Bio.PDB Model object to modify.
        predictions_df: DataFrame with prediction results.
        output_pdb_path: Path to save the colored PDB file.
    """
    if not _biopython_available:
        logger.error("BioPython not available. Cannot create colored PDB.")
        return

    logger.info(f"Generating colored PDB file (using B-factor column): {output_pdb_path}")

    required_cols = ['chain_id', 'resid', 'predicted_class']
    if not all(col in predictions_df.columns for col in required_cols):
         logger.error(f"Predictions DataFrame must contain columns: {required_cols}. Found: {predictions_df.columns.tolist()}")
         return
    try:
        predictions_df['resid'] = predictions_df['resid'].astype(int)
        pred_map = predictions_df.set_index(['chain_id', 'resid'])['predicted_class'].to_dict()
    except Exception as e:
         logger.error(f"Error creating prediction map: {e}", exc_info=True)
         return

    io = PDBIO()
    io.set_structure(structure_model)
    ensure_dir(os.path.dirname(output_pdb_path))

    if hasattr(ColorByFlexibilitySelect, "_missing_logged"):
        del ColorByFlexibilitySelect._missing_logged

    try:
        io.save(output_pdb_path, select=ColorByFlexibilitySelect(pred_map, default_b=20.0))
        logger.info(f"Colored PDB saved successfully to {output_pdb_path}")
    except Exception as e:
        logger.error(f"Failed to save colored PDB file: {e}", exc_info=True)


def generate_pymol_script(
    predictions_df: pd.DataFrame, # Must contain 'chain_id', 'resid', 'predicted_class'
    config: Dict[str, Any],
    output_pml_path: str,
    pdb_filename: Optional[str] = None # Optional: PDB filename to load in script
):
    """
    Generates a PyMOL (.pml) script to color a structure by flexibility class.

    Args:
        predictions_df: DataFrame with prediction results.
        config: Main configuration dictionary (for colors).
        output_pml_path: Path to save the PyMOL script.
        pdb_filename: Optional name/path of the PDB file to be loaded in the script.
                      If None, assumes the structure is already loaded in PyMOL.
    """
    logger.info(f"Generating PyMOL script: {output_pml_path}")
    colors_map = get_visualization_colors(config)
    class_names_map = get_class_names(config)
    num_classes = config.get('binning', {}).get('num_classes', 5)

    if len(colors_map) < num_classes:
        logger.warning(f"Visualization colors defined ({len(colors_map)}) are fewer than number of classes ({num_classes}). Coloring may be incomplete.")

    script_lines = [
        f"# PyMOL Script generated by drWiggle to color by flexibility",
        f"# Timestamp: {pd.Timestamp.now()}",
        "bg_color white",
        "set cartoon_fancy_helices, 1",
        "set cartoon_smooth_loops, 1",
        "show cartoon",
        "color grey80, all"
    ]

    if pdb_filename:
        safe_pdb_filename = pdb_filename.replace("\\", "/")
        script_lines.insert(1, f"load {safe_pdb_filename}")
        obj_name = os.path.splitext(os.path.basename(safe_pdb_filename))[0]
        script_lines.append(f"disable all")
        script_lines.append(f"enable {obj_name}")
        script_lines.append(f"show cartoon, {obj_name}")
        script_lines.append(f"color grey80, {obj_name}")

    pymol_color_names = {}
    for class_idx in range(num_classes):
        color_hex = colors_map.get(class_idx)
        class_name_safe = class_names_map.get(class_idx, f"class_{class_idx}").replace(" ", "_").replace("-","_").replace("/","_")
        color_name_pymol = f"flex_{class_name_safe}"

        if color_hex:
            try:
                color_hex = color_hex.lstrip('#')
                r = int(color_hex[0:2], 16) / 255.0
                g = int(color_hex[2:4], 16) / 255.0
                b = int(color_hex[4:6], 16) / 255.0
                script_lines.append(f"set_color {color_name_pymol}, [{r:.3f}, {g:.3f}, {b:.3f}]")
                pymol_color_names[class_idx] = color_name_pymol
            except Exception:
                logger.warning(f"Invalid hex color format '{color_hex}' for class {class_idx}. Using grey80.")
                pymol_color_names[class_idx] = "grey80"
        else:
             logger.warning(f"Color not defined for class {class_idx}. Using grey80.")
             pymol_color_names[class_idx] = "grey80"

    required_cols = ['chain_id', 'resid', 'predicted_class']
    if not all(col in predictions_df.columns for col in required_cols):
         logger.error(f"Predictions DataFrame for PyMOL script must contain columns: {required_cols}. Found: {predictions_df.columns.tolist()}")
         return

    try: # Add try-except around this block
        predictions_df['resid'] = predictions_df['resid'].astype(int)

        for class_idx in range(num_classes):
            class_residues = predictions_df[predictions_df['predicted_class'] == class_idx]
            color_name = pymol_color_names.get(class_idx, "grey80")

            if not class_residues.empty:
                selection_parts = []
                for chain, group in class_residues.groupby('chain_id'):
                     res_ids_str = "+".join(map(str, sorted(group['resid'].unique())))
                     selection_parts.append(f"(chain {chain} and resi {res_ids_str})")

                if selection_parts:
                    full_selection = " or ".join(selection_parts)
                    script_lines.append(f"color {color_name}, ({full_selection})")
                else:
                     logger.debug(f"No residues found for class {class_idx} to color.")

        script_lines.append("zoom vis")
        script_lines.append(f"print('drWiggle coloring applied using colors: {pymol_color_names}')")

    except Exception as e:
        logger.error(f"Error occurred while generating PyMOL color commands: {e}", exc_info=True)
        # Optionally add a line indicating failure in the script itself
        script_lines.append("print('ERROR: Failed to generate complete coloring commands.')")


    ensure_dir(os.path.dirname(output_pml_path))
    try:
        with open(output_pml_path, 'w') as f:
            f.write("\n".join(script_lines))
        logger.info(f"PyMOL script saved successfully to {output_pml_path}")
    except Exception as e:
        logger.error(f"Failed to write PyMOL script: {e}", exc_info=True)

# --- End File ---