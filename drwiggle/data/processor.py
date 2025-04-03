import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder # For simple categorical encoding

from drwiggle.config import get_feature_config, get_enabled_features, get_window_config, get_split_config, get_system_config

logger = logging.getLogger(__name__)

# --- Default Mappings (can be adjusted or loaded from config/files) ---
# Standard 20 amino acids + UNK (Unknown) + GAP (-) potentially
AA_ORDER = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
    'TYR', 'VAL', 'UNK', 'GAP' # Add others if needed (e.g., SEC)
]
# Create mapping dictionary {AA_NAME: index}
AA_MAP = {name: i for i, name in enumerate(AA_ORDER)}
AA_UNK_CODE = AA_MAP['UNK'] # Default code for unknown AAs

# DSSP Secondary Structure categories (simplified mapping)
# H=AlphaHelix, G=310Helix, I=PiHelix -> Helix (0)
# E=Strand, B=Bridge -> Sheet (1)
# T=Turn, S=Bend, C=Coil/Loop, -=Unknown/Gap -> Coil/Other (2)
SS_MAP = {'H': 0, 'G': 0, 'I': 0, # Helix-like
          'E': 1, 'B': 1,         # Sheet-like
          'T': 2, 'S': 2, 'C': 2, '-': 2, # Coil/Other/Unknown
          '?': 2 # Handle DSSP '?' for unknown
          }
SS_UNK_CODE = SS_MAP['-']

# Core/Exterior mapping (example)
LOC_MAP = {'CORE': 0, 'SURFACE': 1, 'EXTERIOR': 1, 'UNK': 0} # Map Surface/Exterior to 1, Core/Unknown to 0
LOC_UNK_CODE = LOC_MAP['UNK']

# --- Feature Processing Functions ---

def _get_column_name(df_columns: List[str], base_name: str) -> Optional[str]:
    """Finds column name case-insensitively."""
    base_lower = base_name.lower()
    for col in df_columns:
        if col.lower() == base_lower:
            return col
    return None

def clean_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Basic data cleaning: handles NaNs in expected numeric/object columns.
    Uses median for numeric and specific placeholders ('UNK', 'C', 'CORE') for categoricals.
    """
    logger.debug(f"Starting data cleaning. Initial shape: {df.shape}. NaN counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    df_cleaned = df.copy() # Work on a copy

    # Numeric columns: fill with median
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df_cleaned[col].isnull().any():
            # Exclude target columns from NaN filling if present (should be handled later)
            target_col_template = config.get("dataset",{}).get("target", "rmsf_{temperature}")
            # This check is basic, might need refinement if multiple targets exist
            if col.lower().startswith(target_col_template.split('_')[0].lower()):
                 logger.debug(f"Skipping NaN fill for potential target column: {col}")
                 continue

            median_val = df_cleaned[col].median()
            # Check if median is NaN (can happen if all values are NaN)
            if pd.isna(median_val):
                 median_val = 0 # Fallback to 0 if median cannot be calculated
                 logger.warning(f"Median for numeric column '{col}' is NaN. Filling with 0.")
            df_cleaned[col].fillna(median_val, inplace=True)
            logger.debug(f"Filled NaNs in numeric column '{col}' with median ({median_val:.3f}).")

    # Object/Categorical columns: fill with specific placeholders
    object_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
    df_columns_list = df_cleaned.columns.tolist() # For case-insensitive check

    # --- Specific Columns ---
    # Residue Name ('resname', 'residue', etc.)
    resname_col = _get_column_name(df_columns_list, 'resname') or _get_column_name(df_columns_list, 'residue_name')
    if resname_col and df_cleaned[resname_col].isnull().any():
        df_cleaned[resname_col].fillna('UNK', inplace=True)
        logger.debug(f"Filled NaNs in '{resname_col}' with 'UNK'.")

    # Secondary Structure ('dssp', 'ss', 'secondary_structure')
    ss_col = _get_column_name(df_columns_list, 'dssp') or _get_column_name(df_columns_list, 'secondary_structure')
    if ss_col and df_cleaned[ss_col].isnull().any():
        df_cleaned[ss_col].fillna('-', inplace=True) # DSSP standard for unknown/gap
        logger.debug(f"Filled NaNs in '{ss_col}' with '-'.")

    # Core/Exterior ('core_exterior', 'location')
    loc_col = _get_column_name(df_columns_list, 'core_exterior') or _get_column_name(df_columns_list, 'location')
    if loc_col and df_cleaned[loc_col].isnull().any():
        df_cleaned[loc_col].fillna('UNK', inplace=True) # Use UNK placeholder
        logger.debug(f"Filled NaNs in '{loc_col}' with 'UNK'.")

    # --- Generic Object Columns ---
    # For other object columns, fill with 'UNK' or mode if appropriate
    for col in object_cols:
         # Skip already handled specific columns
         if col in [resname_col, ss_col, loc_col]: continue
         if df_cleaned[col].isnull().any():
             fill_val = 'UNK' # General unknown placeholder
             df_cleaned[col].fillna(fill_val, inplace=True)
             logger.debug(f"Filled NaNs in generic object column '{col}' with '{fill_val}'.")

    nan_counts_after = df_cleaned.isnull().sum().sum()
    logger.debug(f"Data cleaning finished. Shape: {df_cleaned.shape}. Total NaNs remaining: {nan_counts_after}")
    if nan_counts_after > 0:
         logger.warning(f"NaNs still present after cleaning. Check target columns or unexpected types:\n{df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0]}")

    return df_cleaned

def encode_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Encode categorical features based on 'use_features' config."""
    logger.debug("Encoding categorical features...")
    df_encoded = df.copy()
    feature_cfg = get_feature_config(config)
    use_features = feature_cfg.get("use_features", {})
    df_columns = df_encoded.columns.tolist()

    # --- Residue Name Encoding ---
    resname_col = _get_column_name(df_columns, 'resname') or _get_column_name(df_columns, 'residue_name')
    if use_features.get("resname_encoded") and resname_col:
        logger.debug(f"Encoding '{resname_col}' using predefined AA map.")
        df_encoded['resname_encoded'] = df_encoded[resname_col].str.upper().map(AA_MAP).fillna(AA_UNK_CODE).astype(int)
    elif use_features.get("resname_encoded"):
        logger.warning("Feature 'resname_encoded' enabled but 'resname' column not found.")

    # --- Secondary Structure Encoding ---
    ss_col = _get_column_name(df_columns, 'dssp') or _get_column_name(df_columns, 'secondary_structure')
    if use_features.get("secondary_structure_encoded") and ss_col:
        logger.debug(f"Encoding '{ss_col}' using predefined SS map.")
        df_encoded['secondary_structure_encoded'] = df_encoded[ss_col].str.upper().map(SS_MAP).fillna(SS_UNK_CODE).astype(int)
    elif use_features.get("secondary_structure_encoded"):
        logger.warning("Feature 'secondary_structure_encoded' enabled but 'dssp' or 'secondary_structure' column not found.")

    # --- Core/Exterior Encoding ---
    loc_col = _get_column_name(df_columns, 'core_exterior') or _get_column_name(df_columns, 'location')
    if use_features.get("core_exterior_encoded") and loc_col:
        logger.debug(f"Encoding '{loc_col}' using predefined Location map (Core=0, Surface=1).")
        df_encoded['core_exterior_encoded'] = df_encoded[loc_col].str.upper().map(LOC_MAP).fillna(LOC_UNK_CODE).astype(int)
    elif use_features.get("core_exterior_encoded"):
        logger.warning("Feature 'core_exterior_encoded' enabled but 'core_exterior' or 'location' column not found.")

    # Add other encodings here if needed (e.g., OneHot for specific features)

    logger.debug(f"Feature encoding finished. Columns added/modified: {[c for c in df_encoded.columns if c not in df.columns or not df_encoded[c].equals(df[c])]}")
    return df_encoded

def normalize_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Normalize features like angles and residue index."""
    logger.debug("Normalizing features...")
    df_norm = df.copy()
    feature_cfg = get_feature_config(config)
    use_features = feature_cfg.get("use_features", {})
    df_columns = df_norm.columns.tolist()
    domain_id_col = _get_column_name(df_columns, 'domain_id') or _get_column_name(df_columns, 'protein_id') # Need a domain/protein identifier
    resid_col = _get_column_name(df_columns, 'resid') or _get_column_name(df_columns, 'res_id') or _get_column_name(df_columns, 'residue_number')


    # --- Angle Normalization (using sin/cos for cyclical nature) ---
    phi_col = _get_column_name(df_columns, 'phi')
    psi_col = _get_column_name(df_columns, 'psi')

    if use_features.get("phi_norm") and phi_col:
        logger.debug(f"Normalizing '{phi_col}' using sin/cos.")
        phi_rad = np.radians(df_norm[phi_col].fillna(0)) # Fill NaNs with 0 degrees before conversion
        df_norm['phi_sin'] = np.sin(phi_rad)
        df_norm['phi_cos'] = np.cos(phi_rad)
    elif use_features.get("phi_norm"):
        logger.warning("Feature 'phi_norm' enabled but 'phi' column not found.")

    if use_features.get("psi_norm") and psi_col:
        logger.debug(f"Normalizing '{psi_col}' using sin/cos.")
        psi_rad = np.radians(df_norm[psi_col].fillna(0))
        df_norm['psi_sin'] = np.sin(psi_rad)
        df_norm['psi_cos'] = np.cos(psi_rad)
    elif use_features.get("psi_norm"):
        logger.warning("Feature 'psi_norm' enabled but 'psi' column not found.")

    # --- Protein Size ---
    # Calculate if needed by other features or if explicitly enabled
    protein_size_col = _get_column_name(df_columns, 'protein_size')
    needs_protein_size = use_features.get("protein_size") or use_features.get("normalized_resid")
    if needs_protein_size and not protein_size_col:
        if not domain_id_col:
             logger.warning("Cannot calculate 'protein_size': 'domain_id' column not found.")
        elif not resid_col:
             logger.warning("Cannot calculate 'protein_size': 'resid' column not found.")
        else:
             logger.debug(f"Calculating 'protein_size' based on '{domain_id_col}' and '{resid_col}'.")
             # Use transform to broadcast the count per group back to the original DataFrame
             df_norm['protein_size'] = df_norm.groupby(domain_id_col)[resid_col].transform('count')
             protein_size_col = 'protein_size' # Update column name
    elif use_features.get("protein_size") and not protein_size_col:
         logger.warning("Feature 'protein_size' enabled but column not found and could not be calculated.")


    # --- Normalized Residue Index ---
    if use_features.get("normalized_resid") and resid_col:
        if not domain_id_col:
             logger.warning("Cannot calculate 'normalized_resid': 'domain_id' column not found.")
        elif not protein_size_col or protein_size_col not in df_norm.columns: # Check if calculation succeeded
             logger.warning("Cannot calculate 'normalized_resid': 'protein_size' column not found or calculated.")
        else:
             logger.debug(f"Calculating 'normalized_resid' using '{resid_col}', '{domain_id_col}', and '{protein_size_col}'.")
             # Normalize within each domain: (resid - min_resid) / (max_resid - min_resid)
             # Assumes resid is numeric and sequential within domain for this normalization.
             # If resids are not sequential, this might not be the desired normalization.
             min_res = df_norm.groupby(domain_id_col)[resid_col].transform('min')
             max_res = df_norm.groupby(domain_id_col)[resid_col].transform('max')
             # Avoid division by zero for single-residue domains or constant resid
             denominator = (max_res - min_res).clip(lower=1e-6) # Add small epsilon
             df_norm['normalized_resid'] = (df_norm[resid_col] - min_res) / denominator
             # Clip to [0, 1] just in case, and fill NaNs (e.g., from single residue domains) with 0
             df_norm['normalized_resid'] = df_norm['normalized_resid'].fillna(0).clip(0, 1)
    elif use_features.get("normalized_resid"):
         logger.warning("Feature 'normalized_resid' enabled but 'resid' column not found.")


    # --- B-Factor Normalization (Example: Z-score per protein) ---
    b_factor_col = _get_column_name(df_columns, 'b_factor') or _get_column_name(df_columns, 'bfactor')
    if use_features.get("b_factor") and b_factor_col: # Check if enabled AND column exists
         if domain_id_col:
             logger.debug(f"Z-score normalizing '{b_factor_col}' per domain ('{domain_id_col}').")
             mean_b = df_norm.groupby(domain_id_col)[b_factor_col].transform('mean')
             std_b = df_norm.groupby(domain_id_col)[b_factor_col].transform('std')
             # Avoid division by zero for domains with constant B-factor or single residue
             std_b = std_b.fillna(1e-6).clip(lower=1e-6)
             df_norm['b_factor_norm'] = (df_norm[b_factor_col] - mean_b) / std_b
             # Fill potential NaNs resulting from calculation with 0 (mean)
             df_norm['b_factor_norm'].fillna(0, inplace=True)
         else:
             logger.warning(f"Cannot normalize '{b_factor_col}' per domain: '{domain_id_col}' column not found. Skipping normalization.")
             # Copy original B-factor if normalization fails but feature is requested? Or drop?
             # For now, do nothing, model prep will fail if 'b_factor' is expected but only 'b_factor_norm' might exist
    # Note: The feature name in use_features is 'b_factor'. If normalization is done,
    # the pipeline/model prep needs to know to use 'b_factor_norm' instead.
    # This logic needs refinement - perhaps rename the column to 'b_factor' after normalization,
    # or adjust get_enabled_features based on processing steps.
    # --> Let's rename 'b_factor_norm' back to 'b_factor' to simplify downstream use.
    if 'b_factor_norm' in df_norm.columns:
         df_norm['b_factor'] = df_norm['b_factor_norm']
         df_norm.drop(columns=['b_factor_norm'], inplace=True)
         logger.debug("Renamed 'b_factor_norm' to 'b_factor' for consistency.")


    logger.debug(f"Feature normalization finished. Columns added/modified: {[c for c in df_norm.columns if c not in df.columns or not df_norm[c].equals(df[c])]}")
    return df_norm


def create_window_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Create window-based features for sequence context."""
    window_cfg = get_window_config(config)
    if not window_cfg.get("enabled", False):
        logger.debug("Window features disabled.")
        return df

    window_size = window_cfg.get("size", 3)
    if not isinstance(window_size, int) or window_size <= 0:
        logger.warning(f"Invalid window size ({window_size}). Must be positive integer. Skipping window features.")
        return df

    # Identify base features to create windows from (must exist in df after processing)
    all_enabled_features = get_enabled_features(config)
    # We need the *encoded* or *normalized* versions if they exist
    potential_window_bases = [
        'resname_encoded', 'secondary_structure_encoded', 'core_exterior_encoded',
        'phi_sin', 'phi_cos', 'psi_sin', 'psi_cos', # Use sin/cos components
        'b_factor', # Use normalized B-factor if calculated
        'relative_accessibility' # Example continuous feature
    ]
    # Filter based on features actually present in df AND enabled in config
    features_to_window = [
        f for f in potential_window_bases
        if f in df.columns and f in all_enabled_features
    ]
    # Handle normalized features replacing originals
    if 'phi_sin' in features_to_window and 'phi_norm' in all_enabled_features: features_to_window.append('phi_norm') # Keep track
    if 'psi_sin' in features_to_window and 'psi_norm' in all_enabled_features: features_to_window.append('psi_norm')

    if not features_to_window:
         logger.warning("Window features enabled, but no suitable base features found in DataFrame. Skipping.")
         return df

    logger.info(f"Creating window features (size {window_size}) for: {features_to_window}")

    df_out = df.copy()
    domain_id_col = _get_column_name(df.columns, 'domain_id') or _get_column_name(df.columns, 'protein_id')
    resid_col = _get_column_name(df.columns, 'resid') or _get_column_name(df.columns, 'res_id')
    if not domain_id_col or not resid_col:
        logger.error("Cannot create window features: 'domain_id' or 'resid' column missing.")
        return df # Return original df

    # Ensure DataFrame is sorted by domain and residue index for correct shifting
    logger.debug(f"Sorting DataFrame by '{domain_id_col}' and '{resid_col}' for windowing.")
    df_out = df_out.sort_values(by=[domain_id_col, resid_col])

    grouped = df_out.groupby(domain_id_col, sort=False) # Use sort=False for potential speedup if already sorted

    window_feature_cols = [] # Keep track of new columns added
    padding_value = 0.0 # Value to use for padding at sequence ends

    for base_feature in features_to_window:
        # Skip the original 'phi_norm'/'psi_norm' if sin/cos used, avoid duplication in window
        if base_feature in ['phi_norm', 'psi_norm'] and f'{base_feature.split("_")[0]}_sin' in features_to_window:
             continue

        for k in range(1, window_size + 1): # Shifts from 1 to window_size
            # Shift backward (preceding residues)
            col_name_prev = f"{base_feature}_prev_{k}"
            df_out[col_name_prev] = grouped[base_feature].shift(k, fill_value=padding_value)
            window_feature_cols.append(col_name_prev)

            # Shift forward (succeeding residues)
            col_name_next = f"{base_feature}_next_{k}"
            df_out[col_name_next] = grouped[base_feature].shift(-k, fill_value=padding_value)
            window_feature_cols.append(col_name_next)

    logger.info(f"Added {len(window_feature_cols)} window feature columns.")

    # Reindex back to the original DataFrame's index if the order might have been disrupted
    # This ensures compatibility if the original index had specific meaning or order elsewhere.
    # However, since we usually split *after* processing, this might not be strictly necessary.
    # For safety, let's reindex if the original df index was not a simple range.
    if not df.index.equals(pd.RangeIndex(start=0, stop=len(df), step=1)):
         logger.debug("Reindexing DataFrame back to original index after windowing.")
         df_out = df_out.reindex(df.index)

    return df_out


def process_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Main feature processing pipeline: clean -> encode -> normalize -> window."""
    logger.info(f"Starting feature processing pipeline. Initial shape: {df.shape}")
    df_processed = clean_data(df, config)
    df_processed = encode_features(df_processed, config)
    df_processed = normalize_features(df_processed, config)
    df_processed = create_window_features(df_processed, config) # Add window features last
    final_cols = df_processed.columns.tolist()
    logger.info(f"Feature processing complete. Final shape: {df_processed.shape}. Final columns: {final_cols}")
    return df_processed

def split_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train, validation, and test sets based on config.
    Handles random splitting and stratified splitting by domain ID.
    """
    split_config = get_split_config(config)
    system_config = get_system_config(config)

    test_size = split_config.get('test_size', 0.2)
    val_size = split_config.get('validation_size', 0.15) # Proportion of original data
    stratify = split_config.get('stratify_by_domain', True)
    random_state = system_config.get('random_state', 42)

    if not (0 < test_size < 1): raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if not (0 <= val_size < 1): raise ValueError(f"validation_size must be between 0 and 1, got {val_size}")
    if test_size + val_size >= 1.0:
         raise ValueError(f"Sum of test_size ({test_size}) and validation_size ({val_size}) must be less than 1.")

    logger.info(f"Splitting data: Test={test_size*100:.1f}%, Val={val_size*100:.1f}%, Stratify by domain={stratify}, Seed={random_state}")

    domain_id_col = _get_column_name(df.columns, 'domain_id') or _get_column_name(df.columns, 'protein_id')

    if stratify:
        if not domain_id_col:
            logger.warning("Cannot stratify by domain: 'domain_id'/'protein_id' column missing. Performing random split instead.")
            stratify = False # Fallback to random split
        else:
            domains = df[domain_id_col].unique()
            n_domains = len(domains)
            logger.info(f"Found {n_domains} unique domains for stratified splitting.")

            if n_domains < 3: # Need at least one domain per potential split
                 logger.warning(f"Too few unique domains ({n_domains}) for stratified 3-way splitting. Performing random split instead.")
                 stratify = False # Fallback
            else:
                 # Split domains first using GroupShuffleSplit
                 # Split into Train+Val vs Test domains
                 gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                 train_val_idx, test_idx = next(gss_test.split(df, groups=df[domain_id_col]))

                 df_train_val = df.iloc[train_val_idx]
                 df_test = df.iloc[test_idx]

                 # Calculate relative validation size within the Train+Val split
                 relative_val_size = val_size / (1.0 - test_size)
                 if relative_val_size >= 1.0: # Should not happen due to initial check, but good safeguard
                      logger.warning(f"Relative validation size ({relative_val_size:.3f}) is >= 1. Adjusting validation split.")
                      # Allocate minimum to validation (e.g., 1 domain or small fraction)
                      # This needs careful handling based on domain counts. Simpler: maybe make val smaller?
                      # For now, proceed but it might merge train/val if val_size is too large.

                 # Split Train+Val into Train vs Val domains
                 gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
                 train_idx_rel, val_idx_rel = next(gss_val.split(df_train_val, groups=df_train_val[domain_id_col]))

                 # Use relative indices to get final train/val DataFrames
                 df_train = df_train_val.iloc[train_idx_rel]
                 df_val = df_train_val.iloc[val_idx_rel]

                 # Verify domain separation
                 train_domains = df_train[domain_id_col].unique()
                 val_domains = df_val[domain_id_col].unique()
                 test_domains = df_test[domain_id_col].unique()
                 if not set(train_domains).isdisjoint(val_domains) or \
                    not set(train_domains).isdisjoint(test_domains) or \
                    not set(val_domains).isdisjoint(test_domains):
                      logger.error("Stratified domain splitting failed! Overlap detected between splits. Check GroupShuffleSplit logic.")
                      # Fallback to random as emergency? Or raise error? Raising for now.
                      raise RuntimeError("Domain overlap detected in stratified split.")
                 logger.debug(f"Stratified domain split: Train={len(train_domains)}, Val={len(val_domains)}, Test={len(test_domains)} domains.")


    if not stratify: # Perform random split if stratification failed or disabled
        logger.debug("Performing random split.")
        # First split into Train+Val and Test
        df_train_val, df_test = train_test_split(
            df, test_size=test_size, random_state=random_state, shuffle=True
        )
        # Then split Train+Val into Train and Val
        # Adjust val_size relative to the remaining data
        relative_val_size = val_size / (1.0 - test_size)
        df_train, df_val = train_test_split(
            df_train_val, test_size=relative_val_size, random_state=random_state, shuffle=True
        )

    logger.info(f"Data split complete: Train={len(df_train)} ({len(df_train)/len(df)*100:.1f}%), "
                f"Val={len(df_val)} ({len(df_val)/len(df)*100:.1f}%), "
                f"Test={len(df_test)} ({len(df_test)/len(df)*100:.1f}%) rows.")

    if df_train.empty or df_test.empty: # Validation can sometimes be empty if val_size is very small
         logger.error("Train or Test split resulted in an empty DataFrame! Check split sizes and data.")
         raise ValueError("Empty Train or Test split created.")
    if df_val.empty and val_size > 0:
         logger.warning("Validation split is empty. Check validation_size and data.")


    # Return copies to avoid SettingWithCopyWarning downstream
    return df_train.copy(), df_val.copy(), df_test.copy()


def prepare_data_for_model(
    df: pd.DataFrame,
    config: Dict[str, Any],
    target_col: Optional[str] = None, # Target column name (e.g., 'flexibility_class')
    features: Optional[List[str]] = None # Optional: Explicit list of features to use
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """
    Selects features and optionally the target column for model input.
    Returns DataFrames/Series.

    Args:
        df: DataFrame with processed features and potentially the target class.
        config: Configuration dictionary.
        target_col: Name of the target column. If None, only X is returned.
        features: Optional list of feature names to use. If None, derived from config and window features.

    Returns:
        Tuple (X_df, y_series | None, final_feature_names).
        y_series is None if target_col is None.
    """
    logger.debug("Preparing DataFrame/Series for model input...")

    if features is None:
        # Determine feature columns automatically from config and generated columns
        base_enabled_features = get_enabled_features(config)
        window_cfg = get_window_config(config)
        final_feature_names = []

        # Add base features (considering normalized versions)
        for feature in base_enabled_features:
             if feature == 'phi_norm' and 'phi_sin' in df.columns:
                 final_feature_names.extend(['phi_sin', 'phi_cos'])
             elif feature == 'psi_norm' and 'psi_sin' in df.columns:
                 final_feature_names.extend(['psi_sin', 'psi_cos'])
             elif feature == 'b_factor' and 'b_factor_norm' in df.columns: # Check if normalized version exists
                 final_feature_names.append('b_factor_norm') # Prefer normalized if exists
             elif feature in df.columns:
                 final_feature_names.append(feature)
             else:
                 logger.warning(f"Enabled base feature '{feature}' not found in DataFrame columns. Skipping.")

        # Add window features if enabled
        if window_cfg.get("enabled", False) and window_cfg.get("size", 0) > 0:
            window_size = window_cfg["size"]
            potential_window_bases = [
                 f for f in [ # List encoded/normalized features that form windows
                      'resname_encoded', 'secondary_structure_encoded', 'core_exterior_encoded',
                      'phi_sin', 'phi_cos', 'psi_sin', 'psi_cos',
                      'b_factor', 'relative_accessibility' ]
                 if f in final_feature_names # Check if the base was actually included
             ]
            for base_feature in potential_window_bases:
                 for k in range(1, window_size + 1):
                     col_prev = f"{base_feature}_prev_{k}"
                     col_next = f"{base_feature}_next_{k}"
                     if col_prev in df.columns: final_feature_names.append(col_prev)
                     if col_next in df.columns: final_feature_names.append(col_next)

        # Remove duplicates just in case
        final_feature_names = sorted(list(set(final_feature_names)))

        if not final_feature_names:
             raise ValueError("No features selected based on config or found in the DataFrame.")
        logger.debug(f"Automatically selected {len(final_feature_names)} features based on config and DataFrame columns.")

    else:
         # Use provided feature list, ensuring they exist
         final_feature_names = [f for f in features if f in df.columns]
         if len(final_feature_names) != len(features):
              missing = set(features) - set(final_feature_names)
              logger.warning(f"Provided feature names not found in DataFrame: {missing}. Using {len(final_feature_names)} available features.")
         if not final_feature_names:
             raise ValueError("None of the explicitly provided feature names were found in the DataFrame.")
         logger.debug(f"Using {len(final_feature_names)} explicitly provided feature names.")

    # Extract features (X)
    try:
        X_df = df[final_feature_names]
        # Check for non-numeric types in features
        numeric_types = ['int64', 'float64', 'int32', 'float32']
        non_numeric_cols = X_df.select_dtypes(exclude=numeric_types).columns
        if not non_numeric_cols.empty:
             logger.warning(f"Selected features contain non-numeric columns: {non_numeric_cols.tolist()}. Model training might fail.")
             # Attempt conversion? Or raise error? For now, just warn.

    except KeyError as e:
        logger.error(f"Missing feature column during data preparation: {e}. Available columns: {df.columns.tolist()}")
        raise ValueError(f"Feature column required for model input not found: {e}")
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise

    # Extract target (y) if requested
    y_series: Optional[pd.Series] = None
    if target_col:
        target_col_actual = _get_column_name(df.columns, target_col)
        if not target_col_actual:
             raise ValueError(f"Target column '{target_col}' not found in DataFrame columns: {df.columns.tolist()}")
        try:
            y_series = df[target_col_actual]
            # Ensure target is integer type for classification
            if not pd.api.types.is_integer_dtype(y_series):
                 logger.warning(f"Target column '{target_col_actual}' is not integer type ({y_series.dtype}). Attempting conversion.")
                 try:
                      # Check for NaNs before conversion
                      if y_series.isnull().any():
                           logger.error(f"Target column '{target_col_actual}' contains NaN values. Cannot convert to integer.")
                           raise ValueError(f"NaNs found in target column '{target_col_actual}'.")
                      y_series = y_series.astype(int)
                 except (ValueError, TypeError) as e:
                      logger.error(f"Could not convert target column '{target_col_actual}' to integer: {e}")
                      raise ValueError(f"Target column '{target_col_actual}' must be integer or convertible to integer.")
        except KeyError:
            logger.error(f"Target column '{target_col_actual}' not found after name resolution check. This shouldn't happen.")
            raise ValueError(f"Target column '{target_col_actual}' unexpectedly not found.")
        except Exception as e:
             logger.error(f"Error extracting target column '{target_col_actual}': {e}")
             raise

    logger.debug(f"Model data prepared: X shape={X_df.shape}, y shape={y_series.shape if y_series is not None else 'N/A'}")
    return X_df, y_series, final_feature_names
