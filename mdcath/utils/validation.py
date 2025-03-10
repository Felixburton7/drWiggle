#!/usr/bin/env python3
"""
Validation utilities for the mdCATH processor pipeline.
Provides functions to verify data integrity and consistency.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

def validate_pdb_structure(pdb_content: str) -> Dict[str, Any]:
    """
    Validate PDB file content for basic structural integrity.
    
    Args:
        pdb_content: PDB file content as string
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "atom_count": 0,
        "residue_count": 0,
        "chains": set(),
    }
    
    if not pdb_content:
        results["errors"].append("Empty PDB content")
        return results
    
    lines = pdb_content.strip().split('\n')
    atom_lines = [line for line in lines if line.startswith("ATOM")]
    
    if not atom_lines:
        results["errors"].append("No ATOM records found")
        return results
    
    results["atom_count"] = len(atom_lines)
    
    # Check residue information
    resids = set()
    chains = set()
    
    for line in atom_lines:
        if len(line) < 27:  # Minimum length for residue info
            results["warnings"].append(f"Malformed ATOM record: {line}")
            continue
        
        try:
            # Extract chain and residue ID
            chain_id = line[21:22].strip()
            resid = int(line[22:26].strip())
            
            chains.add(chain_id)
            resids.add((chain_id, resid))
        except ValueError:
            results["warnings"].append(f"Invalid residue ID in line: {line}")
    
    results["residue_count"] = len(resids)
    results["chains"] = chains
    
    # Check for warnings about chain IDs
    if '0' in chains:
        results["warnings"].append("Chain ID '0' found (should be replaced with 'A')")
    
    # Structure is valid if there are atom records and no serious errors
    results["valid"] = results["atom_count"] > 0 and not results["errors"]
    
    return results

def validate_rmsf_data(rmsf_df: pd.DataFrame, domain_id: str, 
                       temperature: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate RMSF data for consistency and completeness.
    
    Args:
        rmsf_df: DataFrame with RMSF data
        domain_id: Domain identifier
        temperature: Temperature value (optional)
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "residue_count": 0,
        "statistics": {}
    }
    
    if rmsf_df.empty:
        results["errors"].append("Empty RMSF data")
        return results
    
    # Check required columns
    required_columns = ["protein_id", "resid", "resname"]
    if temperature:
        rmsf_col = f"rmsf_{temperature}"
        required_columns.append(rmsf_col)
    else:
        rmsf_col = "average_rmsf"
        required_columns.append(rmsf_col)
    
    missing_columns = [col for col in required_columns if col not in rmsf_df.columns]
    if missing_columns:
        results["errors"].append(f"Missing required columns: {', '.join(missing_columns)}")
        return results
    
    # Check domain ID
    if "protein_id" in rmsf_df.columns:
        domain_ids = rmsf_df["protein_id"].unique()
        if domain_id not in domain_ids:
            results["errors"].append(f"Domain ID {domain_id} not found in protein_id column")
        elif len(domain_ids) > 1:
            results["warnings"].append(f"Multiple domain IDs found: {', '.join(domain_ids)}")
    
    # Check for missing values
    na_counts = rmsf_df[required_columns].isna().sum()
    for col, count in na_counts.items():
        if count > 0:
            results["warnings"].append(f"{count} missing values in column {col}")
    
    # Check for duplicate residues
    if "resid" in rmsf_df.columns:
        duplicate_resids = rmsf_df["resid"].duplicated()
        if duplicate_resids.any():
            dup_count = duplicate_resids.sum()
            results["warnings"].append(f"{dup_count} duplicate residue IDs found")
    
    # Check RMSF values
    if rmsf_col in rmsf_df.columns:
        rmsf_values = rmsf_df[rmsf_col]
        
        # Check for negative values
        neg_values = (rmsf_values < 0).sum()
        if neg_values > 0:
            results["errors"].append(f"{neg_values} negative RMSF values found")
        
        # Check for unreasonably large values (> 5 nm)
        large_values = (rmsf_values > 5).sum()
        if large_values > 0:
            results["warnings"].append(f"{large_values} unusually large RMSF values (>5 nm) found")
        
        # Calculate basic statistics
        results["statistics"] = {
            "mean": rmsf_values.mean(),
            "median": rmsf_values.median(),
            "std": rmsf_values.std(),
            "min": rmsf_values.min(),
            "max": rmsf_values.max()
        }
    
    results["residue_count"] = len(rmsf_df)
    
    # Data is valid if there are residues and no serious errors
    results["valid"] = results["residue_count"] > 0 and not results["errors"]
    
    return results

def validate_trajectory_data(trajectory_data: Dict[str, np.ndarray], 
                           expected_atoms: int) -> Dict[str, Any]:
    """
    Validate trajectory data for consistency and completeness.
    
    Args:
        trajectory_data: Dictionary with trajectory data arrays
        expected_atoms: Expected number of atoms
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "frame_count": 0,
        "statistics": {}
    }
    
    if not trajectory_data:
        results["errors"].append("Empty trajectory data")
        return results
    
    # Check coordinates
    if "coords" not in trajectory_data:
        results["errors"].append("Missing coordinates data")
        return results
    
    coords = trajectory_data["coords"]
    
    # Check dimensions
    if len(coords.shape) != 3:
        results["errors"].append(f"Coordinates have wrong shape: {coords.shape}, expected 3D array")
        return results
    
    frames, atoms, dims = coords.shape
    
    if atoms != expected_atoms:
        results["errors"].append(f"Coordinates have {atoms} atoms, expected {expected_atoms}")
    
    if dims != 3:
        results["errors"].append(f"Coordinates have {dims} dimensions, expected 3")
    
    results["frame_count"] = frames
    
    # Check forces if available
    if "forces" in trajectory_data:
        forces = trajectory_data["forces"]
        
        if forces.shape != coords.shape:
            results["warnings"].append(f"Forces have shape {forces.shape}, different from coordinates {coords.shape}")
    
    # Check DSSP data if available
    if "dssp" in trajectory_data:
        dssp = trajectory_data["dssp"]
        
        if len(dssp.shape) != 2:
            results["warnings"].append(f"DSSP data has wrong shape: {dssp.shape}, expected 2D array")
        elif dssp.shape[0] != frames:
            results["warnings"].append(f"DSSP data has {dssp.shape[0]} frames, different from coordinates {frames}")
    
    # Check RMSF data if available
    if "rmsf" in trajectory_data:
        rmsf = trajectory_data["rmsf"]
        
        if len(rmsf.shape) != 1:
            results["warnings"].append(f"RMSF data has wrong shape: {rmsf.shape}, expected 1D array")
    
    # Data is valid if there are frames and no serious errors
    results["valid"] = frames > 0 and not results["errors"]
    
    return results

def check_file_integrity(file_path: str) -> Dict[str, Any]:
    """
    Check if a file exists and is readable.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "size_bytes": 0
    }
    
    if not os.path.exists(file_path):
        results["errors"].append(f"File not found: {file_path}")
        return results
    
    if not os.path.isfile(file_path):
        results["errors"].append(f"Not a file: {file_path}")
        return results
    
    try:
        size = os.path.getsize(file_path)
        results["size_bytes"] = size
        
        if size == 0:
            results["warnings"].append("File is empty")
    except Exception as e:
        results["errors"].append(f"Error getting file size: {e}")
        return results
    
    try:
        # Try to open the file
        with open(file_path, 'rb') as f:
            # Read a small chunk to check if readable
            f.read(1024)
    except Exception as e:
        results["errors"].append(f"Error reading file: {e}")
        return results
    
    # File is valid if it exists, is a file, and is readable
    results["valid"] = True
    
    return results