#!/usr/bin/env python3
"""
PDB cleaning module for mdCATH processor pipeline.
Cleans extracted PDB files using pdbUtils.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import tempfile
import shutil

from mdcath.utils.logging_utils import log_info, log_error, log_progress

logger = logging.getLogger(__name__)

def clean_pdb_file(input_pdb: str, output_pdb: str, config: Dict[str, Any]) -> bool:
    """
    Clean a PDB file using pdbUtils.
    
    Args:
        input_pdb: Path to input PDB file
        output_pdb: Path to output PDB file
        config: Cleaning configuration parameters
        
    Returns:
        True if cleaning was successful, False otherwise
    """
    try:
        from pdbUtils import pdbUtils
    except ImportError:
        log_error(f"pdbUtils module not found. Please install it first.")
        return False
    
    try:
        # Extract configuration options
        replace_chain_0 = config.get("replace_chain_0_with_A", True)
        fix_numbering = config.get("fix_atom_numbering", True)
        remove_hydrogens = config.get("remove_hydrogens", False)
        
        # Convert the input PDB file to a DataFrame using pdbUtils
        pdb_df = pdbUtils.pdb2df(input_pdb)
        
        # Get number of atoms before cleaning
        initial_atoms = len(pdb_df)
        
        # Replace chain identifier "0" with "A" if requested
        if replace_chain_0 and len(pdb_df.columns) > 4:
            chain_col = pdb_df.columns[4]
            pdb_df[chain_col] = pdb_df[chain_col].apply(lambda x: 'A' if str(x).strip() == '0' else x)
        
        # Fix atom numbering if requested
        if fix_numbering and "atom_num" in pdb_df.columns:
            pdb_df["atom_num"] = range(1, len(pdb_df) + 1)
        
        # Remove hydrogens if requested
        if remove_hydrogens and "element" in pdb_df.columns:
            pdb_df = pdb_df[pdb_df["element"] != "H"]
        
        # Get number of atoms after cleaning
        final_atoms = len(pdb_df)
        
        # Write the DataFrame back to a PDB file
        pdbUtils.df2pdb(pdb_df, output_pdb)
        
        # Log changes
        if initial_atoms != final_atoms:
            log_info(f"Cleaned PDB: {initial_atoms} atoms -> {final_atoms} atoms")
        else:
            log_info(f"Cleaned PDB without atom count changes")
        
        return True
    except Exception as e:
        log_error(f"Failed to clean PDB {input_pdb}: {e}", exc_info=True)
        return False

def clean_domain_pdbs(domain_dir: str, output_dir: str, 
                     config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean all PDB files for a domain.
    
    Args:
        domain_dir: Directory containing PDB files for the domain
        output_dir: Directory to save cleaned PDB files
        config: Cleaning configuration parameters
        
    Returns:
        Dictionary with cleaning results
    """
    domain_id = os.path.basename(domain_dir)
    log_info(f"Cleaning PDB files for domain {domain_id}", domain_id=domain_id)
    
    results = {
        "domain_id": domain_id,
        "success": False,
        "temperatures_processed": [],
        "pdbs_cleaned": {},
        "total_pdbs": 0
    }
    
    try:
        # Get all temperature subdirectories
        temp_dirs = []
        for item in os.listdir(domain_dir):
            temp_path = os.path.join(domain_dir, item)
            if os.path.isdir(temp_path) and item.isdigit():
                temp_dirs.append((item, temp_path))
        
        if not temp_dirs:
            log_error(f"No temperature directories found for domain {domain_id}", 
                     domain_id=domain_id, error_type="cleaning_error")
            return results
        
        # Create output domain directory
        domain_output_dir = os.path.join(output_dir, domain_id)
        os.makedirs(domain_output_dir, exist_ok=True)
        
        # Process each temperature directory
        for temp, temp_dir in temp_dirs:
            # Create output temperature directory
            temp_output_dir = os.path.join(domain_output_dir, temp)
            os.makedirs(temp_output_dir, exist_ok=True)
            
            # Get all PDB files in this temperature directory
            pdb_files = [f for f in os.listdir(temp_dir) if f.endswith(".pdb")]
            
            if not pdb_files:
                log_info(f"No PDB files found for domain {domain_id}, temperature {temp}", 
                        domain_id=domain_id)
                continue
            
            log_info(f"Cleaning {len(pdb_files)} PDB files for domain {domain_id}, temperature {temp}",
                    domain_id=domain_id)
            
            # Clean each PDB file
            pdbs_cleaned = 0
            
            for pdb_file in pdb_files:
                input_path = os.path.join(temp_dir, pdb_file)
                
                # Generate output filename with _clean suffix
                base_name = os.path.splitext(pdb_file)[0]
                output_path = os.path.join(temp_output_dir, f"{base_name}_clean.pdb")
                
                # Clean the PDB file
                if clean_pdb_file(input_path, output_path, config):
                    pdbs_cleaned += 1
                    log_info(f"Cleaned {pdb_file} -> {output_path}", domain_id=domain_id)
                else:
                    log_error(f"Failed to clean {pdb_file}", domain_id=domain_id, 
                             error_type="cleaning_error")
            
            # Update results
            if pdbs_cleaned > 0:
                results["temperatures_processed"].append(temp)
                results["pdbs_cleaned"][temp] = pdbs_cleaned
                results["total_pdbs"] += pdbs_cleaned
        
        # Mark as successful if at least one temperature was processed
        if results["temperatures_processed"]:
            results["success"] = True
            log_info(f"Successfully cleaned PDB files for domain {domain_id}", domain_id=domain_id)
            log_progress(domain_id, "success")
        else:
            log_error(f"No temperatures were successfully processed for domain {domain_id}", 
                     domain_id=domain_id, error_type="cleaning_error")
            log_progress(domain_id, "error", "cleaning_error")
        
    except Exception as e:
        log_error(f"Error cleaning PDB files for domain {domain_id}: {e}", 
                 domain_id=domain_id, error_type="other_error", exc_info=True)
        log_progress(domain_id, "error", "other_error")
    
    return results

def process_domain_cleaning(args: Tuple) -> Dict[str, Any]:
    """
    Process PDB cleaning for a domain (multiprocessing wrapper).
    
    Args:
        args: Tuple containing (domain_dir, output_dir, config)
        
    Returns:
        Dictionary with cleaning results
    """
    domain_dir, output_dir, config = args
    return clean_domain_pdbs(domain_dir, output_dir, config)

def clean_all_pdbs(input_dir: str, output_dir: str, 
                  config: Dict[str, Any], 
                  num_cores: int = 1) -> Dict[str, Any]:
    """
    Clean PDB files for multiple domains using multiprocessing.
    
    Args:
        input_dir: Base directory containing domain PDB files
        output_dir: Directory to save cleaned PDB files
        config: Cleaning configuration parameters
        num_cores: Number of processor cores to use
        
    Returns:
        Dictionary with cleaning statistics
    """
    import multiprocessing as mp
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all domain directories
    domain_dirs = []
    for item in os.listdir(input_dir):
        domain_path = os.path.join(input_dir, item)
        if os.path.isdir(domain_path):
            domain_dirs.append(domain_path)
    
    if not domain_dirs:
        log_error(f"No domain directories found in {input_dir}")
        return {
            "total_domains": 0,
            "successful_domains": 0,
            "failed_domains": 0,
            "total_pdbs": 0
        }
    
    log_info(f"Cleaning PDB files for {len(domain_dirs)} domains using {num_cores} cores")
    
    # Prepare arguments for each domain
    args_list = [(domain_dir, output_dir, config) for domain_dir in domain_dirs]
    
    # Initialize statistics
    stats = {
        "total_domains": len(domain_dirs),
        "successful_domains": 0,
        "failed_domains": 0,
        "total_pdbs": 0
    }
    
    # Process domains using multiprocessing
    with mp.Pool(processes=num_cores) as pool:
        results = list(pool.map(process_domain_cleaning, args_list))
    
    # Compute statistics
    for result in results:
        if result["success"]:
            stats["successful_domains"] += 1
            stats["total_pdbs"] += result["total_pdbs"]
        else:
            stats["failed_domains"] += 1
    
    log_info(f"PDB cleaning completed: {stats['successful_domains']} successful domains, "
             f"{stats['failed_domains']} failed domains, {stats['total_pdbs']} total PDSs cleaned")
    
    return stats