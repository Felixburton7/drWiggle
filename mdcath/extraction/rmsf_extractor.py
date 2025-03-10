#!/usr/bin/env python3
"""
RMSF extraction module for mdCATH processor pipeline.
Extracts per-residue RMSF values from trajectory data.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any

from mdcath.extraction.h5_reader import MdCathReader
from mdcath.utils.logging_utils import log_info, log_error, log_progress

logger = logging.getLogger(__name__)

def extract_domain_rmsf(reader: MdCathReader, domain_id: str, 
                       temperatures: List[int], output_dir: str) -> Dict[str, Any]:
    """
    Extract RMSF data for a domain across all temperatures and replicas.
    
    Args:
        reader: MdCATH reader instance
        domain_id: Domain identifier
        temperatures: List of temperatures to process
        output_dir: Directory to save output CSV files
        
    Returns:
        Dictionary with extraction results and statistics
    """
    log_info(f"Extracting RMSF data for domain {domain_id}", domain_id=domain_id)
    
    results = {
        "domain_id": domain_id,
        "success": False,
        "temperatures_processed": [],
        "replicas_processed": {},
        "residue_count": 0,
        "average_calculated": False
    }
    
    # Open domain H5 file
    if not reader.open_domain(domain_id):
        log_error(f"Failed to open H5 file for domain {domain_id}", 
                 domain_id=domain_id, error_type="missing_h5")
        return results
    
    try:
        # Get domain metadata
        metadata = reader.get_domain_metadata()
        if not metadata:
            log_error(f"Failed to get metadata for domain {domain_id}", 
                     domain_id=domain_id, error_type="extraction_error")
            return results
        
        log_info(f"Domain {domain_id} has {metadata['num_residues']} residues", domain_id=domain_id)
        results["residue_count"] = metadata["num_residues"]
        
        # Create temperature-specific output directories
        temp_dirs = {}
        for temp in temperatures:
            temp_dir = os.path.join(output_dir, str(temp))
            os.makedirs(temp_dir, exist_ok=True)
            temp_dirs[temp] = temp_dir
        
        # Also create average directory
        avg_dir = os.path.join(output_dir, "average")
        os.makedirs(avg_dir, exist_ok=True)
        
        # Extract RMSF for each available temperature
        all_temps_data = {}
        
        for temp in temperatures:
            if temp not in metadata["temperatures"]:
                log_info(f"Temperature {temp}K not available for domain {domain_id}", domain_id=domain_id)
                continue
            
            log_info(f"Processing temperature {temp}K for domain {domain_id}", domain_id=domain_id)
            
            # Get all replicas for this temperature
            replica_counts = metadata["replica_counts"].get(temp, {})
            replicas_processed = []
            temp_data = []
            
            for replica in range(5):  # Typically 5 replicas
                if replica not in replica_counts or replica_counts[replica] == 0:
                    continue
                
                log_info(f"Processing replica {replica} for domain {domain_id}, temperature {temp}K", 
                        domain_id=domain_id)
                
                # Extract RMSF data
                rmsf_data = reader.get_rmsf_data(temp, replica)
                if not rmsf_data or "rmsf_values" not in rmsf_data:
                    log_error(f"Failed to extract RMSF data for domain {domain_id}, temperature {temp}K, "
                             f"replica {replica}", domain_id=domain_id, error_type="rmsf_error")
                    continue
                
                # Create DataFrame for this replica
                df = pd.DataFrame({
                    "protein_id": domain_id,
                    "resid": rmsf_data["resids"],
                    "resname": rmsf_data["resnames"],
                    f"rmsf_{temp}_r{replica}": rmsf_data["rmsf_values"]
                })
                
                temp_data.append(df)
                replicas_processed.append(replica)
            
            # If no replicas were processed for this temperature, skip
            if not temp_data:
                log_info(f"No valid replicas for domain {domain_id}, temperature {temp}K", domain_id=domain_id)
                continue
            
            # Merge all replicas for this temperature
            if len(temp_data) == 1:
                temp_df = temp_data[0]
            else:
                # Merge on protein_id, resid, and resname
                temp_df = temp_data[0]
                for df in temp_data[1:]:
                    temp_df = pd.merge(temp_df, df, on=["protein_id", "resid", "resname"])
            
            # Calculate average RMSF across replicas for this temperature
            replica_cols = [col for col in temp_df.columns if col.startswith(f"rmsf_{temp}_r")]
            if replica_cols:
                temp_df[f"rmsf_{temp}"] = temp_df[replica_cols].mean(axis=1)
                
                # Save temperature-specific CSV
                temp_csv_path = os.path.join(temp_dirs[temp], f"{domain_id}_temperature_{temp}_average_rmsf.csv")
                temp_df_output = temp_df[["protein_id", "resid", "resname", f"rmsf_{temp}"]]
                temp_df_output.to_csv(temp_csv_path, index=False)
                
                log_info(f"Saved temperature {temp}K RMSF data to {temp_csv_path}", domain_id=domain_id)
                
                # Keep this data for average calculation
                all_temps_data[temp] = temp_df
                
                # Update results
                results["temperatures_processed"].append(temp)
                results["replicas_processed"][temp] = replicas_processed
        
        # Calculate average RMSF across all temperatures
        if all_temps_data:
            # Start with the first temperature's data
            first_temp = list(all_temps_data.keys())[0]
            avg_df = all_temps_data[first_temp][["protein_id", "resid", "resname"]].copy()
            
            # Collect RMSF columns from all temperatures
            rmsf_cols = []
            for temp, df in all_temps_data.items():
                temp_col = f"rmsf_{temp}"
                avg_df[temp_col] = df[temp_col]
                rmsf_cols.append(temp_col)
            
            # Calculate average across temperatures
            if rmsf_cols:
                avg_df["average_rmsf"] = avg_df[rmsf_cols].mean(axis=1)
                
                # Save average RMSF CSV
                avg_csv_path = os.path.join(avg_dir, f"{domain_id}_total_average_rmsf.csv")
                avg_df_output = avg_df[["protein_id", "resid", "resname", "average_rmsf"]]
                avg_df_output.to_csv(avg_csv_path, index=False)
                
                log_info(f"Saved average RMSF data to {avg_csv_path}", domain_id=domain_id)
                results["average_calculated"] = True
        
        # Mark as successful if at least one temperature was processed
        if results["temperatures_processed"]:
            results["success"] = True
            log_info(f"Successfully processed RMSF data for domain {domain_id}", domain_id=domain_id)
            log_progress(domain_id, "success")
        else:
            log_error(f"No temperatures were successfully processed for domain {domain_id}", 
                     domain_id=domain_id, error_type="rmsf_error")
            log_progress(domain_id, "error", "rmsf_error")
        
    except Exception as e:
        log_error(f"Error processing RMSF data for domain {domain_id}: {e}", 
                 domain_id=domain_id, error_type="other_error", exc_info=True)
        log_progress(domain_id, "error", "other_error")
    finally:
        reader.close()
    
    return results

def process_domain_rmsf(args: Tuple) -> Dict[str, Any]:
    """
    Process RMSF data for a domain (multiprocessing wrapper).
    
    Args:
        args: Tuple containing (mdcath_dir, domain_id, temperatures, output_dir)
        
    Returns:
        Dictionary with extraction results
    """
    mdcath_dir, domain_id, temperatures, output_dir = args
    reader = MdCathReader(mdcath_dir)
    return extract_domain_rmsf(reader, domain_id, temperatures, output_dir)

def extract_all_rmsf(mdcath_dir: str, domain_ids: List[str], 
                    temperatures: List[int], output_dir: str,
                    num_cores: int = 1) -> Dict[str, Any]:
    """
    Extract RMSF data for multiple domains using multiprocessing.
    
    Args:
        mdcath_dir: Base directory containing mdCATH HDF5 files
        domain_ids: List of domain identifiers to process
        temperatures: List of temperatures to process
        output_dir: Directory to save output CSV files
        num_cores: Number of processor cores to use
        
    Returns:
        Dictionary with extraction statistics
    """
    import multiprocessing as mp
    from functools import partial
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    log_info(f"Extracting RMSF data for {len(domain_ids)} domains using {num_cores} cores")
    
    # Prepare arguments for each domain
    args_list = [(mdcath_dir, domain_id, temperatures, output_dir) for domain_id in domain_ids]
    
    # Initialize statistics
    stats = {
        "total_domains": len(domain_ids),
        "successful_domains": 0,
        "failed_domains": 0,
        "temperatures_processed": {temp: 0 for temp in temperatures},
        "average_calculated": 0
    }
    
    # Process domains using multiprocessing
    with mp.Pool(processes=num_cores) as pool:
        results = list(pool.map(process_domain_rmsf, args_list))
    
    # Compute statistics
    for result in results:
        if result["success"]:
            stats["successful_domains"] += 1
            
            # Count temperatures processed
            for temp in result["temperatures_processed"]:
                if temp in stats["temperatures_processed"]:
                    stats["temperatures_processed"][temp] += 1
            
            # Count average calculations
            if result["average_calculated"]:
                stats["average_calculated"] += 1
        else:
            stats["failed_domains"] += 1
    
    log_info(f"RMSF extraction completed: {stats['successful_domains']} successful, "
             f"{stats['failed_domains']} failed")
    
    return stats