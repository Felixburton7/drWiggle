#!/usr/bin/env python3
"""
Frame extraction module for mdCATH processor pipeline.
Extracts representative trajectory frames and generates PDB files.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import tempfile

from mdcath.extraction.h5_reader import MdCathReader
from mdcath.utils.logging_utils import log_info, log_error, log_progress

logger = logging.getLogger(__name__)

def select_frames(trajectory_data: Dict[str, np.ndarray], 
                 method: str = "regular", num_frames: int = 10) -> List[int]:
    """
    Select representative frames from trajectory.
    
    Args:
        trajectory_data: Dictionary with trajectory data arrays
        method: Method for frame selection (regular, rmsd, gyration, random)
        num_frames: Number of frames to select
        
    Returns:
        List of selected frame indices
    """
    if not trajectory_data or "coords" not in trajectory_data:
        return []
    
    num_available_frames = trajectory_data["coords"].shape[0]
    if num_available_frames <= num_frames:
        # If we have fewer frames than requested, return all
        return list(range(num_available_frames))
    
    if method == "regular":
        # Select frames at regular intervals
        indices = np.linspace(0, num_available_frames - 1, num_frames, dtype=int)
        return indices.tolist()
    
    elif method == "random":
        # Select random frames
        indices = np.sort(np.random.choice(
            num_available_frames, num_frames, replace=False))
        return indices.tolist()
    
    elif method == "rmsd":
        # Select frames based on RMSD from initial structure
        if "rmsd" in trajectory_data:
            rmsd_values = trajectory_data["rmsd"]
            
            # Find frames with evenly spaced RMSD values
            min_rmsd = np.min(rmsd_values)
            max_rmsd = np.max(rmsd_values)
            target_rmsds = np.linspace(min_rmsd, max_rmsd, num_frames)
            
            indices = []
            for target in target_rmsds:
                # Find frame with closest RMSD to target
                idx = np.argmin(np.abs(rmsd_values - target))
                indices.append(idx)
            
            # Remove duplicates and ensure we have enough frames
            indices = sorted(list(set(indices)))
            while len(indices) < num_frames and len(indices) < num_available_frames:
                # Add frames not already selected
                remaining = list(set(range(num_available_frames)) - set(indices))
                if not remaining:
                    break
                # Add the frame with maximum RMSD from already selected frames
                max_idx = None
                max_dist = -1
                for idx in remaining:
                    min_dist = min(abs(idx - i) for i in indices)
                    if min_dist > max_dist:
                        max_dist = min_dist
                        max_idx = idx
                if max_idx is not None:
                    indices.append(max_idx)
            
            return sorted(indices)
        else:
            # Fall back to regular interval sampling if RMSD not available
            logger.warning("RMSD data not available, falling back to regular interval sampling")
            return select_frames(trajectory_data, "regular", num_frames)
    
    elif method == "gyration":
        # Select frames based on variation in gyration radius
        if "gyrationRadius" in trajectory_data:
            gyration_values = trajectory_data["gyrationRadius"]
            
            # Find frames with evenly spaced gyration radius values
            min_gyration = np.min(gyration_values)
            max_gyration = np.max(gyration_values)
            target_gyrations = np.linspace(min_gyration, max_gyration, num_frames)
            
            indices = []
            for target in target_gyrations:
                # Find frame with closest gyration radius to target
                idx = np.argmin(np.abs(gyration_values - target))
                indices.append(idx)
            
            # Remove duplicates and ensure we have enough frames
            indices = sorted(list(set(indices)))
            while len(indices) < num_frames and len(indices) < num_available_frames:
                remaining = list(set(range(num_available_frames)) - set(indices))
                if not remaining:
                    break
                max_idx = None
                max_dist = -1
                for idx in remaining:
                    min_dist = min(abs(idx - i) for i in indices)
                    if min_dist > max_dist:
                        max_dist = min_dist
                        max_idx = idx
                if max_idx is not None:
                    indices.append(max_idx)
            
            return sorted(indices)
        else:
            # Fall back to regular interval sampling if gyration radius not available
            logger.warning("Gyration radius data not available, falling back to regular interval sampling")
            return select_frames(trajectory_data, "regular", num_frames)
    
    else:
        # Default to regular interval sampling for unknown methods
        logger.warning(f"Unknown frame selection method '{method}', falling back to regular interval sampling")
        return select_frames(trajectory_data, "regular", num_frames)

def create_pdb_from_frame(domain_id: str, atom_data: Dict[str, np.ndarray], 
                         coordinates: np.ndarray, frame_idx: int, 
                         temperature: int, replica: int) -> str:
    """
    Create a PDB file content from frame coordinates.
    
    Args:
        domain_id: Domain identifier
        atom_data: Dictionary with atom data arrays
        coordinates: Coordinates for the frame (N×3)
        frame_idx: Frame index for reference
        temperature: Temperature value
        replica: Replica index
        
    Returns:
        PDB file content as string
    """
    # Validate input data
    if not atom_data or "chain" not in atom_data or "element" not in atom_data:
        return ""
    
    if coordinates.shape[0] != len(atom_data["chain"]):
        logger.error(f"Mismatch between coordinates ({coordinates.shape[0]} atoms) "
                     f"and atom data ({len(atom_data['chain'])} atoms)")
        return ""
    
    # Construct PDB content
    pdb_lines = [f"REMARK   Generated from mdCATH domain {domain_id}"]
    pdb_lines.append(f"REMARK   Temperature: {temperature}K, Replica: {replica}, Frame: {frame_idx}")
    
    for i in range(coordinates.shape[0]):
        x, y, z = coordinates[i]
        atom_serial = i + 1
        
        # Get atom properties
        chain_id = atom_data["chain"][i] if "chain" in atom_data else "A"
        element = atom_data["element"][i] if "element" in atom_data else ""
        resid = atom_data["resid"][i] if "resid" in atom_data else 0
        resname = atom_data["resname"][i] if "resname" in atom_data else "UNK"
        
        # Handle empty chain ID
        if not chain_id or chain_id == "0":
            chain_id = "A"
        
        # Atom name - use element with position number if available
        atom_name = element
        atom_name = f"{atom_name:4s}"
        
        # Format PDB ATOM record
        pdb_line = (f"ATOM  {atom_serial:5d} {atom_name} {resname:3s} {chain_id:1s}{resid:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:2s}")
        pdb_lines.append(pdb_line)
    
    # Add TER and END records
    pdb_lines.append("TER")
    pdb_lines.append("END")
    
    return "\n".join(pdb_lines)

def extract_domain_frames(reader: MdCathReader, domain_id: str, 
                         temperatures: List[int], output_dir: str,
                         frame_selection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract representative frames for a domain.
    
    Args:
        reader: MdCATH reader instance
        domain_id: Domain identifier
        temperatures: List of temperatures to process
        output_dir: Base directory to save output PDB files
        frame_selection: Frame selection parameters
        
    Returns:
        Dictionary with extraction results and statistics
    """
    log_info(f"Extracting frames for domain {domain_id}", domain_id=domain_id)
    
    results = {
        "domain_id": domain_id,
        "success": False,
        "temperatures_processed": [],
        "frames_extracted": {},
        "total_frames": 0
    }
    
    # Extract frame selection parameters
    method = frame_selection.get("method", "regular")
    num_frames = frame_selection.get("num_frames", 10)
    
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
        
        # Get atom data for mapping
        atom_data = reader.get_atom_data()
        if not atom_data:
            log_error(f"Failed to get atom data for domain {domain_id}", 
                     domain_id=domain_id, error_type="extraction_error")
            return results
        
        # Create domain output directory
        domain_dir = os.path.join(output_dir, domain_id)
        
        # Process each temperature
        for temp in temperatures:
            if temp not in metadata["temperatures"]:
                log_info(f"Temperature {temp}K not available for domain {domain_id}", domain_id=domain_id)
                continue
            
            # Create temperature output directory
            temp_dir = os.path.join(domain_dir, str(temp))
            os.makedirs(temp_dir, exist_ok=True)
            
            log_info(f"Processing temperature {temp}K for domain {domain_id}", domain_id=domain_id)
            
            # Select replica with most frames
            replica_counts = metadata["replica_counts"].get(temp, {})
            if not replica_counts:
                log_info(f"No replicas available for domain {domain_id}, temperature {temp}K", 
                        domain_id=domain_id)
                continue
            
            best_replica = max(replica_counts.items(), key=lambda x: x[1])[0]
            log_info(f"Selected replica {best_replica} with {replica_counts[best_replica]} frames", 
                    domain_id=domain_id)
            
            # Get trajectory data
            trajectory_data = reader.get_trajectory_data(temp, best_replica)
            if not trajectory_data or "coords" not in trajectory_data:
                log_error(f"Failed to get trajectory data for domain {domain_id}, temperature {temp}K", 
                         domain_id=domain_id, error_type="extraction_error")
                continue
            
            # Select frames
            frame_indices = select_frames(trajectory_data, method, num_frames)
            if not frame_indices:
                log_error(f"Failed to select frames for domain {domain_id}, temperature {temp}K", 
                         domain_id=domain_id, error_type="extraction_error")
                continue
            
            log_info(f"Selected {len(frame_indices)} frames for domain {domain_id}, temperature {temp}K", 
                    domain_id=domain_id)
            
            # Extract frames and save PDB files
            frames_extracted = 0
            
            for frame_idx in frame_indices:
                # Get coordinates for this frame
                coords = trajectory_data["coords"][frame_idx]
                
                # Create PDB content
                pdb_content = create_pdb_from_frame(
                    domain_id, atom_data, coords, frame_idx, temp, best_replica)
                
                if not pdb_content:
                    log_error(f"Failed to create PDB for domain {domain_id}, temperature {temp}K, "
                             f"frame {frame_idx}", domain_id=domain_id, error_type="extraction_error")
                    continue
                
                # Save PDB file
                pdb_path = os.path.join(temp_dir, f"{domain_id}_temp{temp}_frame{frame_idx}.pdb")
                with open(pdb_path, "w") as f:
                    f.write(pdb_content)
                
                frames_extracted += 1
                log_info(f"Saved frame {frame_idx} to {pdb_path}", domain_id=domain_id)
            
            # Update results
            if frames_extracted > 0:
                results["temperatures_processed"].append(temp)
                results["frames_extracted"][temp] = frames_extracted
                results["total_frames"] += frames_extracted
        
        # Mark as successful if at least one temperature was processed
        if results["temperatures_processed"]:
            results["success"] = True
            log_info(f"Successfully extracted frames for domain {domain_id}", domain_id=domain_id)
            log_progress(domain_id, "success")
        else:
            log_error(f"No temperatures were successfully processed for domain {domain_id}", 
                     domain_id=domain_id, error_type="extraction_error")
            log_progress(domain_id, "error", "extraction_error")
        
    except Exception as e:
        log_error(f"Error extracting frames for domain {domain_id}: {e}", 
                 domain_id=domain_id, error_type="other_error", exc_info=True)
        log_progress(domain_id, "error", "other_error")
    finally:
        reader.close()
    
    return results

def process_domain_frames(args: Tuple) -> Dict[str, Any]:
    """
    Process frames for a domain (multiprocessing wrapper).
    
    Args:
        args: Tuple containing (mdcath_dir, domain_id, temperatures, output_dir, frame_selection)
        
    Returns:
        Dictionary with extraction results
    """
    mdcath_dir, domain_id, temperatures, output_dir, frame_selection = args
    reader = MdCathReader(mdcath_dir)
    return extract_domain_frames(reader, domain_id, temperatures, output_dir, frame_selection)

def extract_all_frames(mdcath_dir: str, domain_ids: List[str], 
                      temperatures: List[int], output_dir: str,
                      frame_selection: Dict[str, Any],
                      num_cores: int = 1) -> Dict[str, Any]:
    """
    Extract frames for multiple domains using multiprocessing.
    
    Args:
        mdcath_dir: Base directory containing mdCATH HDF5 files
        domain_ids: List of domain identifiers to process
        temperatures: List of temperatures to process
        output_dir: Directory to save output PDB files
        frame_selection: Frame selection parameters
        num_cores: Number of processor cores to use
        
    Returns:
        Dictionary with extraction statistics
    """
    import multiprocessing as mp
    from functools import partial
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    log_info(f"Extracting frames for {len(domain_ids)} domains using {num_cores} cores")
    
    # Prepare arguments for each domain
    args_list = [(mdcath_dir, domain_id, temperatures, output_dir, frame_selection) 
                 for domain_id in domain_ids]
    
    # Initialize statistics
    stats = {
        "total_domains": len(domain_ids),
        "successful_domains": 0,
        "failed_domains": 0,
        "temperatures_processed": {temp: 0 for temp in temperatures},
        "total_frames": 0
    }
    
    # Process domains using multiprocessing
    with mp.Pool(processes=num_cores) as pool:
        results = list(pool.map(process_domain_frames, args_list))
    
    # Compute statistics
    for result in results:
        if result["success"]:
            stats["successful_domains"] += 1
            
            # Count temperatures processed
            for temp in result["temperatures_processed"]:
                if temp in stats["temperatures_processed"]:
                    stats["temperatures_processed"][temp] += 1
            
            # Count total frames
            stats["total_frames"] += result["total_frames"]
        else:
            stats["failed_domains"] += 1
    
    log_info(f"Frame extraction completed: {stats['successful_domains']} successful, "
             f"{stats['failed_domains']} failed, {stats['total_frames']} total frames")
    
    return stats