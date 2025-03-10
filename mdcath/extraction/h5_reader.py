#!/usr/bin/env python3
"""
H5 file reader for accessing mdCATH dataset.
Provides functions to extract data from HDF5 files in the mdCATH format.
"""

import os
import h5py
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

class MdCathReader:
    """
    Reader class for mdCATH HDF5 files.
    Provides methods to access trajectory data, structure info, and metadata.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the reader with the base directory of mdCATH HDF5 files.
        
        Args:
            base_dir: Base directory containing mdCATH HDF5 files
        """
        self.base_dir = base_dir
        self.current_file = None
        self.current_domain = None
    
    def get_domain_path(self, domain_id: str) -> str:
        """
        Get path to the HDF5 file for a given domain.
        
        Args:
            domain_id: Domain identifier (e.g., '12asA00')
            
        Returns:
            Full path to the HDF5 file
        """
        return os.path.join(self.base_dir, f"mdcath_dataset_{domain_id}.h5")
    
    def open_domain(self, domain_id: str) -> bool:
        """
        Open the HDF5 file for a given domain.
        
        Args:
            domain_id: Domain identifier
            
        Returns:
            True if successful, False otherwise
        """
        h5_path = self.get_domain_path(domain_id)
        
        if not os.path.exists(h5_path):
            logger.error(f"H5 file not found for domain {domain_id}: {h5_path}")
            return False
        
        try:
            # Close previous file if open
            if self.current_file is not None:
                self.current_file.close()
            
            # Open new file
            self.current_file = h5py.File(h5_path, 'r')
            
            # Verify domain exists in file
            if domain_id not in self.current_file:
                logger.error(f"Domain {domain_id} not found in H5 file {h5_path}")
                self.current_file.close()
                self.current_file = None
                return False
            
            self.current_domain = domain_id
            logger.debug(f"Successfully opened H5 file for domain {domain_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening H5 file for domain {domain_id}: {e}", exc_info=True)
            if self.current_file is not None:
                self.current_file.close()
                self.current_file = None
            return False
    
    def close(self) -> None:
        """Close the currently open HDF5 file."""
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None
            self.current_domain = None
    
    def get_domain_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for the currently open domain.
        
        Returns:
            Dictionary with domain metadata
        """
        if self.current_file is None or self.current_domain is None:
            logger.error("No domain currently open")
            return {}
        
        try:
            domain_group = self.current_file[self.current_domain]
            metadata = {
                "domain_id": self.current_domain,
                "num_chains": domain_group.attrs.get("numChains", 0),
                "num_protein_atoms": domain_group.attrs.get("numProteinAtoms", 0),
                "num_residues": domain_group.attrs.get("numResidues", 0),
                "temperatures": [],
                "replica_counts": {}
            }
            
            # Get available temperatures and replica counts
            for temp in [320, 348, 379, 413, 450]:
                temp_str = str(temp)
                if temp_str in domain_group:
                    metadata["temperatures"].append(temp)
                    replica_counts = {}
                    
                    for replica in range(5):  # Typically 5 replicas
                        replica_str = str(replica)
                        if replica_str in domain_group[temp_str]:
                            num_frames = domain_group[temp_str][replica_str].attrs.get("numFrames", 0)
                            replica_counts[replica] = num_frames
                    
                    metadata["replica_counts"][temp] = replica_counts
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata for domain {self.current_domain}: {e}")
            return {}
    
    def get_atom_data(self) -> Dict[str, np.ndarray]:
        """
        Get atom data for the currently open domain.
        
        Returns:
            Dictionary with atom data arrays
        """
        if self.current_file is None or self.current_domain is None:
            logger.error("No domain currently open")
            return {}
        
        try:
            domain_group = self.current_file[self.current_domain]
            atom_data = {}
            
            # Get atom-level properties
            for prop in ["chain", "element", "resid", "resname", "z"]:
                if prop in domain_group:
                    # Handle string datasets
                    if prop in ["chain", "element", "resname"]:
                        raw_data = domain_group[prop][:]
                        # Convert byte strings to Python strings
                        atom_data[prop] = np.array([x.decode('utf-8') for x in raw_data])
                    else:
                        atom_data[prop] = domain_group[prop][:]
            
            return atom_data
            
        except Exception as e:
            logger.error(f"Error getting atom data for domain {self.current_domain}: {e}")
            return {}
    
    def get_pdb_content(self, protein_only: bool = True) -> str:
        """
        Get PDB content as string for the currently open domain.
        
        Args:
            protein_only: If True, get only protein atoms PDB, else full PDB
            
        Returns:
            PDB content as string
        """
        if self.current_file is None or self.current_domain is None:
            logger.error("No domain currently open")
            return ""
        
        try:
            domain_group = self.current_file[self.current_domain]
            pdb_field = "pdbProteinAtoms" if protein_only else "pdb"
            
            if pdb_field in domain_group:
                return domain_group[pdb_field][()].decode('utf-8')
            else:
                logger.warning(f"{pdb_field} not found for domain {self.current_domain}")
                
                # Try alternative if protein_only requested but not available
                if protein_only and "pdb" in domain_group:
                    logger.info(f"Using full PDB instead for domain {self.current_domain}")
                    return domain_group["pdb"][()].decode('utf-8')
                
                return ""
                
        except Exception as e:
            logger.error(f"Error getting PDB content for domain {self.current_domain}: {e}")
            return ""
    
    def get_trajectory_data(self, temperature: int, replica: int) -> Dict[str, np.ndarray]:
        """
        Get trajectory data for a specific temperature and replica.
        
        Args:
            temperature: Temperature value (e.g., 320)
            replica: Replica index (0-4)
            
        Returns:
            Dictionary with trajectory data arrays
        """
        if self.current_file is None or self.current_domain is None:
            logger.error("No domain currently open")
            return {}
        
        try:
            domain_group = self.current_file[self.current_domain]
            temp_str = str(temperature)
            replica_str = str(replica)
            
            if temp_str not in domain_group:
                logger.error(f"Temperature {temperature} not found for domain {self.current_domain}")
                return {}
            
            if replica_str not in domain_group[temp_str]:
                logger.error(f"Replica {replica} not found for domain {self.current_domain}, temperature {temperature}")
                return {}
            
            replica_group = domain_group[temp_str][replica_str]
            trajectory_data = {}
            
            # Get trajectory properties
            for prop in ["coords", "forces", "dssp", "gyrationRadius", "rmsd", "rmsf", "box"]:
                if prop in replica_group:
                    # For string datasets like DSSP, convert to Python strings
                    if prop == "dssp":
                        raw_data = replica_group[prop][:]
                        # Convert 2D array of byte strings to 2D array of Python strings
                        trajectory_data[prop] = np.array([[c.decode('utf-8') for c in row] for row in raw_data])
                    else:
                        trajectory_data[prop] = replica_group[prop][:]
            
            return trajectory_data
            
        except Exception as e:
            logger.error(f"Error getting trajectory data for domain {self.current_domain}, "
                         f"temperature {temperature}, replica {replica}: {e}")
            return {}
    
    def get_frame_coordinates(self, temperature: int, replica: int, frame_idx: int) -> np.ndarray:
        """
        Get coordinates for a specific frame.
        
        Args:
            temperature: Temperature value
            replica: Replica index
            frame_idx: Frame index
            
        Returns:
            Numpy array of coordinates (N×3)
        """
        if self.current_file is None or self.current_domain is None:
            logger.error("No domain currently open")
            return np.array([])
        
        try:
            domain_group = self.current_file[self.current_domain]
            temp_str = str(temperature)
            replica_str = str(replica)
            
            if temp_str not in domain_group or replica_str not in domain_group[temp_str]:
                logger.error(f"Temperature {temperature} or replica {replica} not found")
                return np.array([])
            
            replica_group = domain_group[temp_str][replica_str]
            
            if "coords" not in replica_group:
                logger.error(f"Coordinates not found for domain {self.current_domain}, "
                             f"temperature {temperature}, replica {replica}")
                return np.array([])
            
            coords_dataset = replica_group["coords"]
            num_frames = coords_dataset.shape[0]
            
            if frame_idx < 0 or frame_idx >= num_frames:
                logger.error(f"Frame index {frame_idx} out of range (0-{num_frames-1})")
                return np.array([])
            
            # Get coordinates for the specified frame
            return coords_dataset[frame_idx]
            
        except Exception as e:
            logger.error(f"Error getting frame coordinates: {e}")
            return np.array([])
    
    def get_rmsf_data(self, temperature: int, replica: int) -> Dict[str, np.ndarray]:
        """
        Get RMSF data for a specific temperature and replica.
        
        Args:
            temperature: Temperature value
            replica: Replica index
            
        Returns:
            Dictionary with RMSF data and related residue information
        """
        if self.current_file is None or self.current_domain is None:
            logger.error("No domain currently open")
            return {}
        
        try:
            # Get atom data for residue mapping
            atom_data = self.get_atom_data()
            if not atom_data:
                return {}
            
            # Get trajectory data for RMSF values
            trajectory_data = self.get_trajectory_data(temperature, replica)
            if not trajectory_data or "rmsf" not in trajectory_data:
                return {}
            
            # Extract unique residues (preserving order)
            resids = atom_data["resid"]
            resnames = atom_data["resname"]
            
            # Get unique residues while preserving order
            unique_resids = []
            unique_resnames = []
            seen = set()
            
            for i in range(len(resids)):
                resid = resids[i]
                if resid not in seen:
                    seen.add(resid)
                    unique_resids.append(resid)
                    unique_resnames.append(resnames[i])
            
            # Validate RMSF length matches number of residues
            rmsf_values = trajectory_data["rmsf"]
            if len(rmsf_values) != len(unique_resids):
                logger.warning(f"RMSF length ({len(rmsf_values)}) doesn't match number of unique "
                               f"residues ({len(unique_resids)}) for domain {self.current_domain}")
                
                # Attempt to recover if RMSF length matches number of residues attribute
                domain_group = self.current_file[self.current_domain]
                num_residues = domain_group.attrs.get("numResidues", 0)
                
                if len(rmsf_values) == num_residues:
                    logger.info(f"RMSF length matches numResidues attribute ({num_residues})")
                else:
                    logger.error(f"Cannot reconcile RMSF length with residue data")
                    return {}
            
            # Compile RMSF data
            rmsf_data = {
                "domain_id": self.current_domain,
                "temperature": temperature,
                "replica": replica,
                "resids": np.array(unique_resids),
                "resnames": np.array(unique_resnames),
                "rmsf_values": rmsf_values
            }
            
            return rmsf_data
            
        except Exception as e:
            logger.error(f"Error getting RMSF data: {e}")
            return {}

def list_available_domains(base_dir: str) -> List[str]:
    """
    List available domains in the mdCATH dataset directory.
    
    Args:
        base_dir: Base directory containing mdCATH HDF5 files
        
    Returns:
        List of domain identifiers
    """
    domains = []
    h5_files = [f for f in os.listdir(base_dir) if f.startswith("mdcath_dataset_") and f.endswith(".h5")]
    
    for h5_file in h5_files:
        # Extract domain ID from filename
        domain_id = h5_file.replace("mdcath_dataset_", "").replace(".h5", "")
        domains.append(domain_id)
    
    return domains