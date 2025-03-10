#!/usr/bin/env python3
"""
Command-line interface for the mdCATH processor pipeline.
Provides commands to run various processing steps.
"""

import os
import sys
import argparse
import logging
import threading
import time
from typing import Dict, List, Optional, Any

from mdcath.config import load_config
from mdcath.utils.logging_utils import (
    setup_logging, initialize_progress_tracking, 
    start_progress_thread, get_progress_summary
)
from mdcath.extraction.h5_reader import MdCathReader, list_available_domains
from mdcath.extraction.rmsf_extractor import extract_all_rmsf
from mdcath.extraction.frame_extractor import extract_all_frames
from mdcath.processing.pdb_cleaner import clean_all_pdbs
from mdcath.processing.rmsf_analyzer import analyze_all_rmsf

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="mdCATH processor pipeline for MD trajectory data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List available domains
    list_parser = subparsers.add_parser("list", help="List available domains")
    list_parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of domains listed (0 for all)"
    )
    
    # Extract RMSF data
    rmsf_parser = subparsers.add_parser("extract-rmsf", help="Extract RMSF data")
    rmsf_parser.add_argument(
        "--domains", type=str, nargs="+", default=[],
        help="List of domains to process (empty for all)"
    )
    rmsf_parser.add_argument(
        "--temps", type=str, nargs="+", default=[],
        help="List of temperatures to process (empty for all)"
    )
    
    # Extract frames
    frames_parser = subparsers.add_parser("extract-frames", help="Extract trajectory frames")
    frames_parser.add_argument(
        "--domains", type=str, nargs="+", default=[],
        help="List of domains to process (empty for all)"
    )
    frames_parser.add_argument(
        "--temps", type=str, nargs="+", default=[],
        help="List of temperatures to process (empty for all)"
    )
    frames_parser.add_argument(
        "--num-frames", type=int, default=0,
        help="Number of frames to extract per temperature (0 for config default)"
    )
    frames_parser.add_argument(
        "--method", type=str, choices=["regular", "rmsd", "gyration", "random"], default="",
        help="Frame selection method (empty for config default)"
    )
    
    # Clean PDB files
    clean_parser = subparsers.add_parser("clean-pdbs", help="Clean extracted PDB files")
    
    # Analyze RMSF data
    analyze_parser = subparsers.add_parser("analyze-rmsf", help="Analyze RMSF data")
    
    # Run all steps
    all_parser = subparsers.add_parser("all", help="Run all processing steps")
    all_parser.add_argument(
        "--domains", type=str, nargs="+", default=[],
        help="List of domains to process (empty for all)"
    )
    all_parser.add_argument(
        "--temps", type=str, nargs="+", default=[],
        help="List of temperatures to process (empty for all)"
    )
    
    return parser.parse_args()

def run_list_domains(config: Dict[str, Any], args) -> None:
    """List available domains in the mdCATH dataset."""
    mdcath_dir = config["input"]["mdcath_dir"]
    domains = list_available_domains(mdcath_dir)
    
    if not domains:
        print(f"No domains found in {mdcath_dir}")
        return
    
    # Sort domains
    domains.sort()
    
    # Apply limit if specified
    if args.limit > 0:
        domains = domains[:args.limit]
    
    print(f"Found {len(domains)} domains in {mdcath_dir}:")
    for i, domain in enumerate(domains, 1):
        print(f"{i:4d}. {domain}")

def run_extract_rmsf(config: Dict[str, Any], args) -> None:
    """Extract RMSF data for specified domains and temperatures."""
    # Get domains
    if args.domains:
        domains = args.domains
    else:
        domains = list_available_domains(config["input"]["mdcath_dir"])
    
    # Get temperatures
    if args.temps:
        temps = [int(t) for t in args.temps]
    else:
        temps = config["input"]["temperatures"]
    
    # Initialize progress tracking
    initialize_progress_tracking(len(domains))
    progress_thread = start_progress_thread()
    
    # Run extraction
    output_dir = os.path.join(config["output"]["base_dir"], config["output"]["rmsf_dir"])
    stats = extract_all_rmsf(
        config["input"]["mdcath_dir"],
        domains,
        temps,
        output_dir,
        config["performance"]["num_cores"]
    )
    
    # Wait for progress thread to finish
    progress_thread.join()
    
    # Print summary
    print("\nRMSF Extraction Summary:")
    print(f"Total domains processed: {stats['total_domains']}")
    print(f"Successful domains: {stats['successful_domains']}")
    print(f"Failed domains: {stats['failed_domains']}")
    print("\nTemperatures processed:")
    for temp, count in stats["temperatures_processed"].items():
        print(f"  {temp}K: {count} domains")
    print(f"\nAverage calculation: {stats['average_calculated']} domains")
    print(f"\nOutput saved to: {output_dir}")

def run_extract_frames(config: Dict[str, Any], args) -> None:
    """Extract trajectory frames for specified domains and temperatures."""
    # Get domains
    if args.domains:
        domains = args.domains
    else:
        domains = list_available_domains(config["input"]["mdcath_dir"])
    
    # Get temperatures
    if args.temps:
        temps = [int(t) for t in args.temps]
    else:
        temps = config["input"]["temperatures"]
    
    # Get frame selection parameters
    frame_selection = config["processing"]["frame_selection"].copy()
    if args.num_frames > 0:
        frame_selection["num_frames"] = args.num_frames
    if args.method:
        frame_selection["method"] = args.method
    
    # Initialize progress tracking
    initialize_progress_tracking(len(domains))
    progress_thread = start_progress_thread()
    
    # Run extraction
    output_dir = os.path.join(config["output"]["base_dir"], config["output"]["pdb_frames_dir"])
    stats = extract_all_frames(
        config["input"]["mdcath_dir"],
        domains,
        temps,
        output_dir,
        frame_selection,
        config["performance"]["num_cores"]
    )
    
    # Wait for progress thread to finish
    progress_thread.join()
    
    # Print summary
    print("\nFrame Extraction Summary:")
    print(f"Total domains processed: {stats['total_domains']}")
    print(f"Successful domains: {stats['successful_domains']}")
    print(f"Failed domains: {stats['failed_domains']}")
    print("\nTemperatures processed:")
    for temp, count in stats["temperatures_processed"].items():
        print(f"  {temp}K: {count} domains")
    print(f"\nTotal frames extracted: {stats['total_frames']}")
    print(f"\nOutput saved to: {output_dir}")

def run_clean_pdbs(config: Dict[str, Any], args) -> None:
    """Clean extracted PDB files."""
    input_dir = os.path.join(config["output"]["base_dir"], config["output"]["pdb_frames_dir"])
    output_dir = os.path.join(config["output"]["base_dir"], "cleaned_pdb_frames")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        print("Run extract-frames first to generate PDB files.")
        return
    
    # Initialize progress tracking
    domains = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    initialize_progress_tracking(len(domains))
    progress_thread = start_progress_thread()
    
    # Run cleaning
    stats = clean_all_pdbs(
        input_dir,
        output_dir,
        config["processing"]["cleaning"],
        config["performance"]["num_cores"]
    )
    
    # Wait for progress thread to finish
    progress_thread.join()
    
    # Print summary
    print("\nPDB Cleaning Summary:")
    print(f"Total domains processed: {stats['total_domains']}")
    print(f"Successful domains: {stats['successful_domains']}")
    print(f"Failed domains: {stats['failed_domains']}")
    print(f"Total PDSs cleaned: {stats['total_pdbs']}")
    print(f"\nOutput saved to: {output_dir}")

def run_analyze_rmsf(config: Dict[str, Any], args) -> None:
    """Analyze RMSF data and generate statistics."""
    input_dir = os.path.join(config["output"]["base_dir"], config["output"]["rmsf_dir"])
    output_dir = os.path.join(config["output"]["base_dir"], config["output"]["summary_dir"])
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        print("Run extract-rmsf first to generate RMSF data.")
        return
    
    # Run analysis
    stats = analyze_all_rmsf(input_dir, output_dir)
    
    # Print summary
    print("\nRMSF Analysis Summary:")
    print(f"Total domains analyzed: {stats['total_domains']}")
    print(f"Successful domains: {stats['successful_domains']}")
    print("\nDomains by temperature:")
    for temp, count in stats["domains_by_temperature"].items():
        print(f"  {temp}{'K' if temp != 'average' else ''}: {count} domains")
    print(f"\nOutput saved to: {output_dir}")

def run_all_steps(config: Dict[str, Any], args) -> None:
    """Run all processing steps in sequence."""
    print("Running all processing steps in sequence...")
    
    # Get domains
    if args.domains:
        domains = args.domains
    else:
        domains = list_available_domains(config["input"]["mdcath_dir"])
    
    # Get temperatures
    if args.temps:
        temps = [int(t) for t in args.temps]
    else:
        temps = config["input"]["temperatures"]
    
    # 1. Extract RMSF data
    print("\n=== Step 1: Extract RMSF data ===")
    
    # Initialize progress tracking
    initialize_progress_tracking(len(domains))
    progress_thread = start_progress_thread()
    
    rmsf_output_dir = os.path.join(config["output"]["base_dir"], config["output"]["rmsf_dir"])
    rmsf_stats = extract_all_rmsf(
        config["input"]["mdcath_dir"],
        domains,
        temps,
        rmsf_output_dir,
        config["performance"]["num_cores"]
    )
    
    # Wait for progress thread to finish
    progress_thread.join()
    
    # 2. Extract frames
    print("\n=== Step 2: Extract trajectory frames ===")
    
    # Initialize progress tracking
    initialize_progress_tracking(len(domains))
    progress_thread = start_progress_thread()
    
    frames_output_dir = os.path.join(config["output"]["base_dir"], config["output"]["pdb_frames_dir"])
    frames_stats = extract_all_frames(
        config["input"]["mdcath_dir"],
        domains,
        temps,
        frames_output_dir,
        config["processing"]["frame_selection"],
        config["performance"]["num_cores"]
    )
    
    # Wait for progress thread to finish
    progress_thread.join()
    
    # 3. Clean PDB files
    print("\n=== Step 3: Clean PDB files ===")
    
    # Initialize progress tracking
    frame_domains = [d for d in os.listdir(frames_output_dir) 
                     if os.path.isdir(os.path.join(frames_output_dir, d))]
    initialize_progress_tracking(len(frame_domains))
    progress_thread = start_progress_thread()
    
    cleaned_output_dir = os.path.join(config["output"]["base_dir"], "cleaned_pdb_frames")
    clean_stats = clean_all_pdbs(
        frames_output_dir,
        cleaned_output_dir,
        config["processing"]["cleaning"],
        config["performance"]["num_cores"]
    )
    
    # Wait for progress thread to finish
    progress_thread.join()
    
    # 4. Analyze RMSF data
    print("\n=== Step 4: Analyze RMSF data ===")
    
    summary_output_dir = os.path.join(config["output"]["base_dir"], config["output"]["summary_dir"])
    analysis_stats = analyze_all_rmsf(rmsf_output_dir, summary_output_dir)
    
    # Print overall summary
    print("\n=== Overall Processing Summary ===")
    print(f"Total domains: {len(domains)}")
    print(f"RMSF extraction: {rmsf_stats['successful_domains']} successful, {rmsf_stats['failed_domains']} failed")
    print(f"Frame extraction: {frames_stats['successful_domains']} successful, {frames_stats['failed_domains']} failed")
    print(f"PDB cleaning: {clean_stats['successful_domains']} successful, {clean_stats['failed_domains']} failed")
    print(f"RMSF analysis: {analysis_stats['successful_domains']} successful domains\n")
    print("Results saved to:")
    print(f"  RMSF data: {rmsf_output_dir}")
    print(f"  PDB frames: {frames_output_dir}")
    print(f"  Cleaned PDSs: {cleaned_output_dir}")
    print(f"  Analysis results: {summary_output_dir}")

def main():
    """Main entry point for the command-line interface."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    
    # Run appropriate command
    try:
        if args.command == "list":
            run_list_domains(config, args)
        elif args.command == "extract-rmsf":
            run_extract_rmsf(config, args)
        elif args.command == "extract-frames":
            run_extract_frames(config, args)
        elif args.command == "clean-pdbs":
            run_clean_pdbs(config, args)
        elif args.command == "analyze-rmsf":
            run_analyze_rmsf(config, args)
        elif args.command == "all":
            run_all_steps(config, args)
        else:
            print("No command specified. Use --help for usage information.")
    except Exception as e:
        logger.error(f"Error executing command {args.command}: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()