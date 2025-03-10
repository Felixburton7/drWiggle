#!/usr/bin/env python3
"""
RMSF analyzer module for mdCATH processor pipeline.
Analyzes RMSF data and generates summary statistics.
"""

import os
import numpy as np
import pandas as pd
import logging
import glob
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns

from mdcath.utils.logging_utils import log_info, log_error

logger = logging.getLogger(__name__)

def load_rmsf_data(csv_path: str) -> pd.DataFrame:
    """
    Load RMSF data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with RMSF data
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        log_error(f"Error loading RMSF data from {csv_path}: {e}")
        return pd.DataFrame()

def analyze_domain_rmsf(domain_id: str, rmsf_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Analyze RMSF data for a single domain.
    
    Args:
        domain_id: Domain identifier
        rmsf_dir: Directory containing RMSF data
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary with analysis results
    """
    log_info(f"Analyzing RMSF data for domain {domain_id}", domain_id=domain_id)
    
    results = {
        "domain_id": domain_id,
        "success": False,
        "temperatures_analyzed": [],
        "has_average": False,
        "statistics": {}
    }
    
    try:
        # Create output directory
        domain_output_dir = os.path.join(output_dir, domain_id)
        os.makedirs(domain_output_dir, exist_ok=True)
        
        # Load RMSF data for each temperature
        temp_data = {}
        
        for temp in ["320", "348", "379", "413", "450"]:
            temp_dir = os.path.join(rmsf_dir, temp)
            if not os.path.exists(temp_dir):
                continue
            
            # Search for RMSF file for this domain and temperature
            pattern = os.path.join(temp_dir, f"{domain_id}_temperature_{temp}_average_rmsf.csv")
            matching_files = glob.glob(pattern)
            
            if not matching_files:
                log_info(f"No RMSF data found for domain {domain_id}, temperature {temp}K", 
                        domain_id=domain_id)
                continue
            
            # Load the first matching file
            rmsf_path = matching_files[0]
            df = load_rmsf_data(rmsf_path)
            
            if df.empty:
                log_error(f"Failed to load RMSF data for domain {domain_id}, temperature {temp}K", 
                         domain_id=domain_id)
                continue
            
            temp_data[temp] = df
            results["temperatures_analyzed"].append(temp)
        
        # Load average RMSF data
        avg_dir = os.path.join(rmsf_dir, "average")
        if os.path.exists(avg_dir):
            pattern = os.path.join(avg_dir, f"{domain_id}_total_average_rmsf.csv")
            matching_files = glob.glob(pattern)
            
            if matching_files:
                avg_path = matching_files[0]
                avg_df = load_rmsf_data(avg_path)
                
                if not avg_df.empty:
                    temp_data["average"] = avg_df
                    results["has_average"] = True
        
        # If no data was loaded, return
        if not temp_data:
            log_error(f"No RMSF data found for domain {domain_id}", domain_id=domain_id)
            return results
        
        # Calculate statistics for each temperature
        statistics = {}
        
        for temp, df in temp_data.items():
            # Get RMSF column name
            rmsf_col = f"rmsf_{temp}" if temp != "average" else "average_rmsf"
            
            if rmsf_col not in df.columns:
                log_error(f"RMSF column {rmsf_col} not found in data for domain {domain_id}, "
                         f"temperature {temp}", domain_id=domain_id)
                continue
            
            # Calculate basic statistics
            rmsf_values = df[rmsf_col]
            stats = {
                "mean": rmsf_values.mean(),
                "median": rmsf_values.median(),
                "std": rmsf_values.std(),
                "min": rmsf_values.min(),
                "max": rmsf_values.max(),
                "quartile_25": rmsf_values.quantile(0.25),
                "quartile_75": rmsf_values.quantile(0.75),
                "num_residues": len(rmsf_values)
            }
            
            # Identify high RMSF residues (above 75th percentile)
            high_rmsf_threshold = stats["quartile_75"] + 1.5 * (stats["quartile_75"] - stats["quartile_25"])
            high_rmsf_residues = df[df[rmsf_col] > high_rmsf_threshold]
            
            stats["high_rmsf_threshold"] = high_rmsf_threshold
            stats["num_high_rmsf_residues"] = len(high_rmsf_residues)
            
            if not high_rmsf_residues.empty:
                # Get top 5 highest RMSF residues
                top_residues = high_rmsf_residues.sort_values(by=rmsf_col, ascending=False).head(5)
                stats["top_high_rmsf_residues"] = [{
                    "resid": row["resid"],
                    "resname": row["resname"],
                    "rmsf": row[rmsf_col]
                } for _, row in top_residues.iterrows()]
            
            statistics[temp] = stats
        
        # Save statistics to CSV
        stats_df = pd.DataFrame(statistics).T
        stats_df.index.name = "temperature"
        stats_df = stats_df.reset_index()
        
        stats_path = os.path.join(domain_output_dir, f"{domain_id}_rmsf_statistics.csv")
        stats_df.to_csv(stats_path, index=False)
        
        # Generate RMSF profile plot if average data available
        if "average" in temp_data:
            avg_df = temp_data["average"]
            
            plt.figure(figsize=(12, 6))
            plt.plot(avg_df["resid"], avg_df["average_rmsf"], marker='o', linestyle='-', markersize=4)
            plt.xlabel("Residue ID")
            plt.ylabel("Average RMSF (nm)")
            plt.title(f"RMSF Profile for Domain {domain_id}")
            plt.grid(True, alpha=0.3)
            
            # Highlight high RMSF residues
            high_rmsf_threshold = statistics["average"]["high_rmsf_threshold"]
            high_rmsf = avg_df[avg_df["average_rmsf"] > high_rmsf_threshold]
            
            if not high_rmsf.empty:
                plt.scatter(high_rmsf["resid"], high_rmsf["average_rmsf"], 
                           color='red', s=50, label=f"High RMSF (>{high_rmsf_threshold:.3f} nm)")
                plt.legend()
            
            plot_path = os.path.join(domain_output_dir, f"{domain_id}_rmsf_profile.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            plt.close()
        
        # Generate temperature comparison plot if multiple temperatures available
        if len([t for t in results["temperatures_analyzed"] if t != "average"]) > 1:
            plt.figure(figsize=(12, 6))
            
            for temp in sorted(results["temperatures_analyzed"]):
                if temp == "average":
                    continue
                
                df = temp_data[temp]
                rmsf_col = f"rmsf_{temp}"
                plt.plot(df["resid"], df[rmsf_col], marker='.', linestyle='-', 
                        label=f"{temp}K", alpha=0.7)
            
            plt.xlabel("Residue ID")
            plt.ylabel("RMSF (nm)")
            plt.title(f"RMSF Profile Across Temperatures for Domain {domain_id}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plot_path = os.path.join(domain_output_dir, f"{domain_id}_rmsf_temperature_comparison.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            plt.close()
        
        # Update results
        results["success"] = True
        results["statistics"] = statistics
        
        log_info(f"Successfully analyzed RMSF data for domain {domain_id}", domain_id=domain_id)
        
    except Exception as e:
        log_error(f"Error analyzing RMSF data for domain {domain_id}: {e}", 
                 domain_id=domain_id, exc_info=True)
    
    return results

def generate_summary_report(results: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
    """
    Generate a summary report from multiple domain analysis results.
    
    Args:
        results: List of domain analysis results
        output_dir: Directory to save summary report
        
    Returns:
        Dictionary with summary statistics
    """
    log_info(f"Generating summary report for {len(results)} domains")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize summary statistics
    summary = {
        "total_domains": len(results),
        "successful_domains": sum(1 for r in results if r["success"]),
        "domains_by_temperature": {},
        "average_statistics": {}
    }
    
    # Count domains for each temperature
    for temp in ["320", "348", "379", "413", "450", "average"]:
        summary["domains_by_temperature"][temp] = sum(
            1 for r in results if r["success"] and 
            (temp in r["temperatures_analyzed"] or (temp == "average" and r["has_average"]))
        )
    
    # Collect statistics across domains
    temp_stats = {temp: [] for temp in ["320", "348", "379", "413", "450", "average"]}
    
    for result in results:
        if not result["success"]:
            continue
        
        for temp, stats in result["statistics"].items():
            if temp in temp_stats:
                temp_stats[temp].append(stats)
    
    # Calculate average statistics for each temperature
    for temp, stats_list in temp_stats.items():
        if not stats_list:
            continue
        
        # Initialize dictionary for this temperature
        avg_stats = {}
        
        # Calculate average for numeric statistics
        for key in ["mean", "median", "std", "min", "max", 
                   "quartile_25", "quartile_75", "high_rmsf_threshold"]:
            values = [s[key] for s in stats_list if key in s]
            if values:
                avg_stats[key] = sum(values) / len(values)
        
        # Count statistics
        for key in ["num_residues", "num_high_rmsf_residues"]:
            values = [s[key] for s in stats_list if key in s]
            if values:
                avg_stats[key] = sum(values) / len(values)
        
        summary["average_statistics"][temp] = avg_stats
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary["average_statistics"]).T
    summary_df.index.name = "temperature"
    summary_df = summary_df.reset_index()
    
    summary_path = os.path.join(output_dir, "rmsf_summary_statistics.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Generate summary plots
    
    # 1. Bar plot of domains processed by temperature
    plt.figure(figsize=(10, 6))
    temps = ["320", "348", "379", "413", "450", "average"]
    counts = [summary["domains_by_temperature"][t] for t in temps]
    
    plt.bar(temps, counts)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Number of Domains")
    plt.title("Domains Processed by Temperature")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "domains_by_temperature.png"), dpi=300)
    plt.close()
    
    # 2. Average RMSF by temperature
    plt.figure(figsize=(10, 6))
    temps = ["320", "348", "379", "413", "450"]
    means = [summary["average_statistics"].get(t, {}).get("mean", 0) for t in temps]
    
    plt.plot(temps, means, marker='o', linestyle='-', linewidth=2)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Average RMSF (nm)")
    plt.title("Average RMSF by Temperature")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "average_rmsf_by_temperature.png"), dpi=300)
    plt.close()
    
    # 3. Box plot of RMSF distributions
    try:
        # Collect all RMSF data for box plot
        all_data = []
        labels = []
        
        for temp in ["320", "348", "379", "413", "450"]:
            # Find all CSV files for this temperature
            temp_dir = os.path.join(os.path.dirname(output_dir), temp)
            if not os.path.exists(temp_dir):
                continue
            
            csv_files = glob.glob(os.path.join(temp_dir, "*_average_rmsf.csv"))
            if not csv_files:
                continue
            
            # Load and concatenate data
            rmsf_values = []
            for csv_file in csv_files[:100]:  # Limit to 100 files for performance
                df = load_rmsf_data(csv_file)
                if not df.empty and f"rmsf_{temp}" in df.columns:
                    rmsf_values.extend(df[f"rmsf_{temp}"].tolist())
            
            if rmsf_values:
                all_data.append(rmsf_values)
                labels.append(f"{temp}K")
        
        if all_data:
            plt.figure(figsize=(10, 6))
            plt.boxplot(all_data, labels=labels, showfliers=False)
            plt.xlabel("Temperature (K)")
            plt.ylabel("RMSF (nm)")
            plt.title("RMSF Distribution by Temperature")
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(output_dir, "rmsf_distribution_by_temperature.png"), dpi=300)
            plt.close()
    except Exception as e:
        log_error(f"Error generating box plot: {e}", exc_info=True)
    
    return summary

def analyze_all_rmsf(rmsf_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Analyze RMSF data for all domains.
    
    Args:
        rmsf_dir: Directory containing RMSF data
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary with analysis statistics
    """
    log_info(f"Analyzing RMSF data in {rmsf_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of domains with RMSF data
    domain_ids = set()
    
    # Check average directory first
    avg_dir = os.path.join(rmsf_dir, "average")
    if os.path.exists(avg_dir):
        for csv_file in glob.glob(os.path.join(avg_dir, "*_total_average_rmsf.csv")):
            # Extract domain ID from filename
            filename = os.path.basename(csv_file)
            domain_id = filename.split("_total_average_rmsf.csv")[0]
            domain_ids.add(domain_id)
    
    # Check temperature directories
    for temp in ["320", "348", "379", "413", "450"]:
        temp_dir = os.path.join(rmsf_dir, temp)
        if not os.path.exists(temp_dir):
            continue
        
        for csv_file in glob.glob(os.path.join(temp_dir, "*_temperature_*_average_rmsf.csv")):
            # Extract domain ID from filename
            filename = os.path.basename(csv_file)
            domain_id = filename.split("_temperature_")[0]
            domain_ids.add(domain_id)
    
    if not domain_ids:
        log_error(f"No domains with RMSF data found in {rmsf_dir}")
        return {
            "total_domains": 0,
            "successful_domains": 0
        }
    
    log_info(f"Found {len(domain_ids)} domains with RMSF data")
    
    # Analyze each domain
    results = []
    for domain_id in sorted(domain_ids):
        result = analyze_domain_rmsf(domain_id, rmsf_dir, output_dir)
        results.append(result)
    
    # Generate summary report
    summary = generate_summary_report(results, output_dir)
    
    log_info(f"RMSF analysis completed: {summary['successful_domains']} successful domains out of {summary['total_domains']}")
    
    return summary