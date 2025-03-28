"""
Process antibody PDB files from SAbDab dataset and extract heavy/light chain sequences
Using multiprocessing for faster processing
"""

import os
from pathlib import Path
import pandas as pd
from typing import Tuple, Optional, Dict, List
import logging
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from functools import partial

from esm.utils.structure.protein_complex import ProteinComplex

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def identify_antibody_chains(complex: ProteinComplex) -> Tuple[Optional[str], Optional[str]]:
    """
    Identify heavy and light chains from a ProteinComplex
    
    Strategy:
    1. First try standard H/L naming
    2. If not found, identify based on sequence length and composition
    
    Args:
        complex: ProteinComplex object containing antibody structure
        
    Returns:
        Tuple of (heavy_chain_sequence, light_chain_sequence)
    """
    chains = list(complex.chain_iter())
    
    if len(chains) < 2:
        logging.warning("Less than 2 chains found")
        return None, None
        
    # Try to find standard H/L naming first
    chain_dict = {chain.chain_id: chain for chain in chains}
    if 'H' in chain_dict and 'L' in chain_dict:
        return (chain_dict['H'].sequence, chain_dict['L'].sequence)
    
    # If standard naming not found, use length-based identification
    # Sort chains by length (typically heavy chain is longer)
    chains = sorted(chains, key=lambda x: len(x.sequence), reverse=True)
    
    # Basic validation of chain lengths
    if len(chains[0].sequence) < 100 or len(chains[1].sequence) < 100:
        logging.warning(f"Suspicious chain lengths: {[len(c.sequence) for c in chains]}")
        return None, None
        
    # Additional validation based on typical length ranges
    heavy_len = len(chains[0].sequence)
    light_len = len(chains[1].sequence)
    
    # Typical ranges for antibody chains
    if not (400 > heavy_len > 100 and 300 > light_len > 100):
        logging.warning(f"Chain lengths outside typical ranges: H={heavy_len}, L={light_len}")
        return None, None
        
    return chains[0].sequence, chains[1].sequence

def load_antibody_chains(pdb_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Load heavy and light chains from an antibody PDB
    
    Args:
        pdb_path: Path to the antibody PDB file
        
    Returns:
        Tuple of (heavy_chain_sequence, light_chain_sequence)
        Returns (None, None) if loading fails
    """
    try:
        # Load the complex
        complex = ProteinComplex.from_pdb(pdb_path)
        
        # Identify heavy and light chains
        return identify_antibody_chains(complex)
        
    except Exception as e:
        logging.error(f"Error loading {pdb_path}: {str(e)}")
        return None, None

def process_single_pdb(pdb_file: Path) -> Optional[Dict]:
    """
    Process a single PDB file and return its data
    
    Args:
        pdb_file: Path to the PDB file
        
    Returns:
        Dictionary containing the processed data or None if processing failed
    """
    pdb_id = pdb_file.stem
    heavy_seq, light_seq = load_antibody_chains(str(pdb_file))
    
    if heavy_seq is not None and light_seq is not None:
        return {
            'pdb_id': pdb_id,
            'heavy_chain': heavy_seq,
            'light_chain': light_seq,
            'heavy_length': len(heavy_seq),
            'light_length': len(light_seq)
        }
    return None

def process_sabdab_dataset(
    sabdab_dir: str,
    output_csv: str,
    max_files: Optional[int] = None,
    n_processes: Optional[int] = None
) -> None:
    """
    Process all PDB files in SAbDab directory and save sequences to CSV using multiple processes
    
    Args:
        sabdab_dir: Directory containing antibody PDB files
        output_csv: Path to output CSV file
        max_files: Optional maximum number of files to process (for testing)
        n_processes: Number of processes to use (defaults to CPU count)
    """
    # Get list of PDB files
    pdb_files = list(Path(sabdab_dir).glob("*.pdb"))
    if max_files:
        pdb_files = pdb_files[:max_files]
    
    logging.info(f"Found {len(pdb_files)} PDB files to process")
    
    # Set number of processes
    if n_processes is None:
        n_processes = mp.cpu_count()
    logging.info(f"Using {n_processes} processes")
    
    # Process files in parallel
    results = []
    with mp.Pool(processes=n_processes) as pool:
        # Use tqdm to show progress
        for result in tqdm(
            pool.imap(process_single_pdb, pdb_files),
            total=len(pdb_files),
            desc="Processing PDB files"
        ):
            if result is not None:
                results.append(result)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    logging.info(f"Processed {len(results)} antibodies successfully")
    logging.info(f"Results saved to {output_csv}")
    
    # Print statistics
    if len(results) > 0:
        logging.info("\nSequence length statistics:")
        logging.info(f"Heavy chain: mean={df['heavy_length'].mean():.1f}, min={df['heavy_length'].min()}, max={df['heavy_length'].max()}")
        logging.info(f"Light chain: mean={df['light_length'].mean():.1f}, min={df['light_length'].min()}, max={df['light_length'].max()}")

def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    sabdab_dir = project_root / "datasets" / "raw" / "sabdab"
    output_dir = project_root / "datasets" / "processed"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process dataset
    process_sabdab_dataset(
        sabdab_dir=str(sabdab_dir),
        output_csv=str(output_dir / "sabdab_sequences.csv"),
        max_files=None,  # Set to a number for testing with subset of data
        n_processes=8  # Will use CPU count by default
    )

if __name__ == "__main__":
    main()