"""
Baseline Capture Script (Phase 0.1)

Captures baseline outputs for regression testing
"""

import sys
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def calculate_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def capture_baseline(
    output_dir: Path,
    baseline_dir: Path,
    random_seed: int = 42
) -> dict:
    """
    Capture baseline outputs for regression testing
    
    Parameters
    ----------
    output_dir : Path
        Directory containing pipeline outputs
    baseline_dir : Path
        Directory to save baseline reference
    random_seed : int
        Random seed used for this run
        
    Returns
    -------
    dict
        Baseline metadata with file hashes
    """
    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_data = {
        'timestamp': datetime.now().isoformat(),
        'random_seed': random_seed,
        'output_dir': str(output_dir),
        'files': {}
    }
    
    # Find all result files
    result_files = []
    for pattern in ['*.xlsx', '*.csv', '*.json']:
        result_files.extend(output_dir.rglob(pattern))
    
    # Calculate hashes for classical NTCP outputs
    classical_ntcp_files = [
        f for f in result_files 
        if 'ntcp' in f.name.lower() and 'ml' not in f.name.lower()
    ]
    
    for file_path in classical_ntcp_files:
        try:
            file_hash = calculate_file_hash(file_path)
            relative_path = file_path.relative_to(output_dir)
            baseline_data['files'][str(relative_path)] = {
                'hash': file_hash,
                'size': file_path.stat().st_size,
                'type': 'classical_ntcp'
            }
        except Exception as e:
            print(f"Warning: Could not hash {file_path}: {e}")
    
    # Save baseline metadata
    baseline_metadata_file = baseline_dir / 'baseline_metadata.json'
    with open(baseline_metadata_file, 'w') as f:
        json.dump(baseline_data, f, indent=2)
    
    # Copy key files to baseline directory
    baseline_outputs = baseline_dir / 'outputs'
    baseline_outputs.mkdir(exist_ok=True)
    
    print(f"Baseline captured: {len(baseline_data['files'])} files")
    print(f"Baseline directory: {baseline_dir}")
    
    return baseline_data


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Capture baseline for regression testing")
    parser.add_argument("--output_dir", required=True, help="Pipeline output directory")
    parser.add_argument("--baseline_dir", default="baseline_reference", help="Baseline storage directory")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed used")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        sys.exit(1)
    
    baseline_data = capture_baseline(
        output_dir,
        Path(args.baseline_dir),
        args.random_seed
    )
    
    print("Baseline capture complete!")
    print(f"Metadata saved to: {Path(args.baseline_dir) / 'baseline_metadata.json'}")


if __name__ == '__main__':
    main()
