"""
LeakageAudit Utility (Phase 1.3)

Hash patient IDs at each pipeline stage and confirm isolation
"""

import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional
from pathlib import Path


class LeakageAudit:
    """
    Audit utility for detecting data leakage
    
    Tracks patient IDs at each pipeline stage and verifies isolation
    """
    
    def __init__(self):
        """Initialize leakage audit"""
        self.stages: Dict[str, Set[str]] = {}
        self.stage_hashes: Dict[str, str] = {}
    
    def register_stage(self, stage_name: str, patient_ids: List[str]):
        """
        Register patient IDs for a pipeline stage
        
        Parameters
        ----------
        stage_name : str
            Name of the pipeline stage
        patient_ids : list
            List of patient IDs at this stage
        """
        patient_set = set(str(pid) for pid in patient_ids if pd.notna(pid))
        self.stages[stage_name] = patient_set
        
        # Create hash for this stage
        sorted_ids = sorted(patient_set)
        id_string = "|".join(sorted_ids)
        stage_hash = hashlib.md5(id_string.encode()).hexdigest()
        self.stage_hashes[stage_name] = stage_hash
    
    def check_isolation(self, stage1: str, stage2: str) -> Dict[str, any]:
        """
        Check isolation between two stages
        
        Parameters
        ----------
        stage1 : str
            First stage name
        stage2 : str
            Second stage name
            
        Returns
        -------
        dict
            Isolation check results
        """
        if stage1 not in self.stages:
            return {
                'passed': False,
                'error': f"Stage '{stage1}' not registered"
            }
        
        if stage2 not in self.stages:
            return {
                'passed': False,
                'error': f"Stage '{stage2}' not registered"
            }
        
        ids1 = self.stages[stage1]
        ids2 = self.stages[stage2]
        
        overlap = ids1 & ids2
        only_in_1 = ids1 - ids2
        only_in_2 = ids2 - ids1
        
        # For train/test split, overlap should be empty
        if 'train' in stage1.lower() and 'test' in stage2.lower():
            passed = len(overlap) == 0
            return {
                'passed': passed,
                'overlap_count': len(overlap),
                'overlapping_ids': list(overlap)[:10],  # First 10
                'only_in_stage1': len(only_in_1),
                'only_in_stage2': len(only_in_2),
                'message': 'PASSED' if passed else f'FAILED: {len(overlap)} patients in both stages'
            }
        else:
            # For other stages, overlap is expected
            return {
                'passed': True,
                'overlap_count': len(overlap),
                'overlapping_ids': list(overlap)[:10],
                'only_in_stage1': len(only_in_1),
                'only_in_stage2': len(only_in_2),
                'message': 'OK'
            }
    
    def generate_report(self) -> str:
        """
        Generate leakage audit report
        
        Returns
        -------
        str
            Human-readable report
        """
        lines = [
            "=" * 60,
            "DATA LEAKAGE AUDIT REPORT",
            "=" * 60,
            f"Registered Stages: {len(self.stages)}",
            ""
        ]
        
        # List all stages
        for stage_name, patient_set in self.stages.items():
            lines.append(f"Stage: {stage_name}")
            lines.append(f"  Patients: {len(patient_set)}")
            lines.append(f"  Hash: {self.stage_hashes[stage_name][:16]}...")
            lines.append("")
        
        # Check train/test isolation
        train_stages = [s for s in self.stages.keys() if 'train' in s.lower()]
        test_stages = [s for s in self.stages.keys() if 'test' in s.lower()]
        
        if train_stages and test_stages:
            lines.append("Train/Test Isolation Checks:")
            for train_stage in train_stages:
                for test_stage in test_stages:
                    check = self.check_isolation(train_stage, test_stage)
                    status = "✓" if check['passed'] else "✗"
                    lines.append(f"  {status} {train_stage} vs {test_stage}: {check['message']}")
                    if not check['passed']:
                        lines.append(f"    Overlap: {check['overlap_count']} patients")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_report(self, output_path: Path):
        """
        Save audit report to file
        
        Parameters
        ----------
        output_path : Path
            Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_report()
        with open(output_path, 'w') as f:
            f.write(report)
