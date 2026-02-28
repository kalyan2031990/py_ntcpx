#!/usr/bin/env python3
"""
Utility functions for robust NTCP pipeline
==========================================
Column normalization and flexible DVH file matching for Head & Neck cancer
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re


def normalize_columns(df):
    """Normalize column names to handle case variations"""
    if df is None or df.empty:
        return df
    
    column_map = {}
    for col in df.columns:
        normalized = col.lower().replace('_', '').replace(' ', '').replace('-', '')
        
        if 'patientid' in normalized or 'patient_id' in normalized:
            column_map[col] = 'PatientID'
        elif 'patientname' in normalized or 'patient_name' in normalized:
            column_map[col] = 'PatientName'
        elif 'anoid' in normalized or 'patient_ano' in normalized or 'patientano' in normalized:
            column_map[col] = 'PatientAnoID'
        elif 'organ' in normalized:
            column_map[col] = 'Organ'
        elif 'toxicity' in normalized and 'observed' not in normalized:
            column_map[col] = 'Toxicity'
        elif 'totaldose' in normalized or 'total_dose' in normalized:
            column_map[col] = 'TotalDose'
        elif 'nfrac' in normalized or 'numfraction' in normalized or 'num_frac' in normalized:
            column_map[col] = 'NumFractions'
        elif 'doseperfracion' in normalized or 'dpf' in normalized or 'dose_per_fraction' in normalized:
            column_map[col] = 'DosePerFraction'
        elif 'alphabeta' in normalized or 'alpha_beta' in normalized:
            column_map[col] = 'AlphaBeta'
        elif 'followup' in normalized or 'follow_up' in normalized:
            column_map[col] = 'FollowUp'
        elif 'duration' in normalized and 'treatment' in normalized:
            column_map[col] = 'TreatmentDuration'
        elif 'technique' in normalized:
            column_map[col] = 'Technique'
        elif 'diagnosis' in normalized:
            column_map[col] = 'Diagnosis'
        elif 'sex' in normalized or 'gender' in normalized:
            column_map[col] = 'Sex'
        elif 'age' in normalized:
            column_map[col] = 'Age'
    
    if column_map:
        df = df.rename(columns=column_map)
    
    return df


def find_dvh_file(dvh_dir, patient_id, patient_name, organ):
    """Flexible DVH file finder with multiple matching strategies for H&N OARs"""
    import os
    from pathlib import Path
    import re
    
    dvh_path = Path(dvh_dir)
    
    # Normalize organ name - H&N specific variants
    organ_variants = {
        'Parotid': ['parotid', 'prtd', 'parot', 'tot_parotd', 'comb_prtd', 'lt_parotid', 'rt_parotid', 'parotid_gland'],
        'SpinalCord': ['spinalcord', 'cord', 'spinal_cord', 'spinal', 'sc'],
        'Larynx': ['larynx', 'glottic', 'supraglottic', 'laryngeal'],
        'OralCavity': ['oral', 'oralcavity', 'mucosa', 'oral_cavity', 'oral_mucosa'],
        'Submandibular': ['submandibular', 'submand', 'smg', 'lt_submand', 'rt_submand', 'submandibular_gland'],
        'PharyngealConstrictor': ['pharyngeal', 'constrictor', 'pcm', 'superior_constrictor', 'middle_constrictor', 'inferior_constrictor', 'pharyngeal_constrictor'],
        'Mandible': ['mandible', 'jaw', 'mandibular'],
        'Esophagus': ['esophagus', 'oesophagus', 'eso'],
        'BrachialPlexus': ['brachial', 'plexus', 'brachial_plexus', 'bp'],
        'Cochlea': ['cochlea', 'inner_ear'],
        'OpticNerve': ['optic', 'optic_nerve', 'opticnerve', 'on'],
        'Chiasm': ['chiasm', 'optic_chiasm', 'oc'],
        'Brainstem': ['brainstem', 'brain_stem', 'bs']
    }
    
    # Get organ patterns
    organ_patterns = organ_variants.get(organ, [organ.lower()])
    
    # Normalize patient identifiers
    patient_id_str = str(patient_id).lower().replace('-', '').replace('_', '').replace(' ', '')
    patient_name_str = str(patient_name).lower().replace(' ', '').replace('-', '').replace('_', '') if pd.notna(patient_name) else ''
    
    # Try multiple matching patterns
    for file in dvh_path.glob('*.csv'):
        filename_lower = file.stem.lower().replace('_', '').replace(' ', '').replace('-', '')
        
        # Check if organ matches
        organ_match = any(o.replace('_', '').replace('-', '') in filename_lower for o in organ_patterns)
        
        if organ_match:
            # Check patient match (ID, AnoID, or Name)
            patient_match = (
                patient_id_str in filename_lower or
                patient_name_str in filename_lower or
                any(part in filename_lower for part in patient_name_str.split() if len(part) > 2)
            )
            
            if patient_match:
                return file
    
    # Also try .txt files (for code1 input)
    for file in dvh_path.glob('*.txt'):
        filename_lower = file.stem.lower().replace('_', '').replace(' ', '').replace('-', '')
        
        organ_match = any(o.replace('_', '').replace('-', '') in filename_lower for o in organ_patterns)
        
        if organ_match:
            patient_match = (
                patient_id_str in filename_lower or
                patient_name_str in filename_lower or
                any(part in filename_lower for part in patient_name_str.split() if len(part) > 2)
            )
            
            if patient_match:
                return file
    
    return None

