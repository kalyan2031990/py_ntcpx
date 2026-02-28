# FILE: tests/test_data/generate_synthetic_data.py

import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_dvh(patient_id: str, organ: str, output_dir: Path):
    """Generate realistic synthetic DVH for testing"""
    np.random.seed(hash(patient_id + organ) % 2**32)
    
    # Realistic dose range for HN radiotherapy
    max_dose = np.random.uniform(50, 75)  # Gy
    doses = np.linspace(0, max_dose, 100)
    
    # Sigmoid-like cumulative DVH
    d50 = np.random.uniform(20, 40)
    slope = np.random.uniform(0.1, 0.3)
    volumes = 100 / (1 + np.exp(slope * (doses - d50)))
    
    # Add small noise
    volumes = np.clip(volumes + np.random.normal(0, 1, len(volumes)), 0, 100)
    volumes[0] = 100  # V(0) = 100%
    volumes = np.sort(volumes)[::-1]  # Ensure monotonic
    
    df = pd.DataFrame({'Dose[Gy]': doses, 'Volume[%]': volumes})
    
    output_file = output_dir / f"{patient_id}_{organ}.csv"
    df.to_csv(output_file, index=False)
    return output_file


def generate_synthetic_clinical_data(n_patients: int = 54) -> pd.DataFrame:
    """Generate synthetic clinical data matching your cohort structure"""
    np.random.seed(42)
    
    patients = []
    for i in range(n_patients):
        patient_id = f"SYN_{i:03d}"
        
        # Realistic distributions based on HN cancer cohorts
        dmean = np.random.uniform(15, 50)
        
        # Toxicity probability increases with dose (realistic)
        toxicity_prob = 1 / (1 + np.exp(-0.1 * (dmean - 30)))
        toxicity = np.random.binomial(1, toxicity_prob)
        
        patients.append({
            'PatientID': patient_id,
            'PrimaryPatientID': patient_id,  # Add PrimaryPatientID for compatibility
            'Organ': 'Parotid',
            'Dmean': dmean,
            'V30': np.random.uniform(30, 80),
            'V45': np.random.uniform(10, 50),
            'Age': np.random.randint(40, 80),
            'Chemotherapy': np.random.choice([0, 1]),
            'T_Stage': np.random.choice(['T1', 'T2', 'T3', 'T4']),
            'Observed_Toxicity': toxicity
        })
    
    return pd.DataFrame(patients)


def generate_full_test_dataset(output_dir: Path):
    """Generate complete synthetic dataset for pipeline testing"""
    output_dir = Path(output_dir)
    dvh_dir = output_dir / 'dvh'
    dvh_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate clinical data
    clinical_df = generate_synthetic_clinical_data(54)
    clinical_df.to_excel(output_dir / 'synthetic_clinical.xlsx', index=False)
    
    # Generate DVH files for each patient
    for _, row in clinical_df.iterrows():
        generate_synthetic_dvh(row['PatientID'], row['Organ'], dvh_dir)
    
    print(f"Generated synthetic dataset:")
    print(f"   - {len(clinical_df)} patients")
    print(f"   - Clinical data: {output_dir / 'synthetic_clinical.xlsx'}")
    print(f"   - DVH files: {dvh_dir}")
    
    return output_dir


if __name__ == '__main__':
    generate_full_test_dataset(Path('tests/test_data/synthetic'))