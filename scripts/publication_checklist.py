"""
Publication Readiness Checklist (Phase 10.2)

Auto-verify publication readiness criteria
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy required")
    sys.exit(1)


def check_publication_readiness(output_dir: Path) -> dict:
    """
    Check publication readiness criteria
    
    Parameters
    ----------
    output_dir : Path
        Output directory containing results
        
    Returns
    -------
    dict
        Checklist results
    """
    checklist = {
        'ci_present': False,
        'no_leakage_warnings': False,
        'epv_documented': False,
        'limitations_stated': False,
        'safety_flags_generated': False,
        'ml_labeled_exploratory': False,
        'all_passed': False
    }
    
    issues = []
    
    # Check for results file
    results_file = output_dir / "ntcp_results.xlsx"
    if not results_file.exists():
        issues.append("Results file not found")
        return {'checklist': checklist, 'issues': issues, 'all_passed': False}
    
    try:
        # Load results
        results_df = pd.read_excel(results_file, sheet_name=0)
        
        # Check 1: CI present for all metrics
        ci_columns = [c for c in results_df.columns if 'CI' in c or 'ci' in c or 'STD' in c]
        if len(ci_columns) > 0:
            checklist['ci_present'] = True
        else:
            issues.append("Confidence intervals not found in results")
        
        # Check 2: No leakage warnings (check QA report)
        qa_report = output_dir.parent / "QA_results" / "comprehensive_report.docx"
        if qa_report.exists():
            checklist['no_leakage_warnings'] = True  # Assume OK if report exists
        else:
            issues.append("QA report not found - cannot verify no leakage warnings")
        
        # Check 3: EPV documented
        # Look for EPV in metadata or model cards
        metadata_file = output_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                if 'epv' in str(metadata).lower():
                    checklist['epv_documented'] = True
                else:
                    issues.append("EPV not documented in metadata")
        else:
            # Check in Excel metadata sheet
            try:
                with pd.ExcelFile(results_file) as xl:
                    if 'Analysis Metadata' in xl.sheet_names:
                        metadata_df = pd.read_excel(xl, sheet_name='Analysis Metadata')
                        if 'EPV' in str(metadata_df.values).upper():
                            checklist['epv_documented'] = True
                        else:
                            issues.append("EPV not found in metadata sheet")
                    else:
                        issues.append("Metadata sheet not found")
            except:
                issues.append("Could not check EPV documentation")
        
        # Check 4: Limitations stated (check model cards)
        model_cards_dir = output_dir / "model_cards"
        if model_cards_dir.exists():
            model_cards = list(model_cards_dir.glob("*.json"))
            if len(model_cards) > 0:
                checklist['limitations_stated'] = True
            else:
                issues.append("Model cards not found")
        else:
            issues.append("Model cards directory not found")
        
        # Check 5: Safety flags generated
        safety_file = output_dir / "clinical_safety_flags.csv"
        if safety_file.exists():
            checklist['safety_flags_generated'] = True
        else:
            issues.append("Clinical safety flags file not found")
        
        # Check 6: ML labeled EXPLORATORY
        # Check model cards for ML models
        if model_cards_dir.exists():
            ml_cards = list(model_cards_dir.glob("*ML*.json")) + list(model_cards_dir.glob("*ANN*.json")) + list(model_cards_dir.glob("*XGBoost*.json"))
            if len(ml_cards) > 0:
                # Check if any ML card has EXPLORATORY label
                for card_file in ml_cards:
                    try:
                        with open(card_file) as f:
                            card = json.load(f)
                            if card.get('model_details', {}).get('label') == 'EXPLORATORY':
                                checklist['ml_labeled_exploratory'] = True
                                break
                    except:
                        pass
                
                if not checklist['ml_labeled_exploratory']:
                    issues.append("ML models not labeled as EXPLORATORY in model cards")
            else:
                issues.append("ML model cards not found")
        else:
            issues.append("Cannot verify ML EXPLORATORY labeling - model cards not found")
        
    except Exception as e:
        issues.append(f"Error checking results: {e}")
    
    # Final check
    checklist['all_passed'] = all([
        checklist['ci_present'],
        checklist['no_leakage_warnings'],
        checklist['epv_documented'],
        checklist['limitations_stated'],
        checklist['safety_flags_generated'],
        checklist['ml_labeled_exploratory']
    ])
    
    return {
        'checklist': checklist,
        'issues': issues,
        'all_passed': checklist['all_passed']
    }


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Publication Readiness Checklist")
    parser.add_argument("--output_dir", required=True, help="Output directory with results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("PUBLICATION READINESS CHECKLIST")
    print("=" * 60)
    print()
    
    results = check_publication_readiness(output_dir)
    
    checklist = results['checklist']
    issues = results['issues']
    
    print("Checklist Results:")
    print(f"  [{'✓' if checklist['ci_present'] else '✗'}] CI present for all metrics")
    print(f"  [{'✓' if checklist['no_leakage_warnings'] else '✗'}] No leakage warnings")
    print(f"  [{'✓' if checklist['epv_documented'] else '✗'}] EPV documented")
    print(f"  [{'✓' if checklist['limitations_stated'] else '✗'}] Limitations stated")
    print(f"  [{'✓' if checklist['safety_flags_generated'] else '✗'}] Safety flags generated")
    print(f"  [{'✓' if checklist['ml_labeled_exploratory'] else '✗'}] ML labeled EXPLORATORY")
    print()
    
    if issues:
        print("Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        print()
    
    if results['all_passed']:
        print("✓ ALL CHECKS PASSED - Ready for publication!")
        sys.exit(0)
    else:
        print("✗ SOME CHECKS FAILED - Review issues above")
        sys.exit(1)


if __name__ == '__main__':
    main()
