#!/usr/bin/env python3
"""
run_pipeline.py - Complete NTCP Pipeline Orchestrator
=====================================================

Orchestrates the complete py_ntcpx v1.0 pipeline execution:
1. code1_dvh_preprocess
2. code2_dvh_plot_and_summary
3. code2_bDVH (NEW - Biological DVH)
4. code3_ntcp_analysis_ml
5. code4_ntcp_output_QA_reporter
6. code5_ntcp_factors_analysis
7. code6_publication_diagrams
8. shap_code7 (True-Model SHAP - Clinical Grade)
9. supp_results_summary (NEW - Publication tables)

Software: py_ntcpx_v1.0.0
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

# Windows-safe encoding configuration
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

# Import Windows-safe utilities
try:
    from windows_safe_utils import safe_encode_unicode, safe_log
except ImportError:
    # Fallback if module not available
    def safe_encode_unicode(text):
        return str(text).replace('✓', '[OK]').replace('✗', '[FAIL]').replace('→', '->')
    def safe_log(log_func, message, *args, **kwargs):
        log_func(safe_encode_unicode(message), *args, **kwargs)

# Import contract validator
try:
    from contract_validator import ContractValidator
except ImportError:
    ContractValidator = None
    logger.warning("Contract validator not available - contract validation will be skipped")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrate complete NTCP pipeline execution"""
    
    def __init__(self, base_output_dir: Path = Path('out2')):
        """
        Initialize orchestrator
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output directories (single canonical layout; no code5_output created until Step 5 runs)
        self.code0_output = self.base_output_dir / 'code0_output'
        self.code1_output = self.base_output_dir / 'code1_output'
        self.code2_output = self.base_output_dir / 'code2_output'
        self.code2_bdvh_output = self.base_output_dir / 'code2_bDVH_output'
        self.code3_output = self.base_output_dir / 'code3_output'
        self.code4_output = self.base_output_dir / 'code4_output'
        self.code5_output = self.base_output_dir / 'code5_output'  # Step 5 writes here (single place for factors)
        self.code6_output = self.base_output_dir / 'code6_output'
        self.shap_output = self.base_output_dir / 'code7_shap'
        self.supp_output = self.base_output_dir / 'supp_results_summary_output'
        self.tiered_output = self.base_output_dir / 'tiered_output'
        
        # Initialize contract validator
        self.contracts_dir = self.base_output_dir / 'contracts'
        if ContractValidator is not None:
            self.contract_validator = ContractValidator(self.contracts_dir)
        else:
            self.contract_validator = None
    
    def validate_contract_for_step(self, step_name: str, required_contract: str) -> bool:
        """Validate required contract exists before executing a step"""
        if self.contract_validator is None:
            return True  # Skip validation if contract validator not available
        
        if required_contract and not self.contract_validator.validate_contract_exists(required_contract):
            logger.error(f"[ERROR] Required contract '{required_contract}.xlsx' not found for {step_name}")
            logger.error("Run previous steps to generate required contracts.")
            return False
        return True
    
    def _generate_step3_contract(self):
        """Generate Step3_NTCPDataset contract from code3 output"""
        if self.contract_validator is None:
            return
        
        try:
            # Find NTCP results file
            results_file = None
            for f in self.code3_output.glob("*.xlsx"):
                if "result" in f.name.lower() or "complete" in f.name.lower() or "ntcp" in f.name.lower():
                    results_file = f
                    break
            
            if results_file and results_file.exists():
                contract_path = self.contract_validator.get_contract_path("Step3_NTCPDataset")
                # Copy results file as contract (or create a simplified version)
                import shutil
                shutil.copy2(results_file, contract_path)
                logger.info(f"[CONTRACT] Step3_NTCPDataset.xlsx created from {results_file.name}")
        except Exception as e:
            logger.warning(f"Could not generate Step3 contract: {e}")
    
    def run_command(self, cmd: list, step_name: str) -> bool:
        """
        Run a command and handle errors (hardened for Windows compatibility)
        
        Args:
            cmd: Command as list of strings
            step_name: Name of the step for logging
        
        Returns:
            True if successful, False otherwise
        """
        safe_log(logger.info, "\n%s", '='*60)
        safe_log(logger.info, "Step: %s", step_name)
        safe_log(logger.info, "Command: %s", ' '.join(cmd))
        safe_log(logger.info, "%s\n", '='*60)
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=3600  # 1 hour timeout
            )
            
            # Log stdout
            if result.stdout:
                safe_log(logger.info, result.stdout)
            
            # Log stderr if present
            if result.stderr:
                safe_log(logger.warning, result.stderr)
            
            safe_log(logger.info, "[OK] %s completed successfully", step_name)
            return True
            
        except subprocess.CalledProcessError as e:
            safe_log(logger.error, "[FAIL] %s failed with exit code %d", step_name, e.returncode)
            if e.stdout:
                safe_log(logger.error, "STDOUT: %s", e.stdout)
            if e.stderr:
                safe_log(logger.error, "STDERR: %s", e.stderr)
            safe_log(logger.error, "Exit code: %d", e.returncode)
            return False
        except FileNotFoundError:
            safe_log(logger.error, "[FAIL] Command not found: %s", cmd[0])
            logger.error("Please ensure Python scripts are in the current directory")
            return False
        except subprocess.TimeoutExpired:
            safe_log(logger.error, "[ERROR] %s timed out after 1 hour", step_name)
            return False
        except Exception as e:
            safe_log(logger.error, "[ERROR] Unexpected error in %s: %s", step_name, str(e))
            import traceback
            safe_log(logger.error, "Traceback: %s", traceback.format_exc())
            return False
    
    def step1_dvh_preprocess(self, input_txt_dir: Path) -> bool:
        """Step 1: DVH preprocessing"""
        cmd = [
            sys.executable,
            'code1_dvh_preprocess.py',
            str(input_txt_dir),
            '--outdir', str(self.code1_output)
        ]
        return self.run_command(cmd, "Step 1: DVH Preprocessing")
    
    def step2_dvh_plot_summary(self) -> bool:
        """Step 2: DVH plotting and summary - with contract validation"""
        # Validate Step1 contract exists
        if not self.validate_contract_for_step("Step 2", "Step1_DVHRegistry"):
            return False
        
        # code2_dvh_plot_and_summary.py expects: code1_dir (positional) --outdir (optional)
        if not self.code1_output.exists():
            logger.error(f"Code1 output directory not found: {self.code1_output}")
            return False
        
        cmd = [
            sys.executable,
            'code2_dvh_plot_and_summary.py',
            str(self.code1_output),  # Positional argument: code1_dir
            '--outdir', str(self.code2_output)
        ]
        success = self.run_command(cmd, "Step 2: DVH Plotting & Summary")
        
        # Generate Step2 contract after successful execution
        if success and self.contract_validator is not None:
            try:
                # Create summary contract from code2 output
                summary_file = self.code2_output / "dose_metrics_cohort.xlsx"
                if summary_file.exists():
                    contract_path = self.contract_validator.get_contract_path("Step2_DVHSummary")
                    import shutil
                    shutil.copy2(summary_file, contract_path)
                    logger.info(f"[CONTRACT] Step2_DVHSummary.xlsx created")
            except Exception as e:
                logger.warning(f"Could not generate Step2 contract: {e}")
        
        return success
    
    def step2_bdvh(self, clinical_file: Optional[Path] = None) -> bool:
        """Step 2b: Biological DVH generation - with contract validation"""
        # Validate Step1 contract exists
        if not self.validate_contract_for_step("Step 2b", "Step1_DVHRegistry"):
            return False
        
        ddvh_dir = self.code1_output / 'dDVH_csv'
        if not ddvh_dir.exists():
            logger.error(f"dDVH directory not found: {ddvh_dir}")
            return False
        
        cmd = [
            sys.executable,
            'code2_bDVH.py',
            '--input_dir', str(ddvh_dir),
            '--output_dir', str(self.code2_bdvh_output),
            '--method', 'EQD2'
        ]
        
        if clinical_file and clinical_file.exists():
            cmd.extend(['--clinical_file', str(clinical_file)])
        
        success = self.run_command(cmd, "Step 2b: Biological DVH Generation")
        
        # Generate Step2b contract after successful execution
        if success and self.contract_validator is not None:
            try:
                # Create bDVH registry contract
                summary_file = self.code2_bdvh_output / "bDVH_summary.xlsx"
                if summary_file.exists():
                    contract_path = self.contract_validator.get_contract_path("Step2b_bDVHRegistry")
                    import shutil
                    shutil.copy2(summary_file, contract_path)
                    logger.info(f"[CONTRACT] Step2b_bDVHRegistry.xlsx created")
            except Exception as e:
                logger.warning(f"Could not generate Step2b contract: {e}")
        
        return success
    
    def step0_clinical_reconciliation(self, clinical_data_file: Path) -> Tuple[bool, Optional[Path]]:
        """Step 0: Clinical reconciliation (runs after Step 1, before Step 3)"""
        if not clinical_data_file.exists():
            logger.error(f"Clinical data file not found: {clinical_data_file}")
            return False, None
        
        cmd = [
            sys.executable,
            'code0_clinical_reconciliation.py',
            '--clinical_data', str(clinical_data_file),
            '--contracts_dir', str(self.contracts_dir),
            '--output_dir', str(self.base_output_dir / 'code0_output')
        ]
        
        success = self.run_command(cmd, "Step 0: Clinical Reconciliation")
        
        if success:
            # Check if reconciled file was created
            reconciled_file = self.base_output_dir / 'code0_output' / 'clinical_patient_data_reconciled.xlsx'
            if reconciled_file.exists():
                logger.info(f"[OK] Reconciled clinical data: {reconciled_file}")
                return True, reconciled_file
            else:
                # Original file was already valid
                return True, clinical_data_file
        
        return False, None
    
    def _run_step0_and_update_reconciled(self, patient_data_file: Path, reconciled_file_container: dict) -> bool:
        """Internal helper to run Step 0 and update reconciled_file reference"""
        success, reconciled_file_result = self.step0_clinical_reconciliation(patient_data_file)
        if not success:
            logger.error("[FAIL] Step 0 (Clinical Reconciliation) failed. Cannot proceed to Step 3.")
            return False
        # Update reconciled_file in the container (will be used by step3 and step5)
        if reconciled_file_result:
            reconciled_file_container['file'] = reconciled_file_result
        return True
    
    def step3_ntcp_analysis(self, patient_data_file: Path) -> bool:
        """Step 3: NTCP analysis with ML - with contract validation"""
        ddvh_dir = self.code1_output / 'dDVH_csv'
        if not ddvh_dir.exists():
            logger.error(f"dDVH directory not found: {ddvh_dir}")
            return False
        
        # CRITICAL: Clinical Contract v2 - check that reconciled file exists
        # Step 0 already validated the original clinical file and created clinical_reconciled.xlsx
        # Step 3 now uses the reconciled file (Clinical Contract v2) instead of the original
        reconciled_file = self.base_output_dir / 'code0_output' / 'clinical_reconciled.xlsx'
        
        if not reconciled_file.exists():
            logger.error("=" * 60)
            logger.error("CLINICAL CONTRACT v2 VALIDATION FAILED")
            logger.error("=" * 60)
            logger.error(f"Reconciled clinical file not found: {reconciled_file}")
            logger.error("Run Step 0 (code0_clinical_reconciliation.py) first to generate clinical_reconciled.xlsx")
            logger.error("=" * 60)
            return False
        
        # Step 0 already validated Clinical Contract v2 (patient_id, xerostomia_grade2plus, followup_months)
        # and identity matching (DVH.PrimaryPatientID == clinical.patient_id)
        # No need to re-validate here - just verify the file exists and has patient_id
        try:
            with pd.ExcelFile(reconciled_file) as xl:
                clinical_df = pd.read_excel(xl, sheet_name=xl.sheet_names[0])
            
            # Quick check: Clinical Contract v2 requires patient_id
            if 'patient_id' not in clinical_df.columns:
                logger.error("=" * 60)
                logger.error("CLINICAL CONTRACT v2 VALIDATION FAILED")
                logger.error("=" * 60)
                logger.error("Reconciled clinical file missing 'patient_id' column (Clinical Contract v2)")
                logger.error("=" * 60)
                return False
            
            logger.info(f"[OK] Clinical Contract v2 validated: {len(clinical_df)} patients in reconciled file")
            
            # Log match statistics if contract validator available
            if self.contract_validator is not None:
                registry_df = self.contract_validator.load_step1_registry()
                if registry_df is not None:
                    # Map patient_id to PrimaryPatientID for statistics (DVH uses PrimaryPatientID)
                    clinical_df_for_stats = clinical_df.copy()
                    clinical_df_for_stats['PrimaryPatientID'] = clinical_df_for_stats['patient_id']
                    self.contract_validator.log_match_statistics(registry_df, clinical_df_for_stats)
        except Exception as e:
            logger.warning(f"Could not validate reconciled clinical data: {e}")
        
        cmd = [
            sys.executable,
            'code3_ntcp_analysis_ml.py',
            '--dvh_dir', str(ddvh_dir),
            '--patient_data', str(patient_data_file),
            '--output_dir', str(self.code3_output)
        ]
        success = self.run_command(cmd, "Step 3: NTCP Analysis with ML")
        
        # Generate Step3 contract after successful execution
        if success and self.contract_validator is not None:
            self._generate_step3_contract()
        
        return success
    
    def step3_quantec_stratification(self) -> bool:
        """Step 3b: QUANTEC stratification (runs after Step 3)"""
        # Find NTCP results file
        ntcp_results_file = None
        for f in self.code3_output.glob("*.xlsx"):
            if "result" in f.name.lower() or "ntcp" in f.name.lower():
                ntcp_results_file = f
                break
        
        if not ntcp_results_file or not ntcp_results_file.exists():
            logger.warning("NTCP results file not found for QUANTEC stratification")
            return False
        
        bins_config = Path('quantification/quantec_bins.json')
        if not bins_config.exists():
            logger.warning(f"QUANTEC bins configuration not found: {bins_config}")
            return False
        
        quantec_output = self.code3_output / 'quantec_validation'
        
        cmd = [
            sys.executable,
            'quantification/quantec_stratifier.py',
            '--ntcp_results', str(ntcp_results_file),
            '--bins_config', str(bins_config),
            '--output_dir', str(quantec_output)
        ]
        
        success = self.run_command(cmd, "Step 3b: QUANTEC Stratification")
        return success
    
    def step3c_tiered_analysis(self, clinical_file: Optional[Path] = None) -> bool:
        """Step 3c: Tiered NTCP Analysis (runs after Step 3)"""
        # Check if code3 output exists
        if not self.code3_output.exists():
            logger.error(f"Code3 output not found: {self.code3_output}")
            return False
        
        # Check if DVH directory exists
        dvh_dir = self.code1_output / 'dDVH_csv'
        if not dvh_dir.exists():
            logger.error(f"DVH directory not found: {dvh_dir}")
            return False
        
        # Create tiered output directory only when step runs
        tiered_output = self.tiered_output
        tiered_output.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable,
            'tiered_ntcp_analysis.py',
            '--code3_output', str(self.code3_output),
            '--dvh_dir', str(dvh_dir),
            '--output_dir', str(tiered_output)
        ]
        
        if clinical_file and clinical_file.exists():
            cmd.extend(['--clinical_file', str(clinical_file)])
        
        success = self.run_command(cmd, "Step 3c: Tiered NTCP Analysis")
        return success
    
    def step4_qa_reporter(self) -> bool:
        """Step 4: QA reporter - with contract validation"""
        # Validate Step1 and Step3 contracts exist
        if not self.validate_contract_for_step("Step 4", "Step1_DVHRegistry"):
            return False
        if not self.validate_contract_for_step("Step 4", "Step3_NTCPDataset"):
            return False
        
        # Validate Step3 dataset matches registry
        if self.contract_validator is not None:
            registry_df = self.contract_validator.load_step1_registry()
            if registry_df is not None:
                step3_contract = self.contract_validator.get_contract_path("Step3_NTCPDataset")
                is_valid, error_msg = self.contract_validator.validate_step3_dataset(step3_contract, registry_df)
                if not is_valid:
                    logger.error(f"[ERROR] Step3 dataset validation failed: {error_msg}")
                    logger.error("Cannot proceed with ROC/calibration - dataset mismatch")
                    return False
        
        if not self.code3_output.exists():
            logger.error(f"Code3 output not found: {self.code3_output}")
            return False
        
        cmd = [
            sys.executable,
            'code4_ntcp_output_QA_reporter.py',
            '--input', str(self.code3_output),
            '--report_outdir', str(self.code4_output)
        ]
        success = self.run_command(cmd, "Step 4: QA Reporter")
        
        # Generate Step4 contract after successful execution
        if success and self.contract_validator is not None:
            try:
                qa_file = self.code4_output / "qa_summary_tables.xlsx"
                if qa_file.exists():
                    contract_path = self.contract_validator.get_contract_path("Step4_QAReport")
                    import shutil
                    shutil.copy2(qa_file, contract_path)
                    logger.info(f"[CONTRACT] Step4_QAReport.xlsx created")
            except Exception as e:
                logger.warning(f"Could not generate Step4 contract: {e}")
        
        return success
    
    def step5_factors_analysis(self, patient_data_file: Path) -> bool:
        """Step 5: Clinical factors analysis (writes to code5_output for single canonical location)"""
        if not self.code3_output.exists():
            logger.error(f"Code3 output not found: {self.code3_output}")
            return False
        
        if not patient_data_file.exists():
            logger.error(f"Patient data file not found: {patient_data_file}")
            return False
        
        # Single output location for factors (no duplicate under code3_output)
        self.code5_output.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            'code5_ntcp_factors_analysis.py',
            '--input_file', str(patient_data_file),
            '--enhanced_output_dir', str(self.code3_output),
            '--output_dir', str(self.code5_output)
        ]
        return self.run_command(cmd, "Step 5: Clinical Factors Analysis")
    
    def step6_publication_diagrams(self) -> bool:
        """Step 6: Publication diagrams"""
        cmd = [
            sys.executable,
            'code6_publication_diagrams.py',
            '--output_dir', str(self.code6_output),
            '--dpi', '1200'
        ]
        return self.run_command(cmd, "Step 6: Publication Diagrams")
    
    def step7_shap_true(self) -> bool:
        """Step 7: True-Model SHAP (Clinical Grade)"""
        if not self.code3_output.exists():
            logger.error(f"Code3 output not found: {self.code3_output}")
            return False
        
        # Check if shap_code7.py exists
        shap_script = Path('shap_code7.py')
        if not shap_script.exists():
            logger.warning(f"SHAP script not found: {shap_script}. Skipping.")
            return True
        
        cmd = [
            sys.executable,
            "shap_code7.py",
            "--code3_dir", str(self.code3_output),
            "--outdir", os.path.join(str(self.base_output_dir), "code7_shap")
        ]
        return self.run_command(cmd, "Step 7: True-Model SHAP (Clinical Grade)")
    
    def step8_supp_results_summary(self, clinical_file: Optional[Path] = None) -> bool:
        """Step 8: Publication tables summary (NEW)"""
        # Check all required outputs exist (excluding code5_output which may not exist)
        required_dirs = [
            self.code1_output,
            self.code2_output,
            self.code3_output,
            self.code4_output
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.error(f"Required output directory not found: {dir_path}")
                return False
        
        # Step 8 input: factors from code5_output only (single canonical location)
        step8_input = self.code5_output
        if not step8_input.exists():
            logger.error("Step-8 cannot find Step-5 outputs. Run Step 5 first. Expected: " + str(step8_input))
            return False
        
        cmd = [
            sys.executable,
            'supp_results_summary.py',
            '--code1_output', str(self.code1_output),
            '--code2_output', str(self.code2_output),
            '--code3_output', str(self.code3_output),
            '--code4_output', str(self.code4_output),
            '--code5_output', str(step8_input),
            '--output_dir', str(self.supp_output)
        ]
        
        if clinical_file and clinical_file.exists():
            cmd.extend(['--clinical_file', str(clinical_file)])
        
        return self.run_command(cmd, "Step 8: Publication Tables Summary")
    
    def run_complete_pipeline(self, input_txt_dir: Path, patient_data_file: Path,
                            clinical_file: Optional[Path] = None,
                            skip_steps: Optional[list] = None,
                            resume_from: Optional[str] = None) -> bool:
        """
        Run complete pipeline
        
        Args:
            input_txt_dir: Directory containing input DVH text files
            patient_data_file: Excel file with patient data and toxicity
            clinical_file: Optional Excel file with clinical factors
            skip_steps: List of step numbers to skip (e.g., [6, 7])
        
        Returns:
            True if all steps successful, False otherwise
        """
        skip_steps = skip_steps or []
        
        logger.info("=" * 60)
        logger.info("py_ntcpx v1.0 - Complete Pipeline Orchestration")
        logger.info("=" * 60)
        logger.info(f"Input DVH directory: {input_txt_dir}")
        logger.info(f"Patient data file: {patient_data_file}")
        logger.info(f"Base output directory: {self.base_output_dir}")
        logger.info("=" * 60)
        
        # Track reconciled file using a mutable container (will be updated by Step 0)
        reconciled_file_container = {'file': patient_data_file}
        
        steps = [
            ('step1', 1, "DVH Preprocessing", lambda: self.step1_dvh_preprocess(input_txt_dir)),
            ('step0', 1.5, "Clinical Reconciliation", lambda: self._run_step0_and_update_reconciled(patient_data_file, reconciled_file_container)),
            ('step2', 2, "DVH Plotting & Summary", self.step2_dvh_plot_summary),
            ('step2b', 2.5, "Biological DVH Generation", lambda: self.step2_bdvh(clinical_file)),
            ('step3', 3, "NTCP Analysis with ML", lambda: self.step3_ntcp_analysis(reconciled_file_container['file'])),
            ('step3b', 3.5, "QUANTEC Stratification", self.step3_quantec_stratification),
            ('step3c', 3.6, "Tiered NTCP Analysis", lambda: self.step3c_tiered_analysis(clinical_file)),
            ('step4', 4, "QA Reporter", self.step4_qa_reporter),
            ('step5', 5, "Clinical Factors Analysis", lambda: self.step5_factors_analysis(reconciled_file_container['file'])),
            ('step6', 6, "Publication Diagrams", self.step6_publication_diagrams),
            ('step7', 7, "True-Model SHAP (Clinical Grade)", self.step7_shap_true),
            ('step8', 8, "Publication Tables Summary", lambda: self.step8_supp_results_summary(clinical_file))
        ]
        
        # Handle resume_from
        if resume_from:
            logger.info(f"\n[RESUME] Resuming pipeline from {resume_from}")
            
            # If resuming from step3 or later, ensure step0 and step1 have run
            if resume_from in ['step3', 'step3b', 'step4', 'step5']:
                # Check if Step 1 registry exists
                registry_path = self.contracts_dir / 'Step1_DVHRegistry.xlsx'
                if not registry_path.exists():
                    logger.error("[FAIL] Step 1 (DVH Preprocessing) must run before Step 0. Cannot resume from this step.")
                    return False
                
                # Check if reconciliation has been done
                reconciled_file_check = self.base_output_dir / 'code0_output' / 'clinical_patient_data_reconciled.xlsx'
                if not reconciled_file_check.exists():
                    logger.warning("Step 0 (Clinical Reconciliation) not completed. Running now...")
                    step0_success, reconciled_file_result = self.step0_clinical_reconciliation(patient_data_file)
                    if not step0_success:
                        logger.error("[FAIL] Step 0 (Clinical Reconciliation) failed. Cannot proceed.")
                        return False
                    if reconciled_file_result:
                        reconciled_file_container['file'] = reconciled_file_result
            
            found_resume = False
            filtered_steps = []
            for step_id, step_num, step_name, step_func in steps:
                if step_id == resume_from:
                    found_resume = True
                    filtered_steps.append((step_id, step_num, step_name, step_func))
                elif found_resume:
                    filtered_steps.append((step_id, step_num, step_name, step_func))
            
            if not found_resume:
                logger.error(f"Invalid resume_from step: {resume_from}")
                return False
            
            steps = filtered_steps
            logger.info(f"Will execute {len(steps)} step(s) starting from {resume_from}")
        
        success_count = 0
        failed_steps = []
        
        for step_id, step_num, step_name, step_func in steps:
            if step_num in skip_steps:
                logger.info(f"\nSkipping step {step_num}: {step_name}")
                continue
            
            if step_func():
                success_count += 1
            else:
                failed_steps.append((step_id, step_num, step_name))
                logger.error(f"\nPipeline stopped at {step_id} (step {step_num}): {step_name}")
                break  # Stop on first failure
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Execution Summary")
        logger.info("=" * 60)
        logger.info(f"Completed steps: {success_count}/{len(steps) - len(skip_steps)}")
        
        if failed_steps:
            logger.error("Failed steps:")
            for step_id, step_num, step_name in failed_steps:
                logger.error(f"  - {step_id} (step {step_num}): {step_name}")
            return False
        else:
            safe_log(logger.info, "[OK] All steps completed successfully!")
            logger.info(f"\nOutput directory: {self.base_output_dir.absolute()}")
            
            # Write single OUTPUT_INDEX and README at base (no copying, no structured_output)
            try:
                self._write_output_index()
            except Exception as e:
                logger.warning(f"Could not write output index: {e}")
            
            return True

    def _write_output_index(self) -> None:
        """
        Write OUTPUT_INDEX.csv and README.txt at base by scanning actual output tree.
        No file copying; single canonical layout only.
        """
        base = self.base_output_dir
        step_from_dir = {
            "code0_output": "Step 0 (Clinical)",
            "code1_output": "Step 1 (DVH data)",
            "code2_output": "Step 2 (DVH summary)",
            "code2_bDVH_output": "Step 2b (bDVH)",
            "code3_output": "Step 3 (NTCP)",
            "code4_output": "Step 4 (QA)",
            "code5_output": "Step 5 (Factors)",
            "code6_output": "Step 6 (Diagrams)",
            "code7_shap": "Step 7 (SHAP/LIME)",
            "tiered_output": "Step 3c (Tiered)",
            "supp_results_summary_output": "Step 8 (Publication)",
            "contracts": "Contracts",
        }
        index_rows = []
        for root, _dirs, files in os.walk(base):
            root_path = Path(root)
            try:
                rel = root_path.relative_to(base)
            except ValueError:
                continue
            parts = rel.parts
            top = parts[0] if parts else ""
            step = step_from_dir.get(top, "Pipeline")
            for f in files:
                if f.startswith(".") or f == "OUTPUT_INDEX.csv" or f == "README.txt":
                    continue
                full = root_path / f
                try:
                    rel_file = full.relative_to(base)
                except ValueError:
                    continue
                index_rows.append({
                    "path": str(rel_file).replace("\\", "/"),
                    "file": f,
                    "step": step,
                })
        if index_rows:
            pd.DataFrame(index_rows).sort_values("path").to_csv(base / "OUTPUT_INDEX.csv", index=False)
        readme = base / "README.txt"
        readme.write_text(
            "py_ntcpx_v1.0.0 – Pipeline output\n"
            "==================================\n\n"
            "Single canonical layout. No duplicate copies.\n\n"
            "• code0_output/   – Clinical reconciliation.\n"
            "• code1_output/    – DVH preprocessing (cDVH, dDVH).\n"
            "• code2_output/    – DVH plots and dose metrics.\n"
            "• code2_bDVH_output/ – Biological DVH.\n"
            "• code3_output/    – NTCP results, models, plots, reports.\n"
            "• code4_output/    – QA summary and report.\n"
            "• code5_output/    – Clinical factors analysis.\n"
            "• code6_output/    – Publication diagrams.\n"
            "• code7_shap/      – SHAP and LIME explanations.\n"
            "• tiered_output/   – Four-tier NTCP.\n"
            "• supp_results_summary_output/ – Publication tables.\n"
            "• contracts/       – Step contracts.\n\n"
            "OUTPUT_INDEX.csv lists every file with path and step. See docs/OUTPUT_INDEX.md for details.\n",
            encoding="utf-8",
        )


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Orchestrate complete py_ntcpx_v1.0.0 pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline execution order:
1. code1_dvh_preprocess - DVH preprocessing
2. code2_dvh_plot_and_summary - DVH plotting
3. code2_bDVH - Biological DVH generation (NEW)
4. code3_ntcp_analysis_ml - NTCP analysis with ML
5. code4_ntcp_output_QA_reporter - QA reporting
6. code5_ntcp_factors_analysis - Clinical factors analysis
7. code6_publication_diagrams - Publication diagrams
8. shap_code7 - True-Model SHAP analysis (Clinical Grade)
9. supp_results_summary - Publication tables (NEW)

Software: py_ntcpx_v1.0.0
        """
    )
    
    parser.add_argument(
        '--input_txt_dir',
        type=str,
        required=True,
        help='Directory containing input DVH text files'
    )
    
    parser.add_argument(
        '--patient_data',
        type=str,
        required=True,
        help='Excel file with patient data and toxicity (PatientID, Organ, Observed_Toxicity)'
    )
    
    parser.add_argument(
        '--clinical_file',
        type=str,
        default=None,
        help='Optional Excel file with clinical factors'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='out2',
        help='Base output directory (default: out2)'
    )
    
    parser.add_argument(
        '--skip',
        type=int,
        nargs='+',
        default=[],
        help='Step numbers to skip (e.g., --skip 6 7)'
    )
    
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        choices=['step1', 'step0', 'step2', 'step2b', 'step3', 'step3b', 'step3c', 'step4', 'step5', 'step6', 'step7'],
        help='Resume pipeline from a specific step (validates required contracts before continuing)'
    )
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator(base_output_dir=Path(args.output_dir))
    
    input_txt_dir = Path(args.input_txt_dir).expanduser().resolve() if args.input_txt_dir else None
    patient_data_file = Path(args.patient_data).expanduser().resolve() if args.patient_data else None
    clinical_file = Path(args.clinical_file).expanduser().resolve() if args.clinical_file else None
    
    # Validate inputs (skip if resuming from later steps)
    if not args.resume_from or args.resume_from == 'step1':
        if not input_txt_dir or not input_txt_dir.exists():
            logger.error(f"Input directory not found: {input_txt_dir}")
            return 1
    
    if not args.resume_from or args.resume_from in ['step3', 'step5']:
        if not patient_data_file or not patient_data_file.exists():
            logger.error(f"Patient data file not found: {patient_data_file}")
            logger.error("Patient data file is required for step3 and step5")
            return 1
    
    success = orchestrator.run_complete_pipeline(
        input_txt_dir=input_txt_dir or Path('dummy'),
        patient_data_file=patient_data_file or Path('dummy'),
        clinical_file=clinical_file,
        skip_steps=args.skip,
        resume_from=args.resume_from
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

