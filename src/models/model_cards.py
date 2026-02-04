"""
Model Cards for trained models (Phase 8)

Auto-generated model cards with:
- Intended use
- Data limits
- Failure modes
- Calibration status
- "EXPLORATORY" label for ML models
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import numpy as np


class ModelCardGenerator:
    """Generate model cards for trained models"""
    
    def __init__(self):
        """Initialize model card generator"""
        pass
    
    def generate_model_card(
        self,
        model_name: str,
        model_type: str,  # 'traditional', 'ml'
        organ: str,
        training_info: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        limitations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate model card
        
        Parameters
        ----------
        model_name : str
            Name of the model (e.g., 'LKB_LogLogit', 'ANN')
        model_type : str
            'traditional' or 'ml'
        organ : str
            Organ name
        training_info : dict
            Training information (n_samples, n_events, n_features, epv, etc.)
        performance_metrics : dict
            Performance metrics (auc, auc_ci, brier, calibration_slope, etc.)
        limitations : list, optional
            Additional limitations to include
            
        Returns
        -------
        dict
            Model card as dictionary
        """
        card = {
            'model_details': {
                'name': model_name,
                'type': model_type,
                'organ': organ,
                'generated_date': datetime.now().isoformat(),
                'version': '3.0.0'
            },
            'intended_use': self._get_intended_use(model_type, organ),
            'training_data': {
                'n_samples': training_info.get('n_samples', 'Unknown'),
                'n_events': training_info.get('n_events', 'Unknown'),
                'n_features': training_info.get('n_features', 'Unknown'),
                'epv': training_info.get('epv', 'Unknown'),
                'epv_adequate': training_info.get('epv', 0) >= 10 if isinstance(training_info.get('epv'), (int, float)) else False
            },
            'performance': {
                'test_auc': performance_metrics.get('auc', 'Unknown'),
                'test_auc_ci': performance_metrics.get('auc_ci', 'Unknown'),
                'brier_score': performance_metrics.get('brier', 'Unknown'),
                'calibration_slope': performance_metrics.get('calibration_slope', 'Unknown'),
                'cv_auc_mean': performance_metrics.get('cv_auc_mean', 'Unknown'),
                'cv_auc_std': performance_metrics.get('cv_auc_std', 'Unknown')
            },
            'limitations': self._get_limitations(model_type, training_info, limitations),
            'failure_modes': self._get_failure_modes(model_type, training_info),
            'calibration_status': self._get_calibration_status(performance_metrics),
            'safety_warnings': self._get_safety_warnings(model_type, training_info, performance_metrics)
        }
        
        # Add EXPLORATORY label for ML models
        if model_type == 'ml':
            card['model_details']['label'] = 'EXPLORATORY'
            card['model_details']['warning'] = (
                'This is an EXPLORATORY machine learning model. '
                'Do not use for clinical decision-making without external validation.'
            )
        
        return card
    
    def _get_intended_use(self, model_type: str, organ: str) -> str:
        """Get intended use description"""
        if model_type == 'traditional':
            return (
                f"Traditional NTCP model for {organ} toxicity prediction. "
                "Based on established radiobiological principles (LKB/RS models). "
                "Suitable for research and hypothesis generation."
            )
        else:  # ml
            return (
                f"EXPLORATORY machine learning model for {organ} toxicity prediction. "
                "For research purposes only. NOT for clinical decision-making without external validation."
            )
    
    def _get_limitations(
        self, 
        model_type: str, 
        training_info: Dict, 
        additional: Optional[List[str]]
    ) -> List[str]:
        """Get limitations list"""
        limitations = []
        
        # EPV limitations
        epv = training_info.get('epv', 0)
        if isinstance(epv, (int, float)) and epv < 10:
            limitations.append(
                f"Low Events Per Variable (EPV={epv:.2f}). "
                "Recommended EPV >= 10 for reliable predictions."
            )
        
        # Sample size limitations
        n_samples = training_info.get('n_samples', 0)
        if isinstance(n_samples, (int, float)) and n_samples < 50:
            limitations.append(
                f"Small sample size (n={n_samples}). "
                "Results may not generalize to other populations."
            )
        
        # ML-specific limitations
        if model_type == 'ml':
            limitations.append(
                "Machine learning models may overfit to training data. "
                "Performance on new data may be lower than reported."
            )
            limitations.append(
                "Model performance depends on feature distribution. "
                "May fail on out-of-distribution inputs."
            )
        
        # Add additional limitations
        if additional:
            limitations.extend(additional)
        
        return limitations
    
    def _get_failure_modes(self, model_type: str, training_info: Dict) -> List[str]:
        """Get failure modes"""
        failure_modes = []
        
        if model_type == 'ml':
            failure_modes.append(
                "Out-of-distribution inputs: Model may produce unreliable predictions "
                "for patients with feature values outside training distribution."
            )
            failure_modes.append(
                "Overfitting: Model may memorize training data patterns that don't generalize."
            )
        
        failure_modes.append(
            "Extreme dose values: Predictions may be unreliable for very high or very low doses."
        )
        
        return failure_modes
    
    def _get_calibration_status(self, performance_metrics: Dict) -> str:
        """Get calibration status"""
        cal_slope = performance_metrics.get('calibration_slope')
        
        if cal_slope is None or np.isnan(cal_slope):
            return "Unknown"
        elif abs(cal_slope - 1.0) < 0.1:
            return "Well-calibrated"
        elif cal_slope < 0.9:
            return "Underconfident (overpredicts)"
        elif cal_slope > 1.1:
            return "Overconfident (underpredicts)"
        else:
            return "Moderately calibrated"
    
    def _get_safety_warnings(
        self, 
        model_type: str, 
        training_info: Dict, 
        performance_metrics: Dict
    ) -> List[str]:
        """Get safety warnings"""
        warnings = []
        
        # EPV warning
        epv = training_info.get('epv', 0)
        if isinstance(epv, (int, float)) and epv < 10:
            warnings.append("LOW EPV: Model reliability may be compromised.")
        
        # Calibration warning
        cal_slope = performance_metrics.get('calibration_slope')
        if cal_slope is not None and not np.isnan(cal_slope):
            if cal_slope > 1.2:
                warnings.append("UNDERPREdICTION RISK: Model may systematically underpredict toxicity.")
            elif cal_slope < 0.8:
                warnings.append("OVERPREdICTION RISK: Model may systematically overpredict toxicity.")
        
        # CV stability warning
        cv_std = performance_metrics.get('cv_auc_std', 0)
        if isinstance(cv_std, (int, float)) and cv_std > 0.15:
            warnings.append("HIGH CV VARIANCE: Model performance is unstable across folds.")
        
        return warnings
    
    def save_model_card(
        self,
        card: Dict[str, Any],
        output_path: Path,
        format: str = 'json'
    ):
        """
        Save model card to file
        
        Parameters
        ----------
        card : dict
            Model card dictionary
        output_path : Path
            Output file path
        format : str
            Output format ('json' or 'txt')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(card, f, indent=2)
        elif format == 'txt':
            # Human-readable format
            lines = [
                "=" * 60,
                f"MODEL CARD: {card['model_details']['name']}",
                "=" * 60,
                f"Organ: {card['model_details']['organ']}",
                f"Type: {card['model_details']['type']}",
                f"Generated: {card['model_details']['generated_date']}",
                ""
            ]
            
            if 'label' in card['model_details']:
                lines.append(f"⚠️  LABEL: {card['model_details']['label']}")
                lines.append(f"   {card['model_details']['warning']}")
                lines.append("")
            
            lines.extend([
                "INTENDED USE:",
                card['intended_use'],
                "",
                "TRAINING DATA:",
                f"  Samples: {card['training_data']['n_samples']}",
                f"  Events: {card['training_data']['n_events']}",
                f"  Features: {card['training_data']['n_features']}",
                f"  EPV: {card['training_data']['epv']}",
                f"  EPV Adequate: {card['training_data']['epv_adequate']}",
                "",
                "PERFORMANCE:",
                f"  Test AUC: {card['performance']['test_auc']}",
                f"  Test AUC CI: {card['performance']['test_auc_ci']}",
                f"  Brier Score: {card['performance']['brier_score']}",
                f"  Calibration Slope: {card['performance']['calibration_slope']}",
                f"  CV AUC: {card['performance']['cv_auc_mean']} ± {card['performance']['cv_auc_std']}",
                "",
                "LIMITATIONS:",
            ])
            
            for lim in card['limitations']:
                lines.append(f"  - {lim}")
            
            lines.extend([
                "",
                "FAILURE MODES:",
            ])
            
            for fm in card['failure_modes']:
                lines.append(f"  - {fm}")
            
            lines.extend([
                "",
                f"CALIBRATION STATUS: {card['calibration_status']}",
                ""
            ])
            
            if card['safety_warnings']:
                lines.append("SAFETY WARNINGS:")
                for warning in card['safety_warnings']:
                    lines.append(f"  ⚠️  {warning}")
            else:
                lines.append("SAFETY WARNINGS: None")
            
            lines.append("=" * 60)
            
            with open(output_path, 'w') as f:
                f.write("\n".join(lines))
