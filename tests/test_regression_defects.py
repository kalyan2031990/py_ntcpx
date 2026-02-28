"""Regression tests for defects fixed in py_ntcpx v1.1.2.

Each test pins a specific failure mode that previously changed a reported result
without raising anything. Run with:  python -m pytest tests/ -v
"""
import sys, math
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --------------------------------------------------------------- C17: zero-dose EQD2
def test_eqd2_of_zero_dose_is_zero_not_nan():
    """A zero-dose DVH bin is legitimate; EQD2(0) = 0.

    Returning NaN propagated through the relative-seriality sum and silently
    dropped the whole patient from the RS evaluation (n = 53 instead of 54).
    """
    from code3_ntcp_analysis_ml import NTCPCalculator as NTCPModels
    m = NTCPModels()
    val = m.convert_to_eqd2(0.0, 3.0, 2.0)
    assert val == 0.0, f"EQD2(0) must be 0, got {val!r}"
    assert not math.isnan(val)


def test_eqd2_negative_dose_is_nan():
    from code3_ntcp_analysis_ml import NTCPCalculator as NTCPModels
    m = NTCPModels()
    assert math.isnan(m.convert_to_eqd2(-1.0, 3.0, 2.0))


def test_rs_poisson_survives_a_zero_dose_bin():
    """The whole point of the C17 fix: one zero bin must not void the patient."""
    import pandas as pd
    from code3_ntcp_analysis_ml import NTCPCalculator as NTCPModels
    m = NTCPModels()
    dvh = pd.DataFrame({'dose_gy': [0.0, 10.0, 20.0, 30.0],
                        'volume_cm3': [10.0, 6.0, 3.0, 1.0]})
    dvh['dose_gy'] = dvh['dose_gy'].apply(lambda d: m.convert_to_eqd2(d, 3.0, 2.0))
    assert not dvh['dose_gy'].isna().any()
    ntcp = m.ntcp_rs_poisson(dvh, 26.3, 0.73, 0.01)
    assert 0.0 <= ntcp <= 1.0 and not math.isnan(ntcp)


# --------------------------------------------------- C15/C18: no silent feature drop
def test_tier3_refuses_to_run_without_clinical_covariates():
    """Omitting --clinical_file used to drop age_over_50 and move LOOCV AUC
    from 0.673 to 0.536 with no warning. It must now raise."""
    import pandas as pd
    from tiered_ntcp_analysis import (add_tier3_logistic_predictions,
                                      ClinicalCovariateError)
    df = pd.DataFrame({'Organ': ['Parotid'] * 25,
                       'Observed_Toxicity': [0, 1] * 12 + [1],
                       'mean_dose': np.linspace(20, 50, 25)})
    with pytest.raises(ClinicalCovariateError):
        add_tier3_logistic_predictions(df, None, '/tmp', include_age=True)


def test_tier3_dvh_only_model_is_opt_in_not_a_fallback():
    """Running without age must be an explicit choice, never a silent default."""
    import inspect
    from tiered_ntcp_analysis import add_tier3_logistic_predictions
    sig = inspect.signature(add_tier3_logistic_predictions)
    assert sig.parameters['include_age'].default is True


# ------------------------------------------------- C16: one overfitting grading rule
def test_overfitting_grade_has_a_single_authoritative_rule():
    from code4_ntcp_output_QA_reporter import flag_overfitting
    assert flag_overfitting(0.891, 0.300)[1] == 'CRITICAL'
    assert flag_overfitting(0.783, 0.381)[1] == 'CRITICAL'
    assert flag_overfitting(0.861, 0.673)[1] == 'HIGH'
    assert flag_overfitting(0.511, 0.550)[1] == 'NONE'
