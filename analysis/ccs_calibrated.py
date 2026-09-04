"""Report the Cohort Consistency Score on a calibrated scale.

CCS = exp(-d^2/2) is a monotone transform of a squared Mahalanobis distance, so
under a multivariate normal reference d^2 follows a chi-square distribution on k
degrees of freedom. A fixed CCS cutoff therefore has no constant meaning: with the
six dosimetric features used here, roughly 60% of a perfectly typical cohort falls
below the 0.10 software flag. This script reports the chi-square tail probability
instead, and recomputes each distance against a reference that excludes the index
patient so that no patient contributes to the centroid and covariance it is judged
against.

Usage:  python analysis/ccs_calibrated.py <enhanced_ntcp_calculations.csv> [out.csv]
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import chi2

FEATURES = ['mean_dose', 'gEUD', 'V30', 'V50', 'D20', 'max_dose']


def mahalanobis_sq(X, leave_one_out=True):
    """Squared Mahalanobis distance per row; the reference excludes the row itself."""
    X = np.asarray(X, float)
    n = len(X)
    out = np.empty(n)
    for i in range(n):
        R = np.delete(X, i, axis=0) if leave_one_out else X
        mu = R.mean(0)
        Si = np.linalg.pinv(np.cov(R, rowvar=False))
        dv = X[i] - mu
        out[i] = dv @ Si @ dv
    return out


def main(src, dst=None):
    d = pd.read_csv(src)
    X = d[FEATURES].values.astype(float)
    k = X.shape[1]
    d2_loo = mahalanobis_sq(X, True)
    d2_in = mahalanobis_sq(X, False)
    tail = chi2.sf(d2_loo, k)
    res = pd.DataFrame({
        'AnonPatientID': d.get('AnonPatientID', pd.RangeIndex(len(d))),
        'CCS_in_sample': np.exp(-d2_in / 2),
        'CCS_leave_one_out': np.exp(-d2_loo / 2),
        'mahalanobis_d2_leave_one_out': d2_loo,
        'chi2_tail_p': tail,
    })
    n = len(d)
    expected = chi2.sf(-2 * np.log(0.10), k)
    print(f'features {FEATURES} (k = {k}), n = {n}')
    print(f'  mean d^2                              {d2_in.mean():.2f}  (expectation {k} under chi2_{k})')
    print(f'  below the 0.10 software flag          {(res.CCS_in_sample < 0.10).sum()}/{n}'
          f' ({100 * (res.CCS_in_sample < 0.10).mean():.0f}%)')
    print(f'  expected below 0.10 under chi2_{k}      {100 * expected:.0f}%'
          '   <- a fixed CCS cutoff is not a measure of atypicality')
    for q in (0.05, 0.01):
        print(f'  chi-square tail p < {q:.2f} (leave-one-out) {(tail < q).sum()}/{n}'
              f' ({100 * (tail < q).mean():.0f}%), against {100 * q:.0f}% expected')
    if dst:
        res.to_csv(dst, index=False)
        print(f'written {dst}')
    return res


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
