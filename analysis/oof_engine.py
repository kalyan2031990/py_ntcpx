"""
py_ntcpx common internal-validation engine.

Produces out-of-fold (OOF) probabilities for every trainable model on ONE shared
set of stratified folds, with all data-dependent steps -- feature selection,
scaling and parameter refitting -- performed inside the training portion only.
Every downstream metric (AUC, Brier, ECE, calibration intercept/slope, decision
curves) is computed from those same OOF probabilities.

Addresses reviewer comments on:
  * apparent-only decision curves for Tier 4,
  * feature selection performed on the full cohort,
  * calibration reported only as a 5-bin ECE.
"""
import json, warnings
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

SEED = 42
DATA = '/home/claude/work/pkg/data'

# ---------------------------------------------------------------- data
d = pd.read_csv(f'{DATA}/results/enhanced_ntcp_calculations.csv')
cl = pd.read_excel(f'{DATA}/py_ntcpx_clinical_v1.1.1.xlsx')
cl = cl.rename(columns={'patient_id': 'AnonPatientID'})
d = d.merge(cl[['AnonPatientID', 'age', 'sex', 'tobacco_exposure']], on='AnonPatientID', how='left')
y = d['Observed_Toxicity'].values.astype(int)
n = len(y)

DOSI = ['mean_dose','max_dose','gEUD','total_volume',
        'V5','V10','V15','V20','V25','V30','V35','V40','V45','V50',
        'D1','D2','D5','D10','D20','D30','D50','D70','D90','D95']
d['age_over_50'] = (d['age'] >= 50).astype(int)
T3_FEATURES = DOSI + ['age_over_50']            # 25 candidate predictors
T3_EPV      = ['age_over_50', 'tobacco_exposure']

# ---------------------------------------------------------------- folds
skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
FOLDS = list(skf.split(np.zeros(n), y))
rskf  = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)

# ---------------------------------------------------------------- metrics
def ece_q(yt, p, nb=5):
    p = np.clip(np.asarray(p, float), 0, 1)
    e = np.unique(np.quantile(p, np.linspace(0, 1, nb + 1)))
    idx = np.digitize(p, e[1:-1]); tot = 0.
    for b in range(len(e) - 1):
        m = idx == b
        if m.sum(): tot += (m.sum() / len(p)) * abs(yt[m].mean() - p[m].mean())
    return tot

def calibration(yt, p):
    """Calibration intercept (calibration-in-the-large) and slope on the logit scale."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    slope = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000
                               ).fit(lp.reshape(-1, 1), yt).coef_[0][0]
    inter = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000
                               ).fit(np.zeros((len(yt), 1)), yt).intercept_[0] - np.mean(lp)
    # calibration-in-the-large: intercept of y ~ offset(lp)
    from statsmodels.api import GLM, families
    try:
        import statsmodels.api as sm
        m = sm.GLM(yt, np.ones((len(yt), 1)), family=sm.families.Binomial(),
                   offset=lp).fit()
        inter = float(m.params[0])
    except Exception:
        pass
    return inter, slope

def nb_curve(yt, p, th):
    p = np.asarray(p, float); ok = ~np.isnan(p); yt2, p2 = yt[ok], p[ok]
    return np.array([((p2 >= t) & (yt2 == 1)).sum() / len(yt2)
                     - ((p2 >= t) & (yt2 == 0)).sum() / len(yt2) * t / (1 - t) for t in th])

# ---------------------------------------------------------------- classical forms
def ll_ntcp(g, td50, g50):
    g = np.clip(np.asarray(g, float), 1e-9, None)
    return np.clip(1.0 / (1.0 + (td50 / g) ** (4.0 * g50)), 1e-9, 1 - 1e-9)

def probit_ntcp(g, td50, m):
    g = np.asarray(g, float)
    return np.clip(norm.cdf((g - td50) / (m * td50)), 1e-9, 1 - 1e-9)

def fit_classical(g, yy, form):
    """Bounded maximum-likelihood refit, bounds as documented in Supplementary Table S2."""
    if form == 'loglogistic':
        f, x0, bnds = ll_ntcp, [28.4, 1.0], [(15.0, 60.0), (0.10, 2.14)]
    else:
        f, x0, bnds = probit_ntcp, [28.4, 0.18], [(15.0, 60.0), (0.05, 1.00)]
    def nll(th):
        p = f(g, *th)
        return -np.sum(yy * np.log(p) + (1 - yy) * np.log(1 - p))
    r = minimize(nll, x0, bounds=bnds, method='L-BFGS-B')
    return r.x

# ---------------------------------------------------------------- Tier 4 / Tier 3
def ml_models(n_ev):
    """Hyperparameters exactly as adapt_for_small_dataset() sets them for n<50 events."""
    return {
        'ANN': lambda: MLPClassifier(hidden_layer_sizes=(8,), max_iter=200,
                                     alpha=0.1, random_state=SEED),
        'XGBoost': lambda: XGBClassifier(n_estimators=20, max_depth=2, learning_rate=0.1,
                                         random_state=SEED, eval_metric='logloss',
                                         verbosity=0),
        'RandomForest': lambda: RandomForestClassifier(n_estimators=100, max_depth=3,
                                                       random_state=SEED),
    }

def run_oof(folds):
    """One pass of OOF prediction for every trainable model on the given folds."""
    out = {k: np.full(n, np.nan) for k in
           ['ANN', 'XGBoost', 'RandomForest', 'T3_full', 'T3_epv',
            'T1B_ll', 'T1B_probit', 'T2_ll', 'T2_probit']}
    sel_counts = {}
    for tr, te in folds:
        ytr = y[tr]
        # ---- Tier 4: k=3 selection INSIDE the training fold (EPV 10.3)
        Xall = d[DOSI].values
        sel = SelectKBest(f_classif, k=3).fit(Xall[tr], ytr)
        picked = tuple(np.array(DOSI)[sel.get_support()])
        sel_counts[picked] = sel_counts.get(picked, 0) + 1
        Xs = sel.transform(Xall)
        sc = StandardScaler().fit(Xs[tr])
        for name, mk in ml_models(ytr.sum()).items():
            mdl = mk()
            if name == 'ANN':
                mdl.fit(sc.transform(Xs[tr]), ytr)
                out[name][te] = mdl.predict_proba(sc.transform(Xs[te]))[:, 1]
            else:
                mdl.fit(Xs[tr], ytr)
                out[name][te] = mdl.predict_proba(Xs[te])[:, 1]
        # ---- Tier 3: L2 logistic, scaling fitted on the training fold only
        for key, feats in [('T3_full', T3_FEATURES), ('T3_epv', T3_EPV)]:
            Xf = d[feats].values.astype(float)
            s2 = StandardScaler().fit(Xf[tr])
            lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                    max_iter=2000).fit(s2.transform(Xf[tr]), ytr)
            out[key][te] = lr.predict_proba(s2.transform(Xf[te]))[:, 1]
        # ---- Tier 1B / Tier 2: classical refits inside the training fold
        g = d['gEUD'].values
        for form, k1b, k2 in [('loglogistic', 'T1B_ll', 'T2_ll'),
                              ('probit', 'T1B_probit', 'T2_probit')]:
            th = fit_classical(g[tr], ytr, form)
            f = ll_ntcp if form == 'loglogistic' else probit_ntcp
            out[k1b][te] = f(g[te], *th)
            out[k2][te]  = f(g[te], *th)
    return out, sel_counts

OOF, SEL = run_oof(FOLDS)

# ---------------------------------------------------------------- apparent reference
APP = {
    'ANN': d['NTCP_ML_ANN'].values,
    'XGBoost': d['NTCP_ML_XGBoost'].values,
    'RandomForest': d['NTCP_ML_RandomForest'].values,
}

# ---------------------------------------------------------------- repeated CV for stability
def repeated_auc():
    res = {k: [] for k in ['ANN', 'XGBoost', 'RandomForest', 'T3_full', 'T3_epv']}
    reps = list(rskf.split(np.zeros(n), y))
    for r in range(20):
        folds = reps[r * 5:(r + 1) * 5]
        o, _ = run_oof(folds)
        for k in res:
            res[k].append(roc_auc_score(y, o[k]))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items()}

REP = repeated_auc()

# ---------------------------------------------------------------- report
TH = np.linspace(0.10, 0.50, 401)
ta = np.array([y.mean() - (1 - y.mean()) * t / (1 - t) for t in TH])
rows = []
for k, p in OOF.items():
    inter, slope = calibration(y, p)
    rows.append(dict(model=k, auc=roc_auc_score(y, p), brier=brier_score_loss(y, p),
                     ece=ece_q(y, p), calib_intercept=inter, calib_slope=slope,
                     nb50=nb_curve(y, p, np.array([0.50]))[0],
                     nb40=nb_curve(y, p, np.array([0.40]))[0]))
R = pd.DataFrame(rows)

app_rows = []
for k, p in APP.items():
    inter, slope = calibration(y, p)
    app_rows.append(dict(model=k, auc=roc_auc_score(y, p), brier=brier_score_loss(y, p),
                         ece=ece_q(y, p), calib_intercept=inter, calib_slope=slope,
                         nb50=nb_curve(y, p, np.array([0.50]))[0],
                         nb40=nb_curve(y, p, np.array([0.40]))[0]))
A = pd.DataFrame(app_rows)

print('=' * 78); print('OUT-OF-FOLD (shared folds, selection inside folds)'); print('=' * 78)
print(R.to_string(index=False, float_format=lambda x: f'{x:8.4f}'))
print(); print('=' * 78); print('APPARENT (deposited predictions)'); print('=' * 78)
print(A.to_string(index=False, float_format=lambda x: f'{x:8.4f}'))
print(); print('treat-all NB: 0.50 -> %.4f   0.40 -> %.4f' % (ta[-1], nb_curve(y, np.ones(n), np.array([0.40]))[0]))
print(); print('repeated 20x5 stratified CV AUC (mean +/- SD):')
for k, (m, s) in REP.items(): print('  %-12s %.3f +/- %.3f' % (k, m, s))
print(); print('Tier-4 feature sets selected per fold:')
for k, v in SEL.items(): print('  %-40s %d/5 folds' % (', '.join(k), v))

np.save('/home/claude/work/oof/oof_predictions.npy', OOF, allow_pickle=True)
pd.DataFrame({**{'AnonPatientID': d.AnonPatientID, 'Observed_Toxicity': y},
              **{f'OOF_{k}': v for k, v in OOF.items()},
              **{f'APP_{k}': v for k, v in APP.items()}}
             ).to_csv('/home/claude/work/oof/oof_predictions.csv', index=False)
json.dump({'repeated_cv_auc': REP,
           'selected_features': {', '.join(k): v for k, v in SEL.items()}},
          open('/home/claude/work/oof/oof_summary.json', 'w'), indent=2)
R.to_csv('/home/claude/work/oof/oof_metrics.csv', index=False)
A.to_csv('/home/claude/work/oof/apparent_metrics.csv', index=False)
print('\nwritten: oof_predictions.csv, oof_metrics.csv, apparent_metrics.csv, oof_summary.json')
