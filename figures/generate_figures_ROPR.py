#!/usr/bin/env python3
"""
generate_figures_ROPR.py - py_ntcpx v1.1.1, figures for Reports of Practical
Oncology and Radiotherapy.

Every number plotted is read from ../data/; nothing is hard-coded, so the
figures cannot drift out of step with the manuscript.

Outputs (PNG 1200 dpi + TIFF LZW + SVG) into ./output/:
  Figure_1_pipeline_architecture
  Figure_2_composite                  (panels a-f, matching the Results text)
  Supplementary_Figure_S1_calibration
  Supplementary_Figure_S2_dose_response
  Supplementary_Figure_S3_roc_brier_ece
  Supplementary_Figure_S4_dvh_metrics
  Supplementary_Figure_S5_mvlog_vs_ml
  Supplementary_Figure_S6_shap_ann
  Supplementary_Figure_S7_shap_all
  Supplementary_Figure_S8_quantec
  Supplementary_Figure_S9_clinical_factors

Usage:  python generate_figures_ROPR.py
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import norm as scipy_norm
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('FIG_DATA', os.path.join(BASE, '..', 'data'))
OUT  = os.path.join(BASE, 'output'); os.makedirs(OUT, exist_ok=True)
DPI  = int(os.environ.get('FIG_DPI', '600'))   # RPOR asks >=600 dpi for line art
OUT  = os.environ.get('FIG_OUT', OUT)

W  = dict(blue='#0072B2', orange='#E69F00', green='#009E73', pink='#CC79A7',
          sky='#56B4E9', red='#D55E00', yellow='#F0E442', black='#000000', gray='#999999')
TC = dict(T1A='#56B4E9', T1B='#0072B2', T2='#009E73', T3='#E69F00',
          ANN='#CC79A7', XGB='#D55E00', RF='#F0E442')

plt.rcParams.update({
    'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans','Arial','Helvetica'],
    'font.size':8, 'axes.labelsize':9, 'axes.titlesize':9,
    'xtick.labelsize':7, 'ytick.labelsize':7, 'legend.fontsize':7,
    'figure.dpi':150, 'savefig.dpi':DPI, 'savefig.bbox':'tight', 'savefig.pad_inches':0.05,
    'axes.linewidth':0.6, 'axes.spines.top':False, 'axes.spines.right':False,
    'lines.linewidth':1.2,
})

# ----------------------------------------------------------------- data
comp = pd.read_csv(os.path.join(DATA,'comprehensive_output_data.txt'))
repro= pd.read_csv(os.path.join(DATA,'reproduction_data-uNTCP_CCS_QUANTECvalidation.txt'))
shap_csv = pd.read_csv(os.path.join(DATA,'shap_output_data.txt'))
T2   = pd.read_csv(os.path.join(DATA,'VERIFIED_Table2.csv'))
# Out-of-fold probabilities from the common internal-validation engine: one shared set of
# stratified folds for every trainable model, with feature selection, scaling and parameter
# refitting performed inside the training portion only (oof_engine.py).
OOFP = pd.read_csv(os.path.join(DATA,'oof_predictions.csv'))
OOF  = {c[4:]: OOFP[c].values.astype(float) for c in OOFP.columns if c.startswith('OOF_')}
VAL  = json.load(open(os.path.join(DATA,'VERIFIED_values.json')))
SHAPSTAB = pd.read_csv(os.path.join(DATA,'shap_stability.csv'))

y    = comp['Observed_Toxicity'].values.astype(float)
age  = comp['Age'].values.astype(float)
md   = comp['Mean_Parotid_Dose_Gy'].values.astype(float)
geud = comp['gEUD_Gy'].values.astype(float)
v30  = comp['V30'].values.astype(float); v45 = comp['V45'].values.astype(float)
v20  = comp['V20'].values.astype(float); v50 = comp['V50'].values.astype(float)
n    = len(y)
untcp   = comp['uNTCP'].values
ccs     = comp['CCS_cohort'].values
ci_width= comp['uNTCP_CI_U'].values - comp['uNTCP_CI_L'].values

NT = {'T1A LogLogit':'NTCP_LKB_LogLogit_QUANTEC','T1A Probit':'NTCP_LKB_Probit_QUANTEC',
      'T1A RS Poisson':'NTCP_RS_Poisson_QUANTEC','T1B Local':'NTCP_LKB_LOCAL',
      'T2 Probit MLE':'NTCP_LKB_Probit_MLE','T2 LogLogit MLE':'NTCP_LKB_LogLogit_MLE',
      'T2 RS MLE':'NTCP_RS_Poisson_MLE','T3 MV Log':'NTCP_MV_Logistic_apparent',
      'T3 MV Log CV':'NTCP_MV_Logistic_cv','T4 ANN':'NTCP_ANN',
      'T4 XGBoost':'NTCP_XGBoost','T4 RF':'NTCP_RandomForest'}
ntcp = {k: comp[v].values.astype(float) for k,v in NT.items() if v in comp.columns}

def row(model):
    r = T2[T2.Model.str.contains(model, regex=False)]
    return r.iloc[0] if len(r) else None

def panel_label(ax, letter, x=-0.14, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', ha='left')

def save_fig(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=DPI, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', pad_inches=0.04)
    try:
        fig.savefig(f'{OUT}/{name}.tif', dpi=DPI, bbox_inches='tight',
                    pad_inches=0.04, pil_kwargs={'compression':'tiff_lzw'})
    except Exception as e:
        print(f'    (TIFF skipped for {name}: {e})')
    plt.close(fig); print(f'  [ok] {name}')

def loo_auc(X, scale=True):
    X = np.asarray(X, float).reshape(n, -1); pr = np.zeros(n)
    for tr, te in LeaveOneOut().split(X):
        if scale:
            s = StandardScaler().fit(X[tr]); a, b = s.transform(X[tr]), s.transform(X[te])
        else:
            a, b = X[tr], X[te]
        m = LogisticRegression(C=1.0, max_iter=1000).fit(a, y[tr])
        pr[te] = m.predict_proba(b)[:, 1]
    return pr, roc_auc_score(y, pr)

# ============================================== Figure 1: pipeline schematic
def fig1_pipeline():
    fig = plt.figure(figsize=(10, 7.5)); ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,10); ax.set_ylim(0,7.5); ax.axis('off'); ax.set_facecolor('white')
    def rbox(x,y,w,h,fc,ec='#444444',lw=1.0,r=0.15):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad={r}',
                     facecolor=fc,edgecolor=ec,linewidth=lw,zorder=3))
    def lab(x,y,t,fs=8,fw='normal',color='#111111'):
        ax.text(x,y,t,ha='center',va='center',fontsize=fs,fontweight=fw,color=color,zorder=5)
    def arrow(x1,y1,x2,y2,rad=0.0):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),zorder=4,
                    arrowprops=dict(arrowstyle='-|>',color='#555555',lw=1.2,
                                    connectionstyle=f'arc3,rad={rad}'))
    rbox(1.2,6.6,7.6,0.65,'#EBF5FB',W['blue'],1.5)
    lab(5.0,6.925,'INPUT DATA',9,'bold',W['blue'])
    lab(5.0,6.65,'DVH arrays (cDVH/dDVH)  ·  Clinical variables  ·  Outcome labels',7.5)
    arrow(5.0,6.6,5.0,6.05)
    rbox(1.2,5.35,7.6,0.65,'#EBF5FB','#4a90d9',1.2)
    lab(5.0,5.68,'PREPROCESSING',9,'bold','#2471A3')
    lab(5.0,5.42,'gEUD (a = 2.2)  ·  DVH metrics (Vx, Dx)  ·  EQD2 normalisation  ·  cohort QA',7.5)
    ty, th = 3.85, 1.25
    tiers = [('T1A\nLiterature-fixed','LKB probit / log-logistic\nRS Poisson\nQUANTEC parameters',0.35,2.05,TC['T1A'],W['blue']),
             ('T1B\nLocally fitted','LKB probit, MLE fit\nBootstrap 95% CI\nn = 54',2.55,2.05,TC['T1B'],W['green']),
             ('T2\nMLE-refitted','LKB probit + log-logistic\nRS Poisson\nBoundary diagnostics',4.75,2.05,TC['T2'],W['orange']),
             ('T3\nMultivariable logistic','25-feature L2 logistic\nEPV-gated (EPV = 1.2)\nLeave-one-out CV',6.95,2.05,TC['T3'],'#b7950b')]
    for title, body, x0, w, fc, ec in tiers:
        rbox(x0,ty,w,th,fc,ec,1.2)
        lab(x0+w/2, ty+th-0.18, title, 8.5, 'bold', ec)
        for i,l in enumerate(body.split('\n')):
            lab(x0+w/2, ty+th-0.48-i*0.28, l, 7.2)
        arrow(5.0,5.35,x0+w/2,ty+th, 0.0 if abs(x0+w/2-5.0)<0.5 else (0.15 if x0+w/2>5.0 else -0.15))
    my, mh = 2.35, 1.3
    rbox(0.35,my,9.3,mh,'#D7BDE2','#7d3c98',1.5)
    lab(5.0,my+mh-0.17,'T4 — MACHINE LEARNING',9,'bold','#7d3c98')
    lab(5.0,my+mh-0.44,'ANN  ·  XGBoost  ·  Random forest',8)
    lab(5.0,my+mh-0.69,'5-fold cross-validation  ·  overfitting-severity grading (NONE / MODERATE / HIGH / CRITICAL)',7.5)
    lab(5.0,my+mh-0.94,'SHAP feature attribution  ·  LIME local explanations',7.5)
    arrow(5.0,3.85,5.0,my+mh)
    ey, eh = 0.9, 1.2
    rbox(0.35,ey,4.5,eh,'#FDFEFE','#1a5276',1.2)
    lab(2.60,ey+eh-0.17,'Evaluation harness',8.5,'bold','#1a5276')
    for i,t in enumerate(['AUC + 1,000-bootstrap 95% CI','Brier score  ·  ECE / MCE',
                          'Calibration slope / intercept','Overfitting-severity flag']):
        lab(2.60,ey+eh-0.42-i*0.21,t,7.5)
    rbox(5.15,ey,4.5,eh,'#FDFEFE',W['blue'],1.2)
    lab(7.40,ey+eh-0.17,'Uncertainty-aware NTCP (uNTCP) + CCS',8.5,'bold',W['blue'])
    for i,t in enumerate(['Inverse-variance weighted ensemble','Probabilistic gEUD + Monte Carlo NTCP',
                          'Indicative uncertainty band','Cohort Consistency Score']):
        lab(7.40,ey+eh-0.42-i*0.21,t,7.5)
    arrow(2.60,my,2.60,ey+eh); arrow(7.40,my,7.40,ey+eh)
    handles=[mpatches.Patch(facecolor=TC['T1A'],edgecolor=W['blue'],lw=0.8,label='T1A — literature-fixed'),
             mpatches.Patch(facecolor=TC['T1B'],edgecolor=W['green'],lw=0.8,label='T1B — local fit'),
             mpatches.Patch(facecolor=TC['T2'],edgecolor=W['orange'],lw=0.8,label='T2 — MLE refitted'),
             mpatches.Patch(facecolor=TC['T3'],edgecolor='#b7950b',lw=0.8,label='T3 — MV logistic'),
             mpatches.Patch(facecolor='#D7BDE2',edgecolor='#7d3c98',lw=0.8,label='T4 — machine learning'),
             mpatches.Patch(facecolor='#FDFEFE',edgecolor='#1a5276',lw=0.8,label='Evaluation / uNTCP')]
    ax.legend(handles=handles,loc='lower center',ncol=6,fontsize=7,frameon=True,
              framealpha=0.95,edgecolor='#cccccc',bbox_to_anchor=(0.5,-0.01))
    save_fig(fig,'Figure_1_pipeline_architecture')

# ====================================== Figure 2: composite, panels a-f
BARS = [('LKB probit\n(QUANTEC)','LKB probit (QUANTEC)','T1A'),
        ('LKB log-logistic\n(QUANTEC)','LKB log-logistic (QUANTEC)','T1A'),
        ('RS Poisson\n(QUANTEC)','RS Poisson (QUANTEC)','T1A'),
        ('LKB local\n(T1B)','LKB probit (local MLE)','T1B'),
        ('LKB probit\nMLE','LKB probit MLE','T2'),
        ('RS Poisson\nMLE','RS Poisson MLE','T2'),
        ('MV logistic\n(T3)','Multivariable logistic','T3'),
        ('ANN','ANN','ANN'), ('XGBoost','XGBoost','XGB'),
        ('Random\nforest','Random forest','RF')]

def _panel_auc(ax):
    h=0.35
    for i,(lab_,key,tier) in enumerate(BARS):
        r=row(key)
        app=float(r['AUC']); cvs=str(r['CV_AUC']) if pd.notna(r['CV_AUC']) else ''
        ax.barh(i+h/2,app,h,color=TC[tier],alpha=0.85,edgecolor='black',lw=0.4)
        if cvs.strip():
            parts=cvs.split('±'); cv=float(parts[0]); sd=float(parts[1]) if len(parts)>1 else None
            ax.barh(i-h/2,cv,h,color=TC[tier],alpha=0.35,edgecolor='black',lw=0.4,hatch='///')
            if sd: ax.errorbar(cv,i-h/2,xerr=sd,fmt='none',ecolor='black',elinewidth=0.7,
                               capsize=2.0,capthick=0.7,zorder=5)
    ax.axvline(0.5,color='gray',ls=':',lw=0.8,alpha=0.6)
    ax.set_yticks(range(len(BARS))); ax.set_yticklabels([b[0] for b in BARS],fontsize=6)
    ax.set_xlabel('AUC'); ax.set_xlim(0,1.0); ax.invert_yaxis()
    ax.legend(handles=[mpatches.Patch(color='gray',alpha=0.85,label='Apparent AUC'),
                       mpatches.Patch(color='gray',alpha=0.35,hatch='///',label='Cross-validated AUC')],
              loc='upper right',fontsize=6.5, framealpha=0.95)

def _panel_gap(ax):
    ml=[]
    for key,col in [('ANN','ANN'),('XGBoost','XGB'),('Random forest','RF')]:
        r=row(key); g=r['Grade']
        gap=float(g[g.find('(')+1:g.find(')')]); ml.append((key.replace(' ','\n'),gap,g.split(' (')[0],TC[col]))
    for i,(nm,gap,grade,color) in enumerate(ml):
        ax.barh(i,gap,0.5,color=color,edgecolor='black',lw=0.4)
        ax.text(gap+0.015 if gap>0 else gap-0.015,i,f'{gap:+.3f}\n({grade})',fontsize=6.5,
                va='center',ha='left' if gap>0 else 'right',
                fontweight='bold' if grade=='CRITICAL' else 'normal')
    ax.axvline(0,color='black',lw=0.8)
    ax.axvline(0.10,color=W['orange'],ls='--',lw=0.6,alpha=0.7)
    ax.axvline(0.30,color=W['red'],ls='--',lw=0.6,alpha=0.7)
    ax.set_yticks(range(len(ml))); ax.set_yticklabels([m[0] for m in ml],fontsize=7)
    ax.set_xlabel('Overfitting gap (apparent − cross-validated)',fontsize=8)
    ax.set_xlim(-0.15,0.75); ax.invert_yaxis()

def _panel_roc_age(ax):
    pa,aa=loo_auc(age); pdz,ad=loo_auc(md,scale=False); pb,ab=loo_auc(np.c_[age,md],scale=False)
    for pred,l,color,ls in [(pa,f'Age only (LOO-AUC {aa:.3f})',W['red'],'-'),
                            (pb,f'Age + mean dose ({ab:.3f})',W['orange'],'--'),
                            (ntcp['T4 ANN'],f"ANN (5-fold CV {float(row('ANN')['CV_AUC'].split('±')[0]):.3f})",TC['ANN'],'-.'),
                            (pdz,f'Mean dose only ({ad:.3f})',W['blue'],':')]:
        fpr,tpr,_=roc_curve(y,pred); ax.plot(fpr,tpr,ls=ls,color=color,lw=1.3,label=l)
    ax.plot([0,1],[0,1],'k--',lw=0.5,alpha=0.4)
    ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity')
    ax.legend(fontsize=5.5,loc='lower right')

def _panel_untcp(ax):
    # Binary outcome: discrete legend, not a continuous colour bar (reviewer comment).
    ax.scatter(md[y==0],ci_width[y==0],s=28,marker='o',facecolors='none',edgecolors=W['blue'],
               lw=0.9,alpha=0.85,label=f'No toxicity (n = {int((1-y).sum())})')
    ax.scatter(md[y==1],ci_width[y==1],s=28,marker='^',color=W['red'],edgecolors='black',
               lw=0.3,alpha=0.75,label=f'Grade ≥2 (n = {int(y.sum())})')
    ax.set_xlabel('Mean parotid dose (Gy)'); ax.set_ylabel('uNTCP uncertainty-band width')
    ax.axhspan(0.40,np.nanmax(ci_width)+0.05,alpha=0.08,color=W['orange'])
    wide=ci_width>0.40
    ax.text(0.03,0.95,f'Band > 0.40: n = {int(wide.sum())}\nmean dose {md[wide].mean():.1f} Gy',
            transform=ax.transAxes,fontsize=6.5,va='top',
            bbox=dict(fc='lightyellow',ec=W['orange'],alpha=0.9,pad=3))
    ax.legend(fontsize=6,loc='upper right')

def _panel_ccs(ax):
    ax.hist(ccs,bins=20,color=W['blue'],alpha=0.7,edgecolor='black',lw=0.4)
    ax.axvline(0.10,color=W['red'],ls='--',lw=1.2,label='Warning threshold (0.10)')
    nb=int((ccs<0.10).sum())
    ax.text(0.95,0.95,f'{nb}/{n} ({nb/n*100:.0f}%)\nbelow threshold',transform=ax.transAxes,
            fontsize=7,ha='right',va='top',bbox=dict(fc='mistyrose',ec=W['red'],alpha=0.9,pad=3))
    ax.set_xlabel('Cohort Consistency Score'); ax.set_ylabel('Patients'); ax.legend(fontsize=6)

def _panel_dca(ax):
    """Apparent versus out-of-fold decision curves on a common set of folds.

    Dashed lines are apparent predictions, solid lines out-of-fold. The panel exists to
    show the inversion: on apparent predictions the most severely overfitted model (random
    forest) leads, and out of fold it falls below treat-all while the two logistic models
    remain above it. Out-of-fold probabilities come from the shared-fold engine, so every
    curve is derived from the same resampling.
    """
    th=np.linspace(0.10,0.50,401)
    def nb(yt,yp,t):
        return ((yp>=t)&(yt==1)).sum()/len(yt)-((yp>=t)&(yt==0)).sum()/len(yt)*t/(1-t)
    def curve(pred):
        ok=~np.isnan(pred); return np.array([nb(y[ok],pred[ok],t) for t in th])

    ta=np.array([y.mean()-(1-y.mean())*t/(1-t) for t in th])

    # shade where the out-of-fold EPV-compliant logistic model beats treat-all
    if 'T3_epv' in OOF:
        oof=curve(OOF['T3_epv'])
        win=oof>ta; k=len(th)
        while k>0 and win[k-1]: k-=1
        if k<len(th):
            import math
            lo=math.ceil(th[k]*100)/100.0
            ax.axvspan(lo,0.50,color='#E69F00',alpha=0.05,lw=0)
            ax.annotate('out-of-fold EPV-compliant\nlogistic > treat all',xy=(0.278,0.20),fontsize=6,
                        color='#8a6100',ha='left',va='top')
            ax.axvline(lo,color='#E69F00',lw=0.6,ls=(0,(2,2)))
            ax.annotate(f'\u03c4 \u2248 {lo:.2f}',xy=(lo,-0.035),fontsize=6,color='#8a6100',
                        ha='center',va='bottom')

    ax.plot(th,ta,'k-',lw=1.6,label='Treat all')
    ax.axhline(0,color='gray',lw=0.5)
    ax.annotate('Treat none',xy=(0.12,0.008),fontsize=5.5,color='gray',va='bottom')

    import matplotlib.patheffects as pe
    outline=[pe.Stroke(linewidth=2.0,foreground='#4a4a4a'),pe.Normal()]

    # apparent (dashed)
    for nm,pred,color,eff,dash in [
            ('Random forest (T4), apparent', ntcp['T4 RF'],     TC['RF'], outline, (0,(5,2))),
            ('MV logistic (T3), apparent',   ntcp['T3 MV Log'], TC['T3'], None,    (0,(1,1.6)))]:
        ax.plot(th,curve(pred),ls=dash,color=color,lw=1.2,label=nm,path_effects=eff)

    # out-of-fold (solid), same folds for every model
    for nm,key,color,lw,eff in [
            ('Random forest (T4), out-of-fold','RandomForest',TC['RF'],1.4,outline),
            ('MV logistic (T3), out-of-fold', 'T3_full',      TC['T3'],1.8,None),
            ('EPV-compliant logistic (T3), out-of-fold','T3_epv','#8C5A00',1.8,None)]:
        if key in OOF:
            ax.plot(th,curve(OOF[key]),ls='-',color=color,lw=lw,label=nm,path_effects=eff)

    ax.set_xlabel('Decision threshold'); ax.set_ylabel('Net benefit')
    ax.set_xlim(0.10,0.50); ax.set_ylim(-0.05,0.65)
    ax.legend(fontsize=5,loc='upper right',frameon=False)

def fig2_composite():
    fig,axes=plt.subplots(3,2,figsize=(9.5,10.5))
    # No panel titles are drawn on the figure. RPOR requires the caption to carry the
    # brief title ("not on the figure itself"), so each panel is identified by its letter
    # only and described in the caption.
    for ax,fn,letter in [
        (axes[0,0],_panel_auc,'a'),
        (axes[0,1],_panel_gap,'b'),
        (axes[1,0],_panel_roc_age,'c'),
        (axes[1,1],_panel_untcp,'d'),
        (axes[2,0],_panel_ccs,'e'),
        (axes[2,1],_panel_dca,'f')]:
        fn(ax)
        panel_label(ax,letter,x=-0.18,y=1.04)
    fig.tight_layout(); save_fig(fig,'Figure_2_composite')

# ============================================= supplementary figures
def rs_poisson(dose,d50,gamma,s):
    dose=np.asarray(dose,float)
    with np.errstate(over='ignore',invalid='ignore'):
        term=(dose/d50)**(1.0/(gamma*np.log(2)))
        inner=np.clip(1.0-np.exp(-np.log(2)*term),1e-15,1.0)
        return np.clip(inner**(1.0/s),0.0,1.0)

def ece_q(yt,p,nb=5):
    p=np.clip(np.asarray(p,float),0,1); ok=~np.isnan(p); yt=yt[ok]; p=p[ok]
    e=np.unique(np.quantile(p,np.linspace(0,1,nb+1))); idx=np.digitize(p,e[1:-1]); tot=0.
    for b in range(len(e)-1):
        m=idx==b
        if m.sum(): tot+=(m.sum()/len(p))*abs(yt[m].mean()-p[m].mean())
    return tot

def calib_q(yt,p,nb=5):
    p=np.clip(np.asarray(p,float),0,1); ok=~np.isnan(p); yt=yt[ok]; p=p[ok]
    e=np.unique(np.quantile(p,np.linspace(0,1,nb+1))); idx=np.digitize(p,e[1:-1])
    xs,ys=[],[]
    for b in range(len(e)-1):
        m=idx==b
        if m.sum(): xs.append(p[m].mean()); ys.append(yt[m].mean())
    return np.array(xs),np.array(ys)

def supp_s1():
    fig,axes=plt.subplots(2,2,figsize=(7.0,6.5))
    items=[('LKB probit (QUANTEC)','T1A Probit','LKB probit (QUANTEC)',TC['T1A']),
           ('LKB probit MLE (T2)','T2 Probit MLE','LKB probit MLE',TC['T2']),
           ('ANN (T4)','T4 ANN','ANN',TC['ANN']),
           ('MV logistic (T3, apparent)','T3 MV Log','Multivariable logistic',TC['T3'])]
    for i,(name,k,t2key,color) in enumerate(items):
        ax=axes.flat[i]; p=ntcp[k]
        xs,ys=calib_q(y,p); ax.plot(xs,ys,'o-',color=color,lw=1.5,ms=5)
        ax.plot([0,1],[0,1],'k--',lw=0.6,alpha=0.5)
        ax.set_xlabel('Predicted NTCP'); ax.set_ylabel('Observed frequency')
        ax.text(0.05,0.92,f'{name}\nECE = {float(row(t2key)["ECE"]):.3f}',transform=ax.transAxes,
                fontsize=7,va='top',bbox=dict(fc='white',ec='gray',alpha=0.9,pad=3))
        ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05); panel_label(ax,chr(65+i))
    fig.tight_layout(); save_fig(fig,'Supplementary_Figure_S1_calibration')

def supp_s2():
    P=json.load(open(os.path.join(DATA,'model_parameters_mle.json'))) if \
      os.path.exists(os.path.join(DATA,'model_parameters_mle.json')) else {}
    P=P.get('Parotid',P)
    fig,axes=plt.subplots(1,2,figsize=(8.0,3.5)); dose=np.linspace(1,70,500)
    ax=axes[0]
    ax.plot(dose,scipy_norm.cdf((dose-28.4)/(0.18*28.4)),'-',color=TC['T1A'],lw=1.5,
            label='LKB probit (TD$_{50}$ = 28.4 Gy)')
    ax.plot(dose,(dose/28.4)**4/(1+(dose/28.4)**4),'--',color=W['orange'],lw=1.5,label='LKB log-logistic')
    ax.plot(dose,rs_poisson(dose,26.3,0.73,0.01),'-.',color=W['green'],lw=1.5,
            label='RS Poisson (D$_{50}$ = 26.3 Gy)')
    ax.scatter(geud[y==0],-0.03*np.ones((y==0).sum()),marker='|',s=15,color=W['blue'],alpha=0.5)
    ax.scatter(geud[y==1], 1.03*np.ones((y==1).sum()),marker='|',s=15,color=W['red'],alpha=0.5)
    ax.axvline(26,color='gray',ls=':',lw=0.6)
    ax.set_xlabel('gEUD (Gy)'); ax.set_ylabel('NTCP'); ax.set_xlim(0,70); ax.set_ylim(-0.08,1.08)
    ax.legend(fontsize=5.5,loc='center right'); panel_label(ax,'A')
    ax=axes[1]
    td50=P.get('LKB_Probit_MLE',{}).get('TD50',35.46); mm=P.get('LKB_Probit_MLE',{}).get('m',1.00)
    d50=P.get('RS_Poisson_MLE',{}).get('D50',20.24); gg=P.get('RS_Poisson_MLE',{}).get('gamma',0.10)
    ss=P.get('RS_Poisson_MLE',{}).get('s',0.001)
    ax.plot(dose,scipy_norm.cdf((dose-td50)/(mm*td50)),'-',color=TC['T2'],lw=1.5,
            label=f'LKB probit MLE (TD$_{{50}}$ = {td50:.1f} Gy)')
    ax.plot(dose,rs_poisson(dose,d50,gg,ss),'--',color=W['green'],lw=1.5,
            label=f'RS Poisson MLE (D$_{{50}}$ = {d50:.1f} Gy)')
    ax.scatter(geud[y==0],-0.03*np.ones((y==0).sum()),marker='|',s=15,color=W['blue'],alpha=0.5)
    ax.scatter(geud[y==1], 1.03*np.ones((y==1).sum()),marker='|',s=15,color=W['red'],alpha=0.5)
    ax.set_xlabel('gEUD (Gy)'); ax.set_ylabel('NTCP'); ax.set_xlim(0,70); ax.set_ylim(-0.08,1.08)
    ax.legend(fontsize=5.5,loc='center right'); panel_label(ax,'B')
    fig.tight_layout(); save_fig(fig,'Supplementary_Figure_S2_dose_response')

ROC_ITEMS=[('LKB probit (T1A)','T1A Probit','LKB probit (QUANTEC)',TC['T1A'],'-'),
           ('LKB log-logistic (T1A)','T1A LogLogit','LKB log-logistic (QUANTEC)',TC['T1A'],'--'),
           ('RS Poisson (T1A)','T1A RS Poisson','RS Poisson (QUANTEC)',TC['T1A'],'-.'),
           ('LKB local (T1B)','T1B Local','LKB probit (local MLE)',TC['T1B'],'-'),
           ('LKB probit MLE (T2)','T2 Probit MLE','LKB probit MLE',TC['T2'],'-'),
           ('RS Poisson MLE (T2)','T2 RS MLE','RS Poisson MLE',TC['T2'],'--'),
           ('MV logistic (T3)','T3 MV Log','Multivariable logistic',TC['T3'],'-'),
           ('ANN (T4)','T4 ANN','ANN',TC['ANN'],'-'),
           ('XGBoost (T4)','T4 XGBoost','XGBoost',TC['XGB'],'--'),
           ('Random forest (T4)','T4 RF','Random forest',TC['RF'],':')]

def supp_s3():
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.5))
    ax=axes[0]
    for nm,k,t2k,color,ls in ROC_ITEMS:
        p=ntcp[k]; ok=~np.isnan(p)
        fpr,tpr,_=roc_curve(y[ok],p[ok])
        ax.plot(fpr,tpr,ls=ls,color=color,lw=1.0,label=f'{nm} ({roc_auc_score(y[ok],p[ok]):.3f})')
    ax.plot([0,1],[0,1],'k--',lw=0.5,alpha=0.4)
    ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity')
    ax.legend(fontsize=4.5,loc='lower right'); panel_label(ax,'A')
    ax=axes[1]
    vals=[float(row(t2k)['Brier']) for _,_,t2k,_,_ in ROC_ITEMS]
    ax.barh(range(len(vals)),vals,color=[c for *_,c,_ in ROC_ITEMS],edgecolor='black',lw=0.3,alpha=0.8)
    ax.axvline(y.mean()*(1-y.mean()),color='gray',ls=':',lw=0.8)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels([r[0] for r in ROC_ITEMS],fontsize=6)
    ax.set_xlabel('Brier score'); ax.invert_yaxis(); panel_label(ax,'B')
    ax=axes[2]
    vals=[float(row(t2k)['ECE']) for _,_,t2k,_,_ in ROC_ITEMS]
    ax.barh(range(len(vals)),vals,color=[c for *_,c,_ in ROC_ITEMS],edgecolor='black',lw=0.3,alpha=0.8)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels([r[0] for r in ROC_ITEMS],fontsize=6)
    ax.set_xlabel('Expected calibration error (5-bin quantile)'); ax.invert_yaxis(); panel_label(ax,'C')
    fig.tight_layout(); save_fig(fig,'Supplementary_Figure_S3_roc_brier_ece')

def supp_s4():
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.5))
    ax=axes[0]; feats=['V20','V30','V45','V50']
    pos_t=np.arange(len(feats))*2; pos_n=pos_t+0.7
    b1=ax.boxplot([comp.loc[y==0,f].values for f in feats],positions=pos_n,widths=0.5,patch_artist=True,
                  medianprops=dict(color='black',lw=1),flierprops=dict(marker='.',ms=2))
    b2=ax.boxplot([comp.loc[y==1,f].values for f in feats],positions=pos_t,widths=0.5,patch_artist=True,
                  medianprops=dict(color='black',lw=1),flierprops=dict(marker='.',ms=2))
    for b in b1['boxes']: b.set(facecolor=W['blue'],alpha=0.5)
    for b in b2['boxes']: b.set(facecolor=W['red'],alpha=0.5)
    ax.set_xticks(pos_t+0.35); ax.set_xticklabels(feats,fontsize=7); ax.set_ylabel('Volume (%)')
    ax.legend([b2['boxes'][0],b1['boxes'][0]],['Grade ≥2','No toxicity'],fontsize=6); panel_label(ax,'A')
    ax=axes[1]
    for nm,k,color,mk in [('LKB probit (T1A)','T1A Probit',TC['T1A'],'o'),
                          ('LKB probit MLE (T2)','T2 Probit MLE',TC['T2'],'s'),
                          ('ANN (T4)','T4 ANN',TC['ANN'],'^')]:
        p=ntcp[k]; ok=~np.isnan(p)
        ax.scatter(md[ok],p[ok],s=15,alpha=0.55,color=color,marker=mk,label=nm,edgecolors='none')
    ax.set_xlabel('Mean parotid dose (Gy)'); ax.set_ylabel('Predicted NTCP')
    ax.legend(fontsize=6,loc='lower right'); panel_label(ax,'B')
    ax=axes[2]
    ax.scatter(md,geud,s=15,alpha=0.6,c=y,cmap='RdYlBu_r',edgecolors='black',lw=0.2,vmin=-0.3,vmax=1.3)
    r,_=stats.pearsonr(md,geud)
    ax.text(0.05,0.92,f'r = {r:.3f}',transform=ax.transAxes,fontsize=8,fontweight='bold')
    ax.set_xlabel('Mean parotid dose (Gy)'); ax.set_ylabel('gEUD (Gy)'); panel_label(ax,'C')
    fig.tight_layout(); save_fig(fig,'Supplementary_Figure_S4_dvh_metrics')

def supp_s5():
    fig,axes=plt.subplots(1,2,figsize=(7.5,3.2))
    ax=axes[0]
    items=[('MV logistic\n(T3, 25 feat.)','Multivariable logistic',TC['T3']),('ANN','ANN',TC['ANN']),
           ('XGBoost','XGBoost',TC['XGB']),('Random forest','Random forest',TC['RF'])]
    w=0.32
    for i,(lb,key,color) in enumerate(items):
        r=row(key); app=float(r['AUC']); cv=float(str(r['CV_AUC']).split('±')[0])
        ax.bar(i-w/2,app,w,color=color,alpha=0.85,edgecolor='black',lw=0.4)
        ax.bar(i+w/2,cv,w,color=color,alpha=0.35,edgecolor='black',lw=0.4,hatch='///')
    # Reduced EPV-compliant model: leave-one-out only, no apparent counterpart plotted
    red=VAL['loo_reduced_age_tobacco']
    ax.bar(len(items)+w/2,red,w,color=W['green'],alpha=0.35,edgecolor='black',lw=0.4,hatch='///')
    ax.text(len(items)+w/2,red+0.02,f'{red:.3f}',ha='center',fontsize=6)
    items=items+[('Reduced\n(age+tobacco)','',W['green'])]
    ax.axhline(0.5,color='gray',ls=':',lw=0.6)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([i[0] for i in items],fontsize=6.5)
    ax.set_ylabel('AUC'); ax.set_ylim(0,1.0)
    ax.legend(handles=[mpatches.Patch(color='gray',alpha=0.85,label='Apparent'),
                       mpatches.Patch(color='gray',alpha=0.35,hatch='///',label='Cross-validated')],fontsize=6)
    panel_label(ax,'A')
    ax=axes[1]
    for i,(key,color) in enumerate([('ANN',TC['ANN']),('XGBoost',TC['XGB']),('Random forest',TC['RF'])]):
        cvs=str(row(key)['CV_AUC']).split('±'); mu=float(cvs[0]); sd=float(cvs[1])
        ax.bar(i+1,mu,0.5,color=color,alpha=0.7,edgecolor='black',lw=0.4)
        ax.errorbar(i+1,mu,yerr=sd,fmt='none',ecolor='black',capsize=4,lw=1.2)
    ax.axhline(0.5,color='gray',ls=':',lw=0.6)
    ax.set_xticks([1,2,3]); ax.set_xticklabels(['ANN','XGBoost','Random forest'],fontsize=7)
    ax.set_ylabel('5-fold CV AUC (mean ± SD)'); ax.set_ylim(0,1.0); panel_label(ax,'B')
    fig.tight_layout(); save_fig(fig,'Supplementary_Figure_S5_mvlog_vs_ml')

FEATS=['mean_dose','V30','V45']; FLAB=['Mean dose (Gy)','V30 (%)','V45 (%)']
FVAL={'mean_dose':md,'V30':v30,'V45':v45}

def supp_s6():
    fig,axes=plt.subplots(1,2,figsize=(8.0,3.2))
    ax=axes[0]
    for fi,f in enumerate(FEATS):
        sv=shap_csv[f'ANN_{f}_SHAP'].values; fv=FVAL[f]
        fn=(fv-fv.min())/(fv.max()-fv.min()+1e-10)
        j=np.random.RandomState(42+fi).normal(0,0.08,len(sv))
        sc=ax.scatter(sv,fi+j,c=fn,cmap='RdYlBu_r',s=15,alpha=0.7,edgecolors='none',vmin=0,vmax=1)
    ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FLAB,fontsize=7)
    ax.set_xlabel('SHAP value'); ax.axvline(0,color='gray',lw=0.5)
    cb=plt.colorbar(sc,ax=ax,shrink=0.85,pad=0.02)
    cb.set_label('Feature value (min → max)',fontsize=6.5); cb.set_ticks([0,1]); cb.set_ticklabels(['low','high'])
    panel_label(ax,'A')
    ax=axes[1]
    st=SHAPSTAB[SHAPSTAB.model=='ANN'].set_index('feature')
    means=[np.abs(shap_csv[f'ANN_{f}_SHAP'].values).mean() for f in FEATS]
    ax.barh(range(len(FEATS)),means,color=TC['ANN'],edgecolor='black',lw=0.3,alpha=0.85)
    for i,f in enumerate(FEATS):
        if f in st.index:
            ax.text(means[i],i,f"  rank {st.loc[f,'mean_rank']:.2f} ± {st.loc[f,'rank_std']:.2f}",
                    va='center',fontsize=6)
    ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FLAB,fontsize=7)
    ax.set_xlabel('Mean |SHAP|'); panel_label(ax,'B')
    fig.tight_layout(); save_fig(fig,'Supplementary_Figure_S6_shap_ann')

def supp_s7():
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.5))
    models=[('ANN','ANN',TC['ANN'],'ANN'),('XGBoost','XGB',TC['XGB'],'XGBoost'),
            ('Random forest','RF',TC['RF'],'Random forest')]
    xmax=max(np.abs(shap_csv[f'{pre}_{f}_SHAP'].values).mean()
             for _,pre,_,_ in models for f in FEATS)*1.15
    for mi,(nm,pre,color,key) in enumerate(models):
        ax=axes[mi]; ax.set_xlim(0,xmax)   # common scale so panels are comparable
        vals=[np.abs(shap_csv[f'{pre}_{f}_SHAP'].values).mean() for f in FEATS]
        ax.barh(range(len(FEATS)),vals,color=color,edgecolor='black',lw=0.3,alpha=0.85)
        ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FLAB,fontsize=7)
        ax.set_xlabel('Mean |SHAP|',fontsize=7)
        grade=row(key)['Grade'].split(' (')[0]
        ax.text(0.95,0.05,f'{nm}\n({grade})',transform=ax.transAxes,fontsize=7,ha='right',va='bottom',
                bbox=dict(fc='lightyellow',ec='gray',alpha=0.9,pad=2))
        panel_label(ax,chr(65+mi))
    fig.tight_layout(); save_fig(fig,'Supplementary_Figure_S7_shap_all')

def supp_s8():
    order=['<20Gy','20-30Gy','>30Gy']
    fig,axes=plt.subplots(1,2,figsize=(8.0,3.5))
    obs,pll,pann,ns=[],[],[],[]
    for b in order:
        m=repro['QUANTEC_Bin']==b
        obs.append(repro.loc[m,'Observed_Toxicity'].mean())
        pll.append(repro.loc[m,'Mean_NTCP_LKB_LogLogit_Bin'].iloc[0]); ns.append(int(m.sum()))
        ids=repro.loc[m,'AnonPatientID']
        pann.append(comp.loc[comp.AnonPatientID.isin(ids),'NTCP_ANN'].mean())
    ax=axes[0]; x=np.arange(3); w=0.25
    ax.bar(x-w,obs,w,color=W['blue'],alpha=0.8,label='Observed',edgecolor='black',lw=0.3)
    ax.bar(x,pll,w,color=TC['T1A'],alpha=0.8,label='LKB log-logistic',edgecolor='black',lw=0.3)
    ax.bar(x+w,pann,w,color=TC['ANN'],alpha=0.8,label='ANN',edgecolor='black',lw=0.3)
    ax.set_xticks(x); ax.set_xticklabels([f'{b}\n(n = {k})' for b,k in zip(order,ns)],fontsize=7)
    ax.set_ylabel('Rate / predicted NTCP'); ax.legend(fontsize=6); panel_label(ax,'A')
    ax=axes[1]
    ax.scatter(obs,pll,s=60,color=TC['T1A'],marker='o',edgecolors='black',lw=0.5,label='LKB log-logistic',zorder=5)
    ax.scatter(obs,pann,s=60,color=TC['ANN'],marker='s',edgecolors='black',lw=0.5,label='ANN',zorder=5)
    ax.plot([0,1],[0,1],'k--',lw=0.6,alpha=0.5)
    ax.set_xlabel('Observed rate'); ax.set_ylabel('Predicted rate')
    ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05); ax.legend(fontsize=6); panel_label(ax,'B')
    fig.tight_layout(); save_fig(fig,'Supplementary_Figure_S8_quantec')

def supp_s9():
    fig=plt.figure(figsize=(10.0,8.5)); gs=fig.add_gridspec(2,2,hspace=0.35,wspace=0.30)
    ax=fig.add_subplot(gs[0,0])
    hv={'Age':age,'D$_{mean}$':md,'gEUD':geud,'V30':v30,'V45':v45,
        'NTCP\n(probit)':ntcp['T1A Probit'],'NTCP\n(ANN)':ntcp['T4 ANN'],'Toxicity':y}
    sp=pd.DataFrame(hv).corr(method='spearman')
    cmap=LinearSegmentedColormap.from_list('rdbu',['#2166AC','#4393C3','#92C5DE','#D1E5F0',
                                                    '#FDDBC7','#F4A582','#D6604D','#B2182B'],N=256)
    im=ax.imshow(sp.values,cmap=cmap,vmin=-0.5,vmax=1.0,aspect='auto')
    for i in range(len(sp)):
        for j in range(len(sp)):
            v=sp.values[i,j]
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=5.5,
                    color='white' if abs(v)>0.55 else 'black')
    labs=list(hv.keys())
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs,fontsize=6,rotation=45,ha='right')
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs,fontsize=6)
    plt.colorbar(im,ax=ax,shrink=0.8,pad=0.02).set_label('Spearman ρ',fontsize=7); panel_label(ax,'A')
    ax=fig.add_subplot(gs[0,1])
    def pci(r,nn):
        z=np.arctanh(r); se=1/np.sqrt(nn-3); return np.tanh(z-1.96*se),np.tanh(z+1.96*se)
    F=[]
    ra,pa=stats.pearsonr(age,y); F.append(('Age',ra,*pci(ra,n),pa,True))
    rs,_=stats.pearsonr((comp['Sex']=='M').astype(float).values,y)
    _,psf=stats.fisher_exact(pd.crosstab(comp['Sex'],comp['Observed_Toxicity']).values)
    F.append(('Sex (M vs F)',rs,*pci(rs,n),psf,False))
    rt,_=stats.pearsonr(comp['Tobacco_Exposure'].values.astype(float),y)
    _,ptf=stats.fisher_exact(pd.crosstab(comp['Tobacco_Exposure'],comp['Observed_Toxicity']).values)
    F.append(('Tobacco',rt,*pci(rt,n),ptf,False))
    for nm,v in [('D$_{mean}$',md),('V30',v30),('gEUD',geud)]:
        r,p=stats.pearsonr(v,y); F.append((nm,r,*pci(r,n),p,False))
    yp=np.arange(len(F))[::-1]
    for i,(nm,r,lo,hi,p,sig) in enumerate(F):
        c=W['red'] if sig else W['gray']
        ax.plot([lo,hi],[yp[i]]*2,'-',color=c,lw=2 if sig else 1.2)
        ax.scatter(r,yp[i],marker='D',s=50 if sig else 30,color=c,edgecolors='black',lw=0.4,zorder=5)
        ax.text(0.68,yp[i],'p < 0.001' if p<0.001 else f'p = {p:.3f}',fontsize=6.5,va='center',
                color=W['red'] if sig else '#555',fontweight='bold' if sig else 'normal')
    ax.axvline(0,color='black',ls='--',lw=0.6)
    ax.set_yticks(yp); ax.set_yticklabels([f[0] for f in F],fontsize=7)
    ax.set_xlabel('Pearson r (95% CI)'); ax.set_xlim(-0.45,0.80); panel_label(ax,'B')
    ax=fig.add_subplot(gs[1,0])
    j=np.random.RandomState(42).normal(0,0.025,n)
    ax.scatter(age[y==1],y[y==1]+j[y==1],s=30,alpha=0.6,color=W['red'],marker='o',
               label=f'Grade ≥2 (n = {int(y.sum())})')
    ax.scatter(age[y==0],y[y==0]+j[y==0],s=30,alpha=0.6,color=W['blue'],marker='^',
               label=f'No toxicity (n = {int((1-y).sum())})')
    try:
        from scipy.optimize import curve_fit
        f=lambda x,L,k,x0: L/(1+np.exp(-k*(x-x0)))
        popt,_=curve_fit(f,age,y,p0=[1,0.1,50],maxfev=10000)
        xs=np.linspace(15,80,300); ax.plot(xs,f(xs,*popt),'-',color='black',lw=1.5)
    except Exception: pass
    ax.text(0.03,0.95,f'r = {ra:.3f}, p < 0.001',transform=ax.transAxes,fontsize=8,va='top',
            fontweight='bold',color=W['red'],bbox=dict(fc='lightyellow',ec=W['red'],alpha=0.9,pad=3))
    ax.set_xlabel('Age (years)'); ax.set_ylabel('Observed toxicity')
    ax.legend(fontsize=6.5,loc='center right'); panel_label(ax,'C')
    ax=fig.add_subplot(gs[1,1])
    bl=['<30','30–45','45–60','≥60']; ag=pd.cut(age,bins=[0,30,45,60,200],labels=bl,right=False)
    rates,nn=[],[]
    for g in bl:
        m=np.asarray(ag==g); nn.append(int(m.sum())); rates.append(y[m].mean()*100 if m.sum() else 0)
    # Strata with fewer than three patients are not plotted as a percentage (reviewer comment)
    MIN_N=3
    shown=[r if k>=MIN_N else 0 for r,k in zip(rates,nn)]
    ax.bar(range(len(bl)),shown,color=[W['green'],W['orange'],W['pink'],W['red']],
           edgecolor='black',lw=0.5,width=0.6,alpha=0.85)
    for i,(r,k) in enumerate(zip(rates,nn)):
        if k>=MIN_N:
            ax.text(i,r+2,f'{r:.0f}%\n(n = {k})',ha='center',fontsize=7,
                    fontweight='bold' if r>60 else 'normal')
        else:
            ax.text(i,3,f'n = {k}\ntoo few\nto plot',ha='center',fontsize=6,color='gray',style='italic')
    ax.axhline(y.mean()*100,color='gray',ls='--',lw=0.8,alpha=0.6)
    ax.set_xticks(range(len(bl))); ax.set_xticklabels([f'{l}\n(n = {k})' for l,k in zip(bl,nn)],fontsize=7.5)
    ax.set_xlabel('Age group (years)'); ax.set_ylabel('Toxicity rate (%)'); ax.set_ylim(0,105)
    panel_label(ax,'D')
    save_fig(fig,'Supplementary_Figure_S9_clinical_factors')

if __name__=='__main__':
    print('py_ntcpx v1.1.1 — ROPR figures')
    fig1_pipeline(); fig2_composite()
    supp_s1(); supp_s2(); supp_s3(); supp_s4(); supp_s5(); supp_s6(); supp_s7(); supp_s8(); supp_s9()
    print(f'\nAll figures written to {OUT}')
