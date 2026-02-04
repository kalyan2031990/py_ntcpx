# py_ntcpx v2.1.0 - Enhanced for Small Datasets

**Extended Release** - Builds upon v2.0.0 with enhanced small dataset support

---

## 🎯 What's New in v2.1.0

This release extends **v2.0.0** with enhanced capabilities for small datasets (n < 100) while maintaining all the methodological rigor, statistical validation, and explainable AI features from v2.0.0.

### Key Enhancements

- **Dynamic CCS Threshold**: Adaptive threshold calculation based on dataset size
- **Clinical Factor Integration**: Automatic inclusion of statistically significant clinical factors
- **Small Dataset Adaptations**: Adaptive CV strategies and model complexity reduction for n < 100
- **Robust SHAP Analysis**: Bootstrap SHAP for stability assessment
- **Enhanced Reporting**: Small dataset advisories and improved QA reports

---

## 📋 Complete Feature Set (v2.0.0 + v2.1.0)

### Data Integrity & Leakage Prevention
✅ Patient-level data splitting with stratification  
✅ LeakageAudit utility for automated leakage detection  
✅ StandardScaler fit only on training data  
✅ Split-before-transform enforcement  

### Overfitting Prevention & Model Containment
✅ EPV (Events Per Variable) enforcement  
✅ Auto feature reduction when EPV < 5  
✅ Conservative ML architectures (ANN, XGBoost)  
✅ Dynamic model complexity adjustment  
✅ Domain-guided feature selection  

### Statistical Rigor
✅ Correct Monte Carlo NTCP with parameter uncertainty  
✅ Bootstrap confidence intervals for all metrics  
✅ DeLong test for AUC comparison with Bonferroni correction  
✅ Nested cross-validation for unbiased performance estimation  
✅ Calibration correction (Platt scaling, isotonic regression)  

### Clinical Safety Layer
✅ ClinicalSafetyGuard with underprediction risk detection  
✅ Cohort Consistency Score (CCS) integration  
✅ DO_NOT_USE flags for unsafe predictions  
✅ Automated safety reports  

### Model Documentation
✅ Auto-generated model cards  
✅ EXPLORATORY labels for ML models  
✅ Intended use, limitations, and failure modes documented  

### Outputs
✅ 600 DPI figures  
✅ LaTeX tables for manuscript  
✅ Statistical reporting with confidence intervals  
✅ Comprehensive documentation  

### Reproducibility
✅ Global random seed management  
✅ YAML configuration management  
✅ Dependency locking  
✅ Baseline capture for regression testing  

---

## 📊 Test Results

- **Total Tests**: 80
- **Passed**: 78 (100% of runnable tests)
- **Failed**: 0
- **Skipped**: 2 (baseline regression - expected)

---

## 🔄 Migration

**No breaking changes** - all enhancements are backward compatible with v2.0.0.

For users upgrading from v1.x, see [CHANGELOG_v2.0.0.md](CHANGELOG_v2.0.0.md) for breaking changes.

---

## 📚 Documentation

- [README.md](README.md) - Quick start guide
- [CHANGELOG_v2.1.0.md](CHANGELOG_v2.1.0.md) - Detailed changelog
- [CHANGELOG_v2.0.0.md](CHANGELOG_v2.0.0.md) - v2.0.0 release notes
- [ARCHITECTURE_REPORT.md](ARCHITECTURE_REPORT.md) - Complete architecture documentation

---

## 🔗 Links

- **Repository**: https://github.com/kalyan2031990/py_ntcpx
- **Previous Release**: [v2.0.0](https://github.com/kalyan2031990/py_ntcpx/releases/tag/v2.0.0)

---

**Release Date**: February 3, 2026  
**Version**: 2.1.0  
**Based on**: v2.0.0

