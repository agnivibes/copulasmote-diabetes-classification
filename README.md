## CopulaSMOTE: A Copula-Based Oversampling Approach for Imbalanced Classification in Diabetes Prediction 🧬📊

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn%20%7C%20XGBoost%20%7C%20LogReg-orange)](https://scikit-learn.org/stable/)
[![Copulas](https://img.shields.io/badge/Dependency%20Modeling-Vine%20Copulas-6f42c1)](https://en.wikipedia.org/wiki/Copula_(probability_theory))
[![SMOTE](https://img.shields.io/badge/Oversampling-SMOTE-ff69b4)](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

This repository contains the implementation of a machine learning pipeline that addresses class imbalance in diabetes classification by leveraging copula-based data generation. We fit a truncated vine copula to the minority class, simulate synthetic samples that preserve its joint dependence structure, and compare this approach against classical interpolation-based oversamplers (SMOTE, Borderline-SMOTE, ADASYN) and a normalizing-flow baseline.

The classification framework uses multiple ML models (Random Forest, Gradient Boosting, XGBoost, Logistic Regression, MLP) and evaluates performance via accuracy, balanced accuracy, precision, recall, F1, AUC, and PR-AUC, with statistical comparison through the Dietterich 5×2 cross-validation paired t-test.

## 📂 Datasets

The three diabetes datasets used in this study are the Pima Indians Diabetes dataset, the Iraqi Diabetes dataset, and the CDC Diabetes Health Indicators dataset derived from BRFSS. Full dataset descriptions, sources, access details, and preprocessing steps are provided in the manuscript. No datasets are redistributed in this repository.

## 📦 Requirements
Install dependencies:

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn scipy statsmodels matplotlib seaborn
pip install pyvinecopulib nflows torch
```

## 🚀 Getting Started
```bash
git clone https://github.com/agnivibes/copulasmote-diabetes-classification.git
cd copulasmote-diabetes-classification

# Run on a chosen dataset: PIMA, IRAQI, or CDC
python copulasmote_diabetes.py PIMA
```

Per-dataset results (metrics, statistical tests, and figures) are written to `./results_<dataset>/`.

## 🔬 Research Paper
Aich, A., Murshed, M. M., Wade, B., and Hewage, S. (2026). CopulaSMOTE: A Copula-Based Oversampling Approach for Imbalanced Classification in Diabetes Prediction. [Manuscript under review]

Corresponding author: Sameera Hewage, Department of Mathematics, Southern Utah University, Cedar City, UT 84720, USA.

## 📊 Citation
If you use this code or method in your own work, please cite:
```bibtex
@article{Aich2026CopulaSMOTE,
  title   = {CopulaSMOTE: A Copula-Based Oversampling Approach for Imbalanced Classification in Diabetes Prediction},
  author  = {Aich, Agnideep and Murshed, Md Monzur and Wade, Bruce and Hewage, Sameera},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## 📬 Contact
For questions or collaborations, feel free to contact:

<!-- TODO: replace placeholder email before publishing -->
Repository contact — Agnideep Aich, Department of Emergency Medicine, Stanford University 
📧 agnideep@stanford.edu

Corresponding author — Sameera Hewage, Department of Mathematics, Southern Utah University
📧 sameerahewage@suu.edu

## 📝 License

This project is licensed under the [MIT License](LICENSE).
