## CopulaSMOTE: A Copula-Based Oversampling Approach for Imbalanced Classification in Diabetes Prediction 🧬📊

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn%20%7C%20XGBoost%20%7C%20LogReg-orange)](https://scikit-learn.org/stable/)
[![Copulas](https://img.shields.io/badge/Dependency%20Modeling-Copulas-6f42c1)](https://en.wikipedia.org/wiki/Copula_(probability_theory))
[![SMOTE](https://img.shields.io/badge/Oversampling-SMOTE-ff69b4)](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
[![Dataset](https://img.shields.io/badge/Data-PIMA%20Diabetes-yellowgreen)](https://www.kaggle.com/datasets/gzdekzlkaya/pima-indians-diabetes-dataset)

This repository contains the full dataset and implementation of a machine learning pipeline that addresses class imbalance in diabetes classification by leveraging copula-based data generation. We generate synthetic samples using the A2 copula, which is designed to capture complex dependency structures, and compare our method's effectiveness against the widely used SMOTE technique.

Our classification framework uses multiple ML models (Random Forest, Gradient Boosting, XGBoost, Logistic Regression, MLP), and evaluates performance via accuracy, precision, recall, F1, AUC, confusion matrices, and McNemar’s test.


## Pima Indians Diabetes Dataset  

**Source:** [Kaggle – Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/gzdekzlkaya/pima-indians-diabetes-dataset/data)  
**License:** [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)  
**Note:** No modifications were made (except formatting for our experiments).


## 📦 Requirements
Install dependencies:

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn statsmodels
```

## 🚀 Getting Started
```bash
git clone https://github.com/agnivibes/copulasmote-diabetes-classification.git
cd copulasmote-diabetes-classification
python main.py
```

## 🔬 Research Paper
Aich, A., Murshed, Murshed, M., Hewage, S. and Mayeaux, A.(2025). CopulaSMOTE: A Copula-Based Oversampling Approach for Imbalanced Classification in Diabetes Prediction. [Manuscript under review]

## 📊 Citation
If you use this code or method in your own work, please cite:

@article{Aich2025CopulaSMOTE,
  title   = {CopulaSMOTE: A Copula-Based Oversampling Approach for Imbalanced Classification in Diabetes Prediction},
  author  = {Aich, Agnideep, Murshed, Monzur Md., Hewage, Sameera and Mayeaux, Amanda},
  year    = {2025},
  note    = {Manuscript under review}
}

## 📬 Contact
For questions or collaborations, feel free to contact:

Agnideep Aich,
Department of Mathematics, University of Louisiana at Lafayette
📧 agnideep.aich1@louisiana.edu

## 📝 License

This project is licensed under the [MIT License](LICENSE).

