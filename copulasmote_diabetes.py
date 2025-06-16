import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from scipy.stats import rankdata
from scipy.optimize import brentq
from statsmodels.stats.contingency_tables import mcnemar

# ─── 1) Load and prepare data ──────────────────────────────────────────────────
df = pd.read_csv("C:/Users/User/Downloads/pima_diabetes_data.csv")
diabetes_data = df[df['Outcome'] == 1] \
                  .drop(columns='Outcome') \
                  .reset_index(drop=True)
feature_1, feature_2 = 'Glucose', 'BMI'

# ─── 2) ECDF helpers ───────────────────────────────────────────────────────────
def ecdf_transform(col):
    return rankdata(col, method='average') / (len(col) + 1)

def inverse_ecdf(orig, u):
    sorted_vals = np.sort(orig)
    idx = (u * len(sorted_vals)).astype(int).clip(0, len(sorted_vals)-1)
    return sorted_vals[idx]

# ─── 3) Copula generator ─────────────────────────────────────────────────────
def phi_A2(t, θ):
    return (((1/t)*(1-t)**2))**θ

def dphi_A2(t, θ):
    g = (1 - t)**2 / t
    num = -(1 - t)*(1 + t)
    den = t**2
    return θ * g**(θ-1) * (num/den)

def phi_A2_inv(y, θ):
    a = 2 + y**(1/θ)
    return (a - np.sqrt(a*a - 4)) / 2

# ─── 4) Sampling utilities ────────────────────────────────────────────────────
def K_fun(x, φ, dφ, θ):
    return x - φ(x,θ)/(dφ(x,θ) + 1e-15)

def K_inv(t, φ, dφ, θ):
    return brentq(lambda x: K_fun(x,φ,dφ,θ) - t, 1e-6, 1-1e-6, xtol=1e-6)

def sample_copula(φ, dφ, φ_inv, θ, n=232, seed=42):
    np.random.seed(seed)
    U = np.empty(n); V = np.empty(n)
    s = np.random.rand(n); t = np.random.rand(n)
    for i in range(n):
        w = K_inv(t[i], φ, dφ, θ)
        φw = φ(w,θ)
        if φw < 1e-15:
            U[i]=V[i]=1.0
        else:
            U[i] = φ_inv(s[i]*φw, θ)
            V[i] = φ_inv((1-s[i])*φw, θ)
    return U, V

def gen_from_copula(data, φ, dφ, φ_inv, θ=10.0, n=232):
    U, V = sample_copula(φ, dφ, φ_inv, θ, n)
    g1 = inverse_ecdf(data[feature_1], U)
    g2 = inverse_ecdf(data[feature_2], V)
    other = (data
        .drop(columns=[feature_1,feature_2])
        .sample(n, replace=True, random_state=0)
        .reset_index(drop=True))
    other[feature_1] = g1
    other[feature_2] = g2
    other['Outcome'] = 1
    return other

# ─── 5) Build augmented datasets ──────────────────────────────────────────────
gen_a2 = gen_from_copula(diabetes_data, phi_A2, dphi_A2, phi_A2_inv, θ=10.0)

c0 = df[df['Outcome'] == 0].sample(500, random_state=42)
c1 = df[df['Outcome'] == 1]

bal_a2 = pd.concat([c0, c1, gen_a2], ignore_index=True).sample(frac=1, random_state=42)

# ─── 6) Model & evaluation helper ────────────────────────────────────────────
models = {
    "RF": RandomForestClassifier(random_state=42),
    "GB": GradientBoostingClassifier(random_state=42),
    "XGB": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LR": LogisticRegression(max_iter=1000, random_state=42)
}

def eval_models(Xtr, ytr, Xte, yte):
    res = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        pr = m.predict_proba(Xte)[:,1] if hasattr(m, "predict_proba") else p
        res[name] = {
            "Acc":  accuracy_score(yte, p),
            "Prec": precision_score(yte, p),
            "Rec":  recall_score(yte, p),
            "F1":   f1_score(yte, p),
            "AUC":  roc_auc_score(yte, pr)
        }
    return pd.DataFrame(res).T

# ─── 8) Evaluate A2 ──────────────────────────────────────────────────────────
X2 = bal_a2.drop(columns="Outcome")
y2 = bal_a2["Outcome"]
X2_tr, X2_te, y2_tr, y2_te = train_test_split(
    X2, y2, stratify=y2, test_size=0.3, random_state=42
)
sc2 = StandardScaler().fit(X2_tr)
res2 = eval_models(sc2.transform(X2_tr), y2_tr, sc2.transform(X2_te), y2_te)

# ─── 9) Evaluate SMOTE ───────────────────────────────────────────────────────
X0 = df.drop(columns="Outcome")
y0 = df["Outcome"]
X0_tr, X0_te, y0_tr, y0_te = train_test_split(
    X0, y0, stratify=y0, test_size=0.3, random_state=42
)
sc0 = StandardScaler().fit(X0_tr)
X0_tr_s, X0_te_s = sc0.transform(X0_tr), sc0.transform(X0_te)
X0_rs, y0_rs = SMOTE(random_state=42).fit_resample(X0_tr_s, y0_tr)
res0 = eval_models(X0_rs, y0_rs, X0_te_s, y0_te)

# ─── 10) Consolidate & report ─────────────────────────────────────────────────
final = pd.concat({"A1": res1, "A2": res2, "SMOTE": res0},
                  names=["Method", "Model"])
print(final)

# ─── 11) RF Confusion Matrices ────────────────────────────────────────────────
rf_a2 = RandomForestClassifier(random_state=42)
rf_a2.fit(sc2.transform(X2_tr), y2_tr)
y2_rf = rf_a2.predict(sc2.transform(X2_te))
print("\nCM: RF on A2")
print(confusion_matrix(y2_te, y2_rf))

rf_sm = RandomForestClassifier(random_state=42)
rf_sm.fit(X0_rs, y0_rs)
y0_rf = rf_sm.predict(X0_te_s)
print("\nCM: RF on SMOTE")
print(confusion_matrix(y0_te, y0_rf))

# ─── 12) McNemar’s Test ─────────────────────────────────────────────
# 12.1 Scale raw SMOTE test set with A2’s scaler:
X0_te_a2s = sc2.transform(X0_te)

# 12.2 A2-RF predicts on that:
y2_on_sm = rf_a2.predict(X0_te_a2s)

# 12.3 Contingency table vs. SMOTE-RF’s predictions (y0_rf):
c00 = np.sum((y2_on_sm == y0_te) & (y0_rf == y0_te))
c01 = np.sum((y2_on_sm == y0_te) & (y0_rf != y0_te))
c10 = np.sum((y2_on_sm != y0_te) & (y0_rf == y0_te))
c11 = np.sum((y2_on_sm != y0_te) & (y0_rf != y0_te))
table = [[c00, c01], [c10, c11]]
print("\nCorrected Contingency Table:")
print(np.array(table))

# 12.4 McNemar’s test:
res = mcnemar(table, exact=True)
print(f"\nMcNemar’s χ² = {res.statistic:.3f}, p = {res.pvalue:.3f}")
