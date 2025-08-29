import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Dict, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

from imblearn.over_sampling import SMOTE
from scipy.optimize import brentq
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.base import clone
from numpy.random import RandomState
from sklearn.metrics import precision_recall_curve, average_precision_score

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except Exception:
    HAS_TF = False

from sklearn.base import BaseEstimator, ClassifierMixin

# ──────────────────────────────────────────────────────────────────────────────
# User settings
# ──────────────────────────────────────────────────────────────────────────────

# <<< SET THIS TO YOUR DATA FILE >>>
DATA_CSV = r"C:\\Users\\User\\Downloads\\pima_diabetes_data.csv" # change me

# Where to save outputs
OUT_DIR = os.path.join(os.path.dirname(__file__) or ".", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Reproducibility
SEED = 42
rng = RandomState(SEED)

FEATURE_1, FEATURE_2 = "Glucose", "BMI"   # top-2 features selected in paper
THETAS = [2, 5, 10]                        # dependence settings for A2 plots/ROCs
N_SYNTH_PER_TRAIN = None                   # if None, auto-balance to 1:1 using train split counts


# ──────────────────────────────────────────────────────────────────────────────
# Math helpers: A2 copula generator, derivative, and inverse
# ──────────────────────────────────────────────────────────────────────────────

def phi_A2(t: np.ndarray, theta: float) -> np.ndarray:
    """
    A2 generator: φ(t) = [ ((1 - t)^2) / t ]^θ, defined for t in (0, 1].
    At t→1, φ(1) = 0. At t→0+, φ → ∞.
    """
    t = np.asarray(t, dtype=float)
    # Guard rails for numerical stability
    eps = 1e-12
    t = np.clip(t, eps, 1.0)  # avoid division by zero
    g = ((1.0 - t) ** 2) / t
    return np.power(g, theta)


def dphi_A2(t: np.ndarray, theta: float) -> np.ndarray:
    """
    Derivative φ'(t) for A2.
    Let g(t) = ((1 - t)^2)/t; then φ(t) = g(t)^θ.
    g'(t) = -(1 - t)*(1 + t)/t^2.
    So φ'(t) = θ * g(t)^(θ - 1) * g'(t).
    """
    t = np.asarray(t, dtype=float)
    eps = 1e-12
    t = np.clip(t, eps, 1.0)
    g = ((1.0 - t) ** 2) / t
    gprime = - (1.0 - t) * (1.0 + t) / (t ** 2)
    return theta * np.power(g, theta - 1.0) * gprime


def phi_A2_inv(y: np.ndarray, theta: float) -> np.ndarray:
    """
    Inverse generator for A2:
    φ^{-1}(y) = [ s + 2 - sqrt((s + 2)^2 - 4) ] / 2, where s = y^{1/θ}.
    Maps y in [0, ∞) to t in (0, 1].
    """
    y = np.asarray(y, dtype=float)
    s = np.power(np.maximum(y, 0.0), 1.0 / theta)
    a = 2.0 + s
    # Guard against tiny negative roundoff under the sqrt
    inner = np.maximum(a * a - 4.0, 0.0)
    t = (a - np.sqrt(inner)) / 2.0
    # Clip into (0,1]
    return np.clip(t, 1e-12, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Genest–Rivest sampler utilities (Archimedean copula)
# ──────────────────────────────────────────────────────────────────────────────

def K_fun(x: float, theta: float) -> float:
    """K(x) = x - φ(x)/φ'(x)."""
    val = dphi_A2(x, theta)
    # add tiny epsilon only in code for safety (not in paper formula)
    return x - (phi_A2(x, theta) / (val + 1e-15))


def K_inv(tval: float, theta: float) -> float:
    """Find w in (0,1) such that K(w) = tval via brentq."""
    a, b = 1e-9, 1.0 - 1e-9
    # Ensure monotone bracket by checking values
    Ka = K_fun(a, theta) - tval
    Kb = K_fun(b, theta) - tval
    # If signs aren't opposite due to numeric noise, nudge bounds
    if Ka * Kb > 0:
        # fallback small expansion around tval by scanning grid
        xs = np.linspace(1e-6, 1.0 - 1e-6, 1000)
        vals = [K_fun(x, theta) - tval for x in xs]
        sign_changes = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
        if len(sign_changes) == 0:
            # As a last resort, return midpoint (very rare); downstream will clip
            return 0.5
        i = sign_changes[0]
        a, b = xs[i], xs[i + 1]
    return brentq(lambda x: K_fun(x, theta) - tval, a, b, xtol=1e-9, rtol=1e-9, maxiter=100)


def sample_copula_A2(theta: float, n: int, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Genest–Rivest sampling for A2 copula. Returns U,V in (0,1]."""
    rs = RandomState(seed)
    s = rs.rand(n)
    t = rs.rand(n)
    U = np.empty(n, dtype=float)
    V = np.empty(n, dtype=float)
    for i in range(n):
        w = K_inv(t[i], theta)
        phiw = phi_A2(w, theta)
        if phiw < 1e-15:
            U[i] = 1.0
            V[i] = 1.0
        else:
            U[i] = phi_A2_inv(s[i] * phiw, theta)
            V[i] = phi_A2_inv((1.0 - s[i]) * phiw, theta)
    return U, V


# ──────────────────────────────────────────────────────────────────────────────
# ECDF inverse (minority-class marginals)
# ──────────────────────────────────────────────────────────────────────────────

def inverse_ecdf(orig: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Map u in (0,1] to sample quantiles of 'orig' via empirical CDF inverse.
    Uses ceil(u*(N)) - 1 indexing, clipped to [0, N-1].
    """
    vals = np.sort(np.asarray(orig, dtype=float))
    N = len(vals)
    # rank index: ceil(u * N) - 1; but since U in (0,1], clip robustly
    idx = np.ceil(np.asarray(u) * (N + 1)).astype(int) - 1
    idx = np.clip(idx, 0, N - 1)
    return vals[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic minority generator from A2 copula (two-feature dependence)
# ──────────────────────────────────────────────────────────────────────────────

def gen_from_copula_A2(minority_train_df: pd.DataFrame,
                       theta: float,
                       n_synth: int,
                       feature_1: str = FEATURE_1,
                       feature_2: str = FEATURE_2,
                       seed: int = SEED) -> pd.DataFrame:
    """
    Generate n_synth synthetic minority rows using A2 copula on (feature_1, feature_2) of minority TRAIN data.
    For the remaining features, bootstrap-with-replacement rows from the minority TRAIN set.
    """
    rs = RandomState(seed)
    U, V = sample_copula_A2(theta=theta, n=n_synth, seed=seed)

    g1 = inverse_ecdf(minority_train_df[feature_1].values, U)
    g2 = inverse_ecdf(minority_train_df[feature_2].values, V)

    other_cols = [c for c in minority_train_df.columns if c not in (feature_1, feature_2)]
    other = minority_train_df[other_cols].sample(n_synth, replace=True, random_state=seed).reset_index(drop=True)

    out = other.copy()
    out[feature_1] = g1
    out[feature_2] = g2
    out["Outcome"] = 1
    return out

class KerasMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 hidden_layers=(64, 32),
                 dropout=0.2,
                 l2=1e-4,
                 learning_rate=1e-3,
                 batch_size=32,
                 epochs=200,
                 patience=20,
                 random_state=42,
                 verbose=0):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.l2 = l2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self._model = None
        self._input_dim = None

    def _build_model(self, input_dim):
        reg = keras.regularizers.l2(self.l2) if self.l2 else None
        model = keras.Sequential()
        for i, h in enumerate(self.hidden_layers):
            if i == 0:
                model.add(keras.layers.Dense(h, activation='relu',
                                             kernel_regularizer=reg, input_shape=(input_dim,)))
            else:
                model.add(keras.layers.Dense(h, activation='relu', kernel_regularizer=reg))
            if self.dropout and self.dropout > 0:
                model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='binary_crossentropy',
                      metrics=[keras.metrics.AUC(name='AUC'),
                               keras.metrics.AUC(curve='PR', name='AUC_PR')])
        return model

    def fit(self, X, y):
        if not HAS_TF:
            raise RuntimeError("TensorFlow/Keras not available. Install tensorflow to use KerasMLPClassifier.")
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        self._input_dim = X.shape[1]
        self._model = self._build_model(self._input_dim)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience,
                                           restore_best_weights=True, verbose=self.verbose)
        self._model.fit(X, y,
                        validation_split=0.2,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        verbose=self.verbose,
                        callbacks=[es])
        return self

    def predict_proba(self, X):
        p = self._model.predict(X, verbose=0).reshape(-1)
        return np.vstack([1.0 - p, p]).T

    def predict(self, X):
        p = self._model.predict(X, verbose=0).reshape(-1)
        return (p >= 0.5).astype(int)

# ──────────────────────────────────────────────────────────────────────────────
# Models & evaluation
# ──────────────────────────────────────────────────────────────────────────────

def get_models() -> Dict[str, object]:
    return {
        "RF": RandomForestClassifier(random_state=SEED),
        "GB": GradientBoostingClassifier(random_state=SEED),
        "XGB": XGBClassifier(eval_metric="logloss", random_state=SEED),
        "LR": LogisticRegression(max_iter=1000, random_state=SEED),
        # Deep-learning baseline (MLP)
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            batch_size=32,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
            random_state=SEED,
        ),
    }



def evaluate_models(models: Dict[str, object],
                    X_tr: np.ndarray, y_tr: np.ndarray,
                    X_te: np.ndarray, y_te: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        m = clone(model)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(X_te)[:, 1]
        else:
            # For models without proba, fall back to labels for AUC (not ideal); all chosen models have proba
            proba = pred.astype(float)
        rows.append({
            "Model": name,
            "Acc": accuracy_score(y_te, pred),
            "Prec": precision_score(y_te, pred),
            "Rec": recall_score(y_te, pred),
            "F1": f1_score(y_te, pred),
            "AUC": roc_auc_score(y_te, proba)
        })
    return pd.DataFrame(rows).set_index("Model")


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(smote_models: Dict[str, object],
                            a2_models: Dict[str, object],
                            X_te_sm: np.ndarray, y_te: np.ndarray,
                            X_te_a2: np.ndarray,
                            title_suffix: str,
                            out_path: str) -> None:
    """
    Two rows × N models: top row SMOTE (Blues), bottom row A2 (Oranges).
    Each model's pair shares the same color scale for fair visual comparison.
    """
    model_names = list(a2_models.keys())  # order from your dict
    n = len(model_names)

    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 9))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, name in enumerate(model_names):
        # Predictions
        yhat_sm = smote_models[name].predict(X_te_sm)
        yhat_a2 = a2_models[name].predict(X_te_a2)

        # Confusion matrices
        cm_sm = confusion_matrix(y_te, yhat_sm)
        cm_a2 = confusion_matrix(y_te, yhat_a2)

        # Share color scale within the pair
        vmax_pair = max(cm_sm.max(), cm_a2.max())

        # --- Top row: SMOTE (Blues) ---
        ax_top = axes[0, i]
        im_top = ax_top.imshow(cm_sm, interpolation="nearest", cmap="Blues",
                               vmin=0, vmax=vmax_pair)
        ax_top.set_title(f"{name} (SMOTE)")
        ax_top.set_xlabel("Predicted"); ax_top.set_ylabel("True")
        for (r, c), v in np.ndenumerate(cm_sm):
            ax_top.text(c, r, str(v), ha="center", va="center")
        fig.colorbar(im_top, ax=ax_top, fraction=0.046, pad=0.04)

        # --- Bottom row: A2 (Oranges) ---
        ax_bot = axes[1, i]
        im_bot = ax_bot.imshow(cm_a2, interpolation="nearest", cmap="Oranges",
                               vmin=0, vmax=vmax_pair)
        ax_bot.set_title(f"{name} (A2 {title_suffix})")
        ax_bot.set_xlabel("Predicted"); ax_bot.set_ylabel("True")
        for (r, c), v in np.ndenumerate(cm_a2):
            ax_bot.text(c, r, str(v), ha="center", va="center")
        fig.colorbar(im_bot, ax=ax_bot, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)





def plot_roc_curves(SM: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    A2_by_theta: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                    models: Dict[str, object],
                    out_path: str) -> None:
    """
    For each model, draw SMOTE ROC and A2 ROC for θ in THETAS.
    Inputs:
      SM: ((X_tr_sm, y_tr_sm), X_te_sm, y_te)
      A2_by_theta: {theta: (X_tr_a2, y_tr_a2, X_te_a2)}
    """
    (X_tr_sm, y_tr_sm), X_te_sm, y_te = SM
    model_names = list(models.keys())
    n = len(model_names)
    rows = math.ceil(n / 2)
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.5 * rows))
    axes = np.array(axes).ravel()

    for i, name in enumerate(model_names):
        ax = axes[i]

        # Fit SMOTE model and ROC
        m_sm = clone(models[name])
        m_sm.fit(X_tr_sm, y_tr_sm)
        proba_sm = m_sm.predict_proba(X_te_sm)[:, 1]
        fpr_sm, tpr_sm, _ = roc_curve(y_te, proba_sm)
        auc_sm = roc_auc_score(y_te, proba_sm)
        ax.plot(fpr_sm, tpr_sm, label=f"SMOTE (AUC={auc_sm:.3f})", lw=2)

        # For each theta
        for theta, (X_tr_a2, y_tr_a2, X_te_a2) in A2_by_theta.items():
            m = clone(models[name])
            m.fit(X_tr_a2, y_tr_a2)
            proba = m.predict_proba(X_te_a2)[:, 1]
            fpr, tpr, _ = roc_curve(y_te, proba)
            auc = roc_auc_score(y_te, proba)
            ax.plot(fpr, tpr, lw=2, label=f"A2 θ={theta} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.grid(True)

    # Hide unused axes if any
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def plot_pr_curves(SM: Tuple[np.ndarray, np.ndarray, np.ndarray],
                   A2_by_theta: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                   models: Dict[str, object],
                   out_path: str) -> None:
    """
    For each model, draw SMOTE PR curve and A2 PR curves for θ in THETAS.
    Inputs:
      SM: ((X_train_sm, y_train_sm), X_test_sm, y_test)
      A2_by_theta: {theta: (X_train_a2, y_train_a2, X_test_a2)}
    """
    (X_tr_sm, y_tr_sm), X_te_sm, y_te = SM
    model_names = list(models.keys())
    n = len(model_names)
    rows = math.ceil(n / 2)
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.5 * rows))
    axes = np.array(axes).ravel()

    for i, name in enumerate(model_names):
        ax = axes[i]

        # SMOTE
        m_sm = clone(models[name])
        m_sm.fit(X_tr_sm, y_tr_sm)
        proba_sm = m_sm.predict_proba(X_te_sm)[:, 1]
        pr_sm, rc_sm, _ = precision_recall_curve(y_te, proba_sm)
        ap_sm = average_precision_score(y_te, proba_sm)
        ax.plot(rc_sm, pr_sm, lw=2, label=f"SMOTE (AP={ap_sm:.3f})")

        # A2 for each theta
        for theta, (X_tr_a2, y_tr_a2, X_te_a2) in A2_by_theta.items():
            m = clone(models[name])
            m.fit(X_tr_a2, y_tr_a2)
            proba = m.predict_proba(X_te_a2)[:, 1]
            pr, rc, _ = precision_recall_curve(y_te, proba)
            ap = average_precision_score(y_te, proba)
            ax.plot(rc, pr, lw=2, label=f"A2 θ={theta} (AP={ap:.3f})")

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left")
        ax.grid(True)

    # Hide unused axes if any
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def plot_mcnemar_contingency(table: np.ndarray, chi2: float, p: float, out_path: str) -> None:
    """
    Plots the 2x2 McNemar contingency as a heatmap with labels:
      [[n11 (both correct), n10 (SMOTE only correct)],
       [n01 (A2 only correct), n00 (both incorrect)]]
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(table, cmap="Blues", vmin=0)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Both Correct", "SMOTE Only Correct"], rotation=20)
    ax.set_yticklabels(["A2 Only Correct", "Both Incorrect"])

    for (r, c), v in np.ndenumerate(table):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_title(f"McNemar Contingency (χ²={chi2:.3f}, p={p:.3g})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_copula_samples_and_overlays(minority_df: pd.DataFrame,
                                     thetas: List[int],
                                     feature_1: str, feature_2: str,
                                     n_samples: int,
                                     out_path: str) -> None:
    fig, axes = plt.subplots(2, len(thetas), figsize=(5.5 * len(thetas), 10))
    original_f1 = minority_df[feature_1].values
    original_f2 = minority_df[feature_2].values

    for i, theta in enumerate(thetas):
        U, V = sample_copula_A2(theta=theta, n=n_samples, seed=SEED + i)
        # row 1: U vs V
        axes[0, i].scatter(U, V, alpha=0.5, s=10)
        axes[0, i].set_title(f"A2 Copula Samples (θ={theta})")
        axes[0, i].set_xlabel("U")
        axes[0, i].set_ylabel("V")
        axes[0, i].set_xlim(0, 1)
        axes[0, i].set_ylim(0, 1)
        axes[0, i].grid(True)

        # row 2: overlay in feature space
        g1 = inverse_ecdf(original_f1, U)
        g2 = inverse_ecdf(original_f2, V)
        axes[1, i].scatter(original_f1, original_f2, alpha=0.35, s=20, label="Original")
        axes[1, i].scatter(g1, g2, alpha=0.55, s=20, label="Synthetic")
        axes[1, i].set_title(f"Original vs Synthetic (θ={theta})")
        axes[1, i].set_xlabel(feature_1)
        axes[1, i].set_ylabel(feature_2)
        axes[1, i].grid(True)
        if i == 0:
            axes[1, i].legend()

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Quick unit checks to catch math/code drift
# ──────────────────────────────────────────────────────────────────────────────

def sanity_checks_A2():
    # avoid the most extreme endpoint; it only stresses floating-point, not math
    tgrid = np.linspace(1e-5, 1 - 1e-9, 2000)
    for theta in THETAS:
        # φ should be non-increasing and ~0 at t=1
        phi_vals = phi_A2(tgrid, theta)
        assert np.all(np.diff(phi_vals) <= 1e-12), "phi_A2 should be non-increasing on (0,1]."
        assert abs(phi_A2(1.0, theta)) < 1e-12, "phi_A2(1) should be ~0."

        # Well-conditioned roundtrip: t -> φ(t) -> φ^{-1}(.) ≈ t
        t_back = phi_A2_inv(phi_vals, theta)
        assert np.allclose(tgrid, t_back, rtol=1e-6, atol=1e-9), "phi_inv(phi(t)) mismatch."

        # Harder roundtrip: y -> φ^{-1}(y) -> φ(.) ≈ y (use a subsample & looser rtol)
        ys = phi_vals[::400]  # pick a few across the range, avoid extremes
        y_back = phi_A2(phi_A2_inv(ys, theta), theta)
        assert np.allclose(ys, y_back, rtol=1e-4, atol=1e-8), "phi(phi_inv(y)) mismatch."



# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    sanity_checks_A2()

    # 1) Load
    df = pd.read_csv(DATA_CSV)
    assert "Outcome" in df.columns, "CSV must contain 'Outcome' column."
    X = df.drop(columns="Outcome")
    y = df["Outcome"].astype(int).values

    # 2) ONE split for everything
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    # Use NumPy consistently for scalers (avoids feature-name warnings)
    X_tr_np = X_tr.to_numpy()
    X_te_np = X_te.to_numpy()

    # Keep around convenience DFs
    train_df = X_tr.copy()
    train_df["Outcome"] = y_tr
    test_df = X_te.copy()
    test_df["Outcome"] = y_te

    # 3) Build SMOTE train (on TRAIN ONLY)
    # --- SMOTE path ---
    scaler_sm = StandardScaler().fit(X_tr_np)
    X_tr_sm_scaled = scaler_sm.transform(X_tr_np)
    X_te_sm_scaled = scaler_sm.transform(X_te_np)
    X_tr_res_scaled, y_tr_res = SMOTE(random_state=SEED).fit_resample(X_tr_sm_scaled, y_tr)

    # 4) Build A2-augmented train (on TRAIN ONLY)
    # Minority/majority counts on TRAIN
    train_min = train_df[train_df["Outcome"] == 1].drop(columns="Outcome").reset_index(drop=True)
    train_maj = train_df[train_df["Outcome"] == 0].drop(columns="Outcome").reset_index(drop=True)

    n_min = len(train_min)
    n_maj = len(train_maj)
    n_synth = (n_maj - n_min) if N_SYNTH_PER_TRAIN is None else int(N_SYNTH_PER_TRAIN)
    if n_synth < 0:
        n_synth = 0  # already balanced or minority larger (unlikely here)

    # We'll train/evaluate at θ=2,5,10 for ROC, but θ=10 for confusion matrices (paper default)
    # Prepare dicts storing scaled train/test per theta
    A2_scaled_by_theta: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # We'll also save fitted models at θ=10 for confusion matrices and McNemar
    models_smote = {}
    models_a2_t10 = {}

    models_dict = get_models()

    # For ROC plot, we need SMOTE (packed as ((X_train, y_train), X_test, y_test))
    SM_for_roc = ((X_tr_res_scaled, y_tr_res), X_te_sm_scaled, y_te)

    # Loop over thetas for A2 prep
    for theta in THETAS:
        if n_synth > 0:
            synth = gen_from_copula_A2(train_min, theta=theta, n_synth=n_synth,
                                       feature_1=FEATURE_1, feature_2=FEATURE_2, seed=SEED)
            # Combine with ORIGINAL train set rows
            a2_train_aug = pd.concat([train_df, synth], ignore_index=True)
        else:
            a2_train_aug = train_df.copy()

        # --- A2 path (inside the theta loop) ---
        Xa2_tr_raw = a2_train_aug.drop(columns="Outcome").to_numpy()
        ya2_tr = a2_train_aug["Outcome"].astype(int).to_numpy()
        scaler_a2 = StandardScaler().fit(Xa2_tr_raw)
        Xa2_tr_scaled = scaler_a2.transform(Xa2_tr_raw)
        Xa2_te_scaled = scaler_a2.transform(X_te_np)

        # Store for ROC plotting
        A2_scaled_by_theta[theta] = (Xa2_tr_scaled, ya2_tr, Xa2_te_scaled)

        # Train/save models at θ=10 for confusion matrices & McNemar
        if theta == 10:
            for name, model in models_dict.items():
                m = clone(model)
                m.fit(Xa2_tr_scaled, ya2_tr)
                models_a2_t10[name] = m

    # Also train/save SMOTE models for confusion matrices & McNemar
    for name, model in models_dict.items():
        m = clone(model)
        m.fit(X_tr_res_scaled, y_tr_res)
        models_smote[name] = m

    # 5) Metrics tables (SMOTE vs A2 θ=10 on SAME test set)
    # Evaluate SMOTE models
    sm_metrics = evaluate_models(models_smote, X_tr_res_scaled, y_tr_res, X_te_sm_scaled, y_te)
    # Evaluate A2 θ=10 models
    Xa2_tr_t10, ya2_tr_t10, Xa2_te_t10 = A2_scaled_by_theta[10]
    a2_metrics = evaluate_models(models_a2_t10, Xa2_tr_t10, ya2_tr_t10, Xa2_te_t10, y_te)

    metrics_table = pd.concat({"A2": a2_metrics, "SMOTE": sm_metrics}, names=["Method", "Model"])
    metrics_csv = os.path.join(OUT_DIR, "metrics_table.csv")
    metrics_table.to_csv(metrics_csv, float_format="%.4f")

    # 6) Confusion matrices (θ=10)
    cm_pdf = os.path.join(OUT_DIR, "confusion_matrices.pdf")
    plot_confusion_matrices(models_smote, models_a2_t10, X_te_sm_scaled, y_te, Xa2_te_t10,
                            title_suffix="θ=10", out_path=cm_pdf)

    # 7) ROC curves for θ ∈ {2,5,10}
    roc_pdf = os.path.join(OUT_DIR, "roc_multi_theta.pdf")
    plot_roc_curves(SM=( (X_tr_res_scaled, y_tr_res), X_te_sm_scaled, y_te ),
                    A2_by_theta=A2_scaled_by_theta,
                    models=models_dict,
                    out_path=roc_pdf)

    # 7b) Precision–Recall curves for θ ∈ {2,5,10}
    pr_pdf = os.path.join(OUT_DIR, "precision-recall_curves.pdf")
    plot_pr_curves(SM=((X_tr_res_scaled, y_tr_res), X_te_sm_scaled, y_te),
                   A2_by_theta=A2_scaled_by_theta,
                   models=models_dict,
                   out_path=pr_pdf)

    # 8) Copula samples & overlays
    copula_pdf = os.path.join(OUT_DIR, "myplots.pdf")
    minority_train_only = train_min.copy()
    plot_copula_samples_and_overlays(minority_df=minority_train_only, thetas=THETAS,
                                     feature_1=FEATURE_1, feature_2=FEATURE_2,
                                     n_samples=500, out_path=copula_pdf)

    # 9) McNemar’s test comparing XGB (θ=10 A2) vs XGB (SMOTE) on SAME test set
    xgb_sm = models_smote["XGB"]
    xgb_a2 = models_a2_t10["XGB"]
    yhat_sm = xgb_sm.predict(X_te_sm_scaled)
    yhat_a2 = xgb_a2.predict(Xa2_te_t10)

    # Build contingency table using correctness vs the TRUE y_te
    # n01: A2 correct, SMOTE incorrect
    # n10: SMOTE correct, A2 incorrect
    a2_correct = (yhat_a2 == y_te)
    sm_correct = (yhat_sm == y_te)
    n01 = np.sum(a2_correct & ~sm_correct)
    n10 = np.sum(~a2_correct & sm_correct)
    n11 = np.sum(a2_correct & sm_correct)
    n00 = np.sum(~a2_correct & ~sm_correct)

    table = np.array([[n11, n10],
                      [n01, n00]], dtype=int)

    mc = mcnemar(table, exact=True)
    mcnemar_json = {
        "contingency_table": {"n11_both_correct": int(n11),
                              "n10_smote_only_correct": int(n10),
                              "n01_a2_only_correct": int(n01),
                              "n00_both_incorrect": int(n00)},
        "statistic": float(mc.statistic if mc.statistic is not None else float("nan")),
        "p_value": float(mc.pvalue)
    }
    with open(os.path.join(OUT_DIR, "mcnemar.json"), "w") as f:
        json.dump(mcnemar_json, f, indent=2)

    contingency_pdf = os.path.join(OUT_DIR, "mcnemar_contingency.pdf")
    chi2_val = float(mc.statistic) if mc.statistic is not None else float("nan")
    plot_mcnemar_contingency(table, chi2=chi2_val, p=float(mc.pvalue), out_path=contingency_pdf)

    # 10) Save a short README with paths
    with open(os.path.join(OUT_DIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Outputs generated by CopulaSMOTE revision pipeline\n")
        f.write(f"- Metrics table: {metrics_csv}\n")
        f.write(f"- Confusion matrices (θ=10): {cm_pdf}\n")
        f.write(f"- ROC curves (θ in {THETAS}): {roc_pdf}\n")
        f.write(f"- Precision–Recall curves (θ in {THETAS}): {pr_pdf}\n")
        f.write(f"- Copula sample plots: {copula_pdf}\n")
        f.write(f"- McNemar results: {os.path.join(OUT_DIR, 'mcnemar.json')}\n")
        f.write(f"- McNemar contingency figure: {contingency_pdf}\n")

    print("✓ Done. Outputs written to:", OUT_DIR)


if __name__ == "__main__":
    if DATA_CSV == "/path/to/your/pima_diabetes_data.csv":
        print("Please edit DATA_CSV in this script to point to your CSV and run again.")
    else:
        main()
