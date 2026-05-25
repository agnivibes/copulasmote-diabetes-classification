"""
Requirements
------------
    pip install numpy pandas scikit-learn xgboost imbalanced-learn
    pip install pyvinecopulib nflows torch scipy statsmodels matplotlib seaborn
"""

#Set thread limits BEFORE any numpy/sklearn imports =
import os
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]     = "1"

import sys
import warnings
import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats

from sklearn.model_selection   import StratifiedKFold
from sklearn.preprocessing     import StandardScaler
from sklearn.impute             import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)
from sklearn.base     import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import pyvinecopulib as pv

from nflows.flows         import Flow
from nflows.distributions import StandardNormal
from nflows.transforms    import (
    CompositeTransform, AffineCouplingTransform, RandomPermutation,
)
from nflows.nn.nets import ResidualNet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

os.chdir(r"G:\My Drive\CopulaSMOTE_5_17")


# CONFIGURATION

SEED = 42
np.random.seed(SEED)          # global numpy seed for reproducibility

CONFIGS = {
    "PIMA": {
        "path"        : "pima.csv",
        "target"      : "Outcome",
        "sample_size" : None,
        "zero_na_cols": ["Glucose", "BloodPressure", "SkinThickness",
                         "Insulin", "BMI"],
    },
    "IRAQI": {
        "path"        : "Diabetes_aravind.csv",
        "target"      : "CLASS",
        "sample_size" : None,
        "zero_na_cols": [],
    },
    "CDC": {
        "path"        : "diabetes_binary_health_indicators_BRFSS2015.csv",
        "target"      : "Diabetes_binary",
        "sample_size" : None,   # set e.g. 50000 for quick debugging
        "zero_na_cols": [],
    },
}


METHOD_ORDER = [
    "SMOTE",
    "BorderlineSMOTE",
    "ADASYN",
    "Flow",
    "CopulaSMOTE",
]

METHOD_COLORS = {
    "SMOTE"          : "#1f77b4",
    "BorderlineSMOTE": "#9467bd",
    "ADASYN"         : "#8c564b",
    "Flow"           : "#ff7f0e",
    "CopulaSMOTE"           : "#2ca02c",
}

# Common grids for curve averaging
MEAN_FPR = np.linspace(0, 1, 200)
MEAN_REC = np.linspace(0, 1, 200)[::-1]

# Paper-quality plot constants 
PAPER_FONT       = 11
PAPER_TITLE_FONT = 13
PAPER_DPI        = 300
PAPER_LINEWIDTH  = 1.8
PAPER_ALPHA      = 0.15

# DATASET LOADING & CLEANING

def load_dataset(name: str) -> tuple:
    """Return (X: pd.DataFrame, y: np.ndarray, feature_names: list)."""
    cfg    = CONFIGS[name]
    target = cfg["target"]

    # Read CSV (Iraqi file may use \r line endings)  
    if name == "IRAQI":
        with open(cfg["path"], "rb") as f:
            sample = f.read(10_000)
        if b"\r" in sample and b"\n" not in sample:
            df = pd.read_csv(cfg["path"], lineterminator="\r")
        else:
            df = pd.read_csv(cfg["path"])
    else:
        df = pd.read_csv(cfg["path"])

    #  Iraqi-specific cleaning  
    if name == "IRAQI":
        df = df.drop(columns=["ID", "No_Pation"], errors="ignore")

        df[target] = df[target].astype(str).str.strip().str.upper()

        n_P = (df[target] == "P").sum()
        if n_P > 0:
            print(f"  [Iraqi] Dropping {n_P} pre-diabetic 'P' samples")
        df = df[df[target].isin(["N", "Y"])].copy()
        df[target] = df[target].map({"N": 0, "Y": 1}).astype(int)

        if "Gender" in df.columns:
            df["Gender"] = (
                df["Gender"].astype(str).str.strip().str.upper()
                .map({"F": 0, "M": 1})
            )

        feat_cols = [c for c in df.columns if c != target]
        for col in feat_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().reset_index(drop=True)

    # Optional sub-sampling 
    if cfg["sample_size"] and len(df) > cfg["sample_size"]:
        df = df.sample(n=cfg["sample_size"], random_state=SEED).reset_index(drop=True)

    # Zero-coded missing values (PIMA) 
    for col in cfg["zero_na_cols"]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    X = df.drop(columns=target)
    y = df[target].astype(int).values
    return X, y, list(X.columns)


# VINE COPULA


def ecdf_transform(X: np.ndarray) -> np.ndarray:
    """Weibull plotting-position pseudo-observations."""
    n, d = X.shape
    U = np.zeros_like(X, dtype=float)
    for j in range(d):
        ranks      = np.argsort(np.argsort(X[:, j])) + 1
        U[:, j]    = ranks / (n + 1)
    return U


def inverse_ecdf(ref_df: pd.DataFrame, U: np.ndarray) -> pd.DataFrame:
    """Nearest-neighbour inverse ECDF (quantile mapping).

    Note: U values near 1.0 are clipped to index n-1, which means the
    upper tail is slightly underrepresented. This is expected behaviour
    for empirical quantile inversion and is negligible in practice.
    """
    X_syn = np.zeros_like(U, dtype=float)
    for j, col in enumerate(ref_df.columns):
        vals      = np.sort(ref_df[col].values.astype(float))
        n         = len(vals)
        idx       = np.clip((U[:, j] * n).astype(int), 0, n - 1)
        X_syn[:, j] = vals[idx]
    return pd.DataFrame(X_syn, columns=ref_df.columns)


def fit_vine(df: pd.DataFrame, seed: int) -> pv.Vinecop:
    rng = np.random.RandomState(seed)
    X   = df.values.astype(float)

    # Continuous extension: tiny jitter for tied / discrete values
    X += 1e-6 * rng.randn(*X.shape)
    U  = ecdf_transform(X)

    controls = pv.FitControlsVinecop(
        family_set=[
            pv.BicopFamily.indep,
            pv.BicopFamily.gaussian,
            pv.BicopFamily.student,
            pv.BicopFamily.clayton,
            pv.BicopFamily.gumbel,
            pv.BicopFamily.frank,
        ],
        trunc_lvl=min(3, max(1, df.shape[1] - 1)),
        selection_criterion="bic",
        allow_rotations=True,
    )
    return pv.Vinecop.from_data(U, controls=controls)


def gen_vine(minority_df: pd.DataFrame, n_synth: int, seed: int) -> pd.DataFrame:
    if n_synth <= 0:
        return minority_df.iloc[0:0].copy()
    vine = fit_vine(minority_df, seed)
    U    = vine.simulate(n_synth, seeds=[seed])
    U    = np.clip(U, 1e-6, 1 - 1e-6)
    return inverse_ecdf(minority_df, U)



# NORMALIZING FLOW


def build_flow(d: int) -> Flow:
    transforms = []
    for _ in range(4):
        transforms.append(RandomPermutation(features=d))
        transforms.append(
            AffineCouplingTransform(
                mask=torch.tensor(np.arange(d) % 2, dtype=torch.uint8),
                transform_net_create_fn=lambda in_f, out_f: ResidualNet(
                    in_features=in_f,
                    out_features=out_f,
                    hidden_features=32,
                    num_blocks=2,
                ),
            )
        )
    return Flow(CompositeTransform(transforms), StandardNormal([d]))


def train_flow(
    X_train: np.ndarray,
    seed: int,
    epochs: int = 200,
    lr: float = 1e-3,
) -> Flow:
    torch.manual_seed(seed)
    flow  = build_flow(X_train.shape[1])
    optim = torch.optim.Adam(flow.parameters(), lr=lr)
    X_t   = torch.tensor(X_train, dtype=torch.float32)

    flow.train()
    for epoch in range(epochs):
        optim.zero_grad()
        loss = -flow.log_prob(X_t).mean()
        loss.backward()
        optim.step()
        if (epoch + 1) % 50 == 0:
            print(f"    [Flow] epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    return flow


def gen_flow(
    flow: Flow, n_synth: int, columns: list, seed: int
) -> pd.DataFrame:
    torch.manual_seed(seed)
    flow.eval()
    with torch.no_grad():
        samples = flow.sample(n_synth).numpy()
    return pd.DataFrame(samples, columns=columns)


# CLASSIFIERS

def get_models(seed: int) -> dict:
    """Default-hyperparameter classifiers."""
    return {
        "RF" : RandomForestClassifier(random_state=seed),
        "GB" : GradientBoostingClassifier(random_state=seed),
        "XGB": XGBClassifier(eval_metric="logloss", random_state=seed),
        "LR" : LogisticRegression(max_iter=1000),
        "MLP": MLPClassifier(random_state=seed),
    }



# PLOTTING HELPERS  

def plot_roc_curves_averaged(roc_store: dict, out_path: str, dataset: str):
    """2-column grid of mean ROC ± 1-std band curves, paper-ready."""
    models = sorted({k[1] for k in roc_store})
    n_cols = 2
    n_rows = (len(models) + 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 5.5 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, model in enumerate(models):
        ax = axes_flat[i]
        for method in METHOD_ORDER:
            key = (method, model)
            if key not in roc_store or not roc_store[key]:
                continue

            interp_tprs, aucs = [], []
            for fpr, tpr, auc_val in roc_store[key]:
                interp_tpr    = np.interp(MEAN_FPR, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tprs.append(interp_tpr)
                aucs.append(auc_val)

            mean_tpr      = np.mean(interp_tprs, axis=0)
            mean_tpr[-1]  = 1.0
            std_tpr       = np.std(interp_tprs, axis=0)
            mean_auc      = np.mean(aucs)
            std_auc       = np.std(aucs)

            color = METHOD_COLORS[method]
            ax.plot(
                MEAN_FPR, mean_tpr, color=color, lw=PAPER_LINEWIDTH,
                label=f"{method}  ({mean_auc:.3f}±{std_auc:.3f})",
            )
            ax.fill_between(
                MEAN_FPR,
                np.clip(mean_tpr - std_tpr, 0, 1),
                np.clip(mean_tpr + std_tpr, 0, 1),
                color=color, alpha=PAPER_ALPHA,
            )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.35, lw=1)
        ax.set_title(model, fontsize=PAPER_TITLE_FONT, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=PAPER_FONT)
        ax.set_ylabel("True Positive Rate",  fontsize=PAPER_FONT)
        ax.tick_params(labelsize=PAPER_FONT - 1)
        ax.legend(fontsize=PAPER_FONT - 2, loc="lower right",
                  framealpha=0.9, edgecolor="grey")

    for j in range(len(models), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{dataset} — Mean ROC Curves (5×2 CV, 10 folds)",
        fontsize=PAPER_TITLE_FONT + 1, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=PAPER_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves_averaged(pr_store: dict, out_path: str, dataset: str):
    """2-column grid of mean PR ± 1-std band curves, paper-ready."""
    models = sorted({k[1] for k in pr_store})
    n_cols = 2
    n_rows = (len(models) + 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 5.5 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, model in enumerate(models):
        ax = axes_flat[i]
        for method in METHOD_ORDER:
            key = (method, model)
            if key not in pr_store or not pr_store[key]:
                continue

            interp_precs, aps = [], []
            for prec_arr, rec_arr, ap_val in pr_store[key]:
                interp_prec = np.interp(MEAN_REC, rec_arr[::-1], prec_arr[::-1])
                interp_precs.append(interp_prec)
                aps.append(ap_val)

            mean_prec = np.mean(interp_precs, axis=0)
            std_prec  = np.std(interp_precs, axis=0)
            mean_ap   = np.mean(aps)
            std_ap    = np.std(aps)

            color = METHOD_COLORS[method]
            ax.plot(
                MEAN_REC, mean_prec, color=color, lw=PAPER_LINEWIDTH,
                label=f"{method}  ({mean_ap:.3f}±{std_ap:.3f})",
            )
            ax.fill_between(
                MEAN_REC,
                np.clip(mean_prec - std_prec, 0, 1),
                np.clip(mean_prec + std_prec, 0, 1),
                color=color, alpha=PAPER_ALPHA,
            )

        ax.set_title(model, fontsize=PAPER_TITLE_FONT, fontweight="bold")
        ax.set_xlabel("Recall",    fontsize=PAPER_FONT)
        ax.set_ylabel("Precision", fontsize=PAPER_FONT)
        ax.tick_params(labelsize=PAPER_FONT - 1)
        ax.legend(fontsize=PAPER_FONT - 2, loc="lower left",
                  framealpha=0.9, edgecolor="grey")

    for j in range(len(models), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{dataset} — Mean PR Curves (5×2 CV, 10 folds)",
        fontsize=PAPER_TITLE_FONT + 1, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=PAPER_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices_avg(cm_store: dict, out_path: str, dataset: str):
    """Method × Model confusion matrix grid, paper-ready."""
    methods = METHOD_ORDER
    models  = sorted({k[1] for k in cm_store})

    fig, axes = plt.subplots(
        len(methods), len(models),
        figsize=(4.5 * len(models), 3.8 * len(methods)),
        squeeze=False,
    )

    for r, method in enumerate(methods):
        for c, model in enumerate(models):
            ax  = axes[r, c]
            key = (method, model)
            if key in cm_store and cm_store[key]:
                normed = []
                for cm in cm_store[key]:
                    row_sums = cm.sum(axis=1, keepdims=True)
                    row_sums = np.where(row_sums == 0, 1, row_sums)
                    normed.append(cm / row_sums)
                mean_cm = np.mean(normed, axis=0)
                sns.heatmap(
                    mean_cm, annot=True, fmt=".3f", cmap="Blues",
                    ax=ax, cbar=False, vmin=0, vmax=1,
                    annot_kws={"size": PAPER_FONT - 1},
                )
            ax.set_title(f"{method} — {model}",
                         fontsize=PAPER_FONT, fontweight="bold")
            ax.set_xlabel("Predicted", fontsize=PAPER_FONT - 1)
            ax.set_ylabel("True",      fontsize=PAPER_FONT - 1)
            ax.tick_params(labelsize=PAPER_FONT - 2)

    fig.suptitle(
        f"{dataset} — Mean Normalised Confusion Matrices (10 folds)",
        fontsize=PAPER_TITLE_FONT + 1, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=PAPER_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_metric_boxplots(results_df: pd.DataFrame, out_path: str, dataset: str):
    """Stacked vertical boxplots — one metric per row, paper-ready."""
    metrics = ["AUC", "F1", "PR_AUC"]
    fig, axes = plt.subplots(
        len(metrics), 1,
        figsize=(12, 4.5 * len(metrics)),
        squeeze=False,
    )

    for i, metric in enumerate(metrics):
        ax = axes[i, 0]
        sns.boxplot(
            data=results_df, x="Model", y=metric, hue="Method",
            hue_order=METHOD_ORDER, ax=ax, palette=METHOD_COLORS,
            linewidth=1.2, fliersize=3,
        )
        ax.set_title(metric, fontsize=PAPER_TITLE_FONT, fontweight="bold")
        ax.set_xlabel("Classifier", fontsize=PAPER_FONT)
        ax.set_ylabel(metric,       fontsize=PAPER_FONT)
        ax.tick_params(labelsize=PAPER_FONT - 1)
        ax.legend(fontsize=PAPER_FONT - 2, loc="best",
                  framealpha=0.9, edgecolor="grey")

    fig.suptitle(
        f"{dataset} — Metric Distribution (5×2 CV)",
        fontsize=PAPER_TITLE_FONT + 1, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=PAPER_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(results_df: pd.DataFrame, out_path: str, dataset: str):
    """Method × Model heatmap for mean AUC, paper-ready."""
    pivot = (
        results_df.groupby(["Method", "Model"])["AUC"]
        .mean()
        .unstack()
        .reindex(METHOD_ORDER)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".4f", cmap="YlGnBu",
        ax=ax, linewidths=0.6,
        annot_kws={"size": PAPER_FONT},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        f"{dataset} — Mean AUC (5×2 CV)",
        fontsize=PAPER_TITLE_FONT, fontweight="bold",
    )
    ax.tick_params(labelsize=PAPER_FONT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PAPER_DPI, bbox_inches="tight")
    plt.close(fig)



# 5×2 CV PAIRED t-TEST  (Dietterich, 1998)

def dietterich_5x2_test(scores_a: list, scores_b: list):
    """Proper 5×2 CV paired t-test (Dietterich, 1998).

    scores_a, scores_b : lists of length 10 (5 iterations × 2 folds).

    The numerator uses diffs[0] — the difference from the first fold of
    the first iteration only. This is intentional and matches the original
    Dietterich (1998) formulation exactly; it is NOT the mean of all diffs.

    Returns (t_statistic, p_value).
    """
    assert len(scores_a) == 10 and len(scores_b) == 10

    diffs = [a - b for a, b in zip(scores_a, scores_b)]

    s2 = []
    for i in range(5):
        d1   = diffs[2 * i]
        d2   = diffs[2 * i + 1]
        p_i  = (d1 + d2) / 2.0
        s2_i = (d1 - p_i) ** 2 + (d2 - p_i) ** 2
        s2.append(s2_i)

    numerator   = diffs[0]                       # per Dietterich (1998)
    denominator = np.sqrt(np.mean(s2))

    if denominator < 1e-15:
        return 0.0, 1.0

    t_stat  = numerator / denominator
    p_value = 2 * sp_stats.t.sf(abs(t_stat), df=5)
    return float(t_stat), float(p_value)


# BASELINE RESAMPLERS

def get_classical_resampled_sets(
    X_tr_sc: np.ndarray, y_tr: np.ndarray, seed: int
) -> dict:
    """Returns dict: method_name -> (X_resampled, y_resampled).

    Restricted to pure interpolation-based oversamplers:
      SMOTE, BorderlineSMOTE, ADASYN.
    Hybrid editing methods (e.g. SMOTEENN) are excluded by design.
    """
    resampled = {}

    resampled["SMOTE"] = SMOTE(random_state=seed).fit_resample(X_tr_sc, y_tr)

    resampled["BorderlineSMOTE"] = BorderlineSMOTE(
        random_state=seed, kind="borderline-1"
    ).fit_resample(X_tr_sc, y_tr)

    try:
        resampled["ADASYN"] = ADASYN(random_state=seed).fit_resample(X_tr_sc, y_tr)
    except (RuntimeError, ValueError):
        # Fall back to SMOTE on degenerate neighbourhood conditions
        print("    [ADASYN] Degenerate neighbourhood — falling back to SMOTE.")
        resampled["ADASYN"] = SMOTE(random_state=seed).fit_resample(X_tr_sc, y_tr)

    return resampled


# MAIN EXPERIMENT


def run_experiment(dataset_name: str):
    dataset_name = dataset_name.upper()
    assert dataset_name in CONFIGS, (
        f"Unknown dataset '{dataset_name}'. Choose from: {list(CONFIGS.keys())}"
    )

    out_dir = f"results_{dataset_name.lower()}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  DATASET : {dataset_name}")
    print(f"{'='*70}")

    X, y, feat_cols = load_dataset(dataset_name)
    has_missing     = X.isnull().any().any()

    # Global minority class — fixed for ALL folds 
    # Using a single global definition ensures minority-focused metrics
    # (F1_min, Rec_min, etc.) are computed on the same label throughout,
    # even if fold-level class counts differ slightly.
    global_counts  = pd.Series(y).value_counts()
    minority_global = int(global_counts.idxmin())
    majority_global = int(global_counts.idxmax())

    print(f"  Shape          : {X.shape}")
    print(f"  Class balance  : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Minority class : {minority_global} "
          f"(n={global_counts[minority_global]}, "
          f"{100*global_counts[minority_global]/len(y):.1f}%)")
    if has_missing:
        print(f"  Missing values : {X.isnull().sum().sum()} (imputed per fold)")
    print()

    all_results = []
    cv_scores   = {}   # (model, method, metric) -> list[10 floats]
    all_roc     = {}   # (method, model) -> list[(fpr, tpr, auc)]
    all_pr      = {}   # (method, model) -> list[(prec, rec, ap)]
    all_cm      = {}   # (method, model) -> list[cm arrays]

    for iteration in range(5):
        kf = StratifiedKFold(
            n_splits=2, shuffle=True, random_state=SEED + iteration
        )

        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            fold_id = iteration * 2 + fold + 1
            print(f"  Iteration {iteration+1}/5, Fold {fold+1}/2  "
                  f"(global fold {fold_id}/10)")

            seed = SEED + iteration * 10 + fold

            X_tr = X.iloc[train_idx].copy()
            X_te = X.iloc[test_idx].copy()
            y_tr = y[train_idx].copy()
            y_te = y[test_idx].copy()

            # Impute missing (train-only fit) 
            if has_missing:
                imputer = SimpleImputer(strategy="median")
                X_tr = pd.DataFrame(
                    imputer.fit_transform(X_tr),
                    columns=feat_cols, index=X_tr.index,
                )
                X_te = pd.DataFrame(
                    imputer.transform(X_te),
                    columns=feat_cols, index=X_te.index,
                )

            # Scale (train-only fit) 
            scaler   = StandardScaler().fit(X_tr)
            X_tr_sc  = scaler.transform(X_tr)
            X_te_sc  = scaler.transform(X_te)

            df_tr = pd.DataFrame(X_tr_sc, columns=feat_cols)
            df_tr["_TARGET_"] = y_tr

            # Oversampling amount — use global minority label 
            counts        = pd.Series(y_tr).value_counts()
            n_minority_tr = counts.get(minority_global, 0)
            n_majority_tr = counts.get(majority_global, 0)
            n_synth       = int(n_majority_tr - n_minority_tr)

            minority_df = (
                df_tr[df_tr["_TARGET_"] == minority_global]
                .drop(columns="_TARGET_")
            )

            # Classical baselines 
            classical_sets = get_classical_resampled_sets(
                X_tr_sc, y_tr, seed=seed
            )

            # Normalizing Flow 
            flow_model = train_flow(minority_df.values, seed=seed)
            synth_flow = gen_flow(flow_model, n_synth, feat_cols, seed=seed)
            synth_flow["_TARGET_"] = minority_global
            flow_train = (
                pd.concat([df_tr, synth_flow], ignore_index=True)
                .sample(frac=1, random_state=seed)
                .reset_index(drop=True)
            )
            X_flow = flow_train.drop(columns="_TARGET_").values
            y_flow = flow_train["_TARGET_"].values

            # Vine Copula 
            synth_vine = gen_vine(minority_df, n_synth, seed=seed)
            synth_vine["_TARGET_"] = minority_global
            vine_train = (
                pd.concat([df_tr, synth_vine], ignore_index=True)
                .sample(frac=1, random_state=seed)
                .reset_index(drop=True)
            )
            X_vine = vine_train.drop(columns="_TARGET_").values
            y_vine = vine_train["_TARGET_"].values

            # Collect all training sets 
            method_datasets = []
            for method_name in ["SMOTE", "BorderlineSMOTE", "ADASYN"]:
                X_aug, y_aug = classical_sets[method_name]
                method_datasets.append((method_name, X_aug, y_aug))
            method_datasets.extend([
                ("Flow", X_flow, y_flow),
                ("CopulaSMOTE", X_vine, y_vine),
            ])

            # Train & Evaluate 
            models = get_models(seed)

            for method, X_aug, y_aug in method_datasets:
                for name, model in models.items():
                    m = clone(model)
                    m.fit(X_aug, y_aug)

                    pred    = m.predict(X_te_sc)
                    prob    = m.predict_proba(X_te_sc)[:, 1]

                    f1_val  = f1_score(y_te, pred, zero_division=0)
                    auc_val = roc_auc_score(y_te, prob)
                    ap_val  = average_precision_score(y_te, prob)

                    # Minority-class-focused metrics — use global minority label
                    f1_min   = f1_score(
                        y_te, pred, pos_label=minority_global, zero_division=0
                    )
                    prec_min = precision_score(
                        y_te, pred, pos_label=minority_global, zero_division=0
                    )
                    rec_min  = recall_score(
                        y_te, pred, pos_label=minority_global, zero_division=0
                    )
                    prob_min = prob if minority_global == 1 else (1.0 - prob)
                    ap_min   = average_precision_score(
                        (y_te == minority_global).astype(int), prob_min
                    )

                    all_results.append({
                        "Iteration" : iteration + 1,
                        "Fold"      : fold + 1,
                        "Method"    : method,
                        "Model"     : name,
                        "Acc"       : accuracy_score(y_te, pred),
                        "BalAcc"    : balanced_accuracy_score(y_te, pred),
                        "Prec"      : precision_score(y_te, pred, zero_division=0),
                        "Rec"       : recall_score(y_te, pred, zero_division=0),
                        "F1"        : f1_val,
                        "AUC"       : auc_val,
                        "PR_AUC"    : ap_val,
                        "F1_min"    : f1_min,
                        "Prec_min"  : prec_min,
                        "Rec_min"   : rec_min,
                        "PR_AUC_min": ap_min,
                    })

                    for met_name, met_val in [
                        ("AUC", auc_val), ("F1", f1_val), ("PR_AUC", ap_val)
                    ]:
                        cv_scores.setdefault(
                            (name, method, met_name), []
                        ).append(met_val)

                    fpr, tpr, _ = roc_curve(y_te, prob)
                    all_roc.setdefault((method, name), []).append(
                        (fpr, tpr, auc_val)
                    )

                    prec_c, rec_c, _ = precision_recall_curve(y_te, prob)
                    all_pr.setdefault((method, name), []).append(
                        (prec_c, rec_c, ap_val)
                    )

                    all_cm.setdefault((method, name), []).append(
                        confusion_matrix(y_te, pred)
                    )

    # RESULTS

    results_df = pd.DataFrame(all_results)

    # Raw per-fold results
    raw_path = os.path.join(out_dir, f"{dataset_name.lower()}_all_runs.csv")
    results_df.to_csv(raw_path, index=False)

    # Summary (mean ± std)
    metric_cols = [
        "Acc", "BalAcc", "Prec", "Rec", "F1", "AUC", "PR_AUC",
        "F1_min", "Prec_min", "Rec_min", "PR_AUC_min",
    ]
    summary = (
        results_df.groupby(["Method", "Model"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]

    for m in metric_cols:
        summary[m] = (
            summary[f"{m}_mean"].round(4).astype(str)
            + " ± "
            + summary[f"{m}_std"].round(4).astype(str)
        )

    summary_path = os.path.join(
        out_dir, f"{dataset_name.lower()}_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)
    print(f"\n{'='*70}")
    print(f"  RESULTS — {dataset_name}  (mean ± std over 10 folds)")
    print(f"{'='*70}\n")
    print(summary[["Method", "Model"] + metric_cols].to_string(index=False))

    # 5×2 CV paired t-test (Dietterich) 
    print(f"\n{'='*70}")
    print(f"  5×2 CV PAIRED t-TEST  (Dietterich, 1998)")
    print(f"{'='*70}\n")

    stat_rows   = []
    model_names = sorted(results_df["Model"].unique())

    for model_name in model_names:
        for baseline in ["SMOTE", "BorderlineSMOTE", "ADASYN", "Flow"]:
            for metric in ["AUC", "F1", "PR_AUC"]:
                vine_key = (model_name, "CopulaSMOTE",   metric)
                base_key = (model_name, baseline, metric)

                if vine_key not in cv_scores or base_key not in cv_scores:
                    continue

                t_stat, p_val = dietterich_5x2_test(
                    cv_scores[vine_key], cv_scores[base_key]
                )
                vine_mean = np.mean(cv_scores[vine_key])
                base_mean = np.mean(cv_scores[base_key])

                sig = (
                    "***" if p_val < 0.01 else
                    "**"  if p_val < 0.05 else
                    "*"   if p_val < 0.10 else ""
                )

                stat_rows.append({
                    "Model"      : model_name,
                    "Metric"     : metric,
                    "Comparison" : f"CopulaSMOTE vs {baseline}",
                    "Vine_mean"  : round(vine_mean, 4),
                    "Base_mean"  : round(base_mean, 4),
                    "Diff"       : round(vine_mean - base_mean, 4),
                    "t_stat"     : round(t_stat, 3),
                    "p_value"    : round(p_val,  4),
                    "Sig"        : sig,
                })

                print(
                    f"  {model_name:4s} | {metric:6s} | "
                    f"CopulaSMOTE vs {baseline:15s} | "
                    f"Δ = {vine_mean - base_mean:+.4f} | "
                    f"t = {t_stat:+.3f} | p = {p_val:.4f} {sig}"
                )

    stats_df   = pd.DataFrame(stat_rows)
    stats_path = os.path.join(
        out_dir, f"{dataset_name.lower()}_stat_tests.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    # Plots 
    print("\n  Generating plots...")

    plot_roc_curves_averaged(
        all_roc,
        os.path.join(out_dir, f"{dataset_name.lower()}_roc_curves.pdf"),
        dataset_name,
    )
    plot_pr_curves_averaged(
        all_pr,
        os.path.join(out_dir, f"{dataset_name.lower()}_pr_curves.pdf"),
        dataset_name,
    )
    plot_confusion_matrices_avg(
        all_cm,
        os.path.join(out_dir, f"{dataset_name.lower()}_confusion.pdf"),
        dataset_name,
    )
    plot_metric_boxplots(
        results_df,
        os.path.join(out_dir, f"{dataset_name.lower()}_metric_boxplots.pdf"),
        dataset_name,
    )
    plot_heatmap(
        results_df,
        os.path.join(out_dir, f"{dataset_name.lower()}_heatmap.pdf"),
        dataset_name,
    )

    print(f"\n  All outputs saved to ./{out_dir}/")
    print("  Done.\n")



# ENTRY POINT

if __name__ == "__main__":
    ds = sys.argv[1].upper() if len(sys.argv) > 1 else "CDC"
    assert ds in CONFIGS, (
        f"Unknown dataset '{ds}'. Choose from: {list(CONFIGS.keys())}"
    )
    run_experiment(ds)
