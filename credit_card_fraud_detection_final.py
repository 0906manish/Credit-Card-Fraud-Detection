import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, roc_auc_score, auc,
    classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Continuous optimization
import optuna

plt.rcParams["figure.dpi"] = 120


# Metrics / plotting

def pr_auc_score(y_true, y_prob):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return auc(r, p)

def print_metrics(name: str, y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)
    pr = pr_auc_score(y_true, y_prob)
    print(f"\n=== {name} ===")
    print(f"Threshold  : {thr:.4f}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"ROC-AUC    : {roc:.4f}")
    print(f"PR-AUC     : {pr:.4f}")
    print("Report:\n", classification_report(y_true, y_pred, digits=4))
    return {"Model": name, "Threshold": thr, "Accuracy": acc, "ROC-AUC": roc, "PR-AUC": pr}

def plot_curves(name: str, y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    p, r, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.4f}")
    ax[0].plot([0, 1], [0, 1], "--", linewidth=1)
    ax[0].set_title(f"ROC — {name}")
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()

    ax[1].plot(r, p, label=f"AP={pr_auc_score(y_true, y_prob):.4f}")
    ax[1].set_title(f"Precision-Recall — {name}")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend()

    plt.tight_layout(); plt.show()


# Data loading & preprocessing

def load_and_preprocess(csv_path: str, seed=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    assert "Class" in df.columns, "Dataset must contain 'Class' column."

    X = df.drop(columns=["Class"]).values
    y = df["Class"].values.astype(int)

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_imp)

    X_train, X_test, y_train, y_test = train_test_split(
        X_std, y, test_size=0.2, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


# Mollifier weights via k-NN over FRAUD samples

def mollifier_weights_knn(
    X_all: np.ndarray,
    X_fraud: np.ndarray,
    epsilon: float,
    k: int = 50,
    leaf_size: int = 40,
) -> np.ndarray:
    """ M_eps(x) = mean_{j in kNN_fraud(x)} exp(-||x - x_j||^2 / (2*epsilon^2)) """
    if X_fraud.shape[0] == 0:  # safety
        return np.ones(X_all.shape[0], dtype=float)

    k_eff = min(k, X_fraud.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", leaf_size=leaf_size)
    nn.fit(X_fraud)
    dists, _ = nn.kneighbors(X_all, return_distance=True)  # (n, k_eff)

    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    ker = np.exp(-(dists ** 2) / (2.0 * (epsilon ** 2)))
    m = ker.mean(axis=1)

    return m / (m.mean() + 1e-12)  # normalize mean weight ~ 1


# Composite sample weights (epsilon, lambda, gamma)

def composite_sample_weights(
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    c0: float = 1.0,
    c1: float = 1.0,
    lam: float = 0.6,
    gamma: float = 0.2,   # negative halo on non-fraud near fraud
    k: int = 50
):
    """
    w_i = c0 * [(1 - lam) + gamma * w_mol(x_i)] * 1[y=0]
        + c1 * [ lam * w_mol(x_i)             ] * 1[y=1]
    """
    X_fraud = X[y == 1]
    if X_fraud.shape[0] == 0:
        w = np.where(y == 0, c0*(1-lam), c1*lam).astype(float)
        return w / (w.mean() + 1e-12)

    w_mol = mollifier_weights_knn(X, X_fraud, epsilon=eps, k=k)

    w_neg = c0 * ((1.0 - lam) + gamma * w_mol) * (y == 0)
    w_pos = c1 * (lam * w_mol) * (y == 1)
    w = (w_neg + w_pos).astype(float)

    return w / (w.mean() + 1e-12)


# Threshold utilities

def best_threshold_by_fbeta(y_true, y_prob, beta=1.0):
    P, R, T = precision_recall_curve(y_true, y_prob)
    eps = 1e-12
    fbeta = (1 + beta**2) * (P * R) / (beta**2 * P + R + eps)
    idx = int(fbeta.argmax())
    thr = T[max(idx-1, 0)] if len(T) > 0 else 0.5
    return thr, float(P[idx]), float(R[idx]), float(fbeta[idx])

def precision_recall_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(precision), float(recall)

def meets_constraints(y_true, y_prob, p_min=None, r_min=None):
    P, R, T = precision_recall_curve(y_true, y_prob)
    eps = 1e-12
    f1 = 2*P*R/(P+R+eps)
    candidates = []
    for i in range(len(P)):
        ok_p = (p_min is None) or (P[i] >= p_min)
        ok_r = (r_min is None) or (R[i] >= r_min)
        if ok_p and ok_r:
            thr = T[max(i-1, 0)] if len(T) > 0 else 0.5
            candidates.append((f1[i], thr, P[i], R[i]))
    if candidates:
        candidates.sort(key=lambda z: z[0], reverse=True)
        f1_best, thr, p, r = candidates[0]
        return True, thr, float(p), float(r)
    j = int(f1.argmax())
    thr = T[max(j-1, 0)] if len(T) > 0 else 0.5
    return False, thr, float(P[j]), float(R[j])


# Continuous tuning of (epsilon, lambda, gamma, threshold) 

def tune_eps_lambda_gamma_thr_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    xgb_params: dict,
    seed: int = 42,
    beta: float = 1.0,
    p_min: float = None,
    r_min: float = None,
    k: int = 50,
    n_trials: int = 80,
    eps_bounds: tuple = (1e-2, 2.0),     # log-uniform over [0.01, 2.0]
    lam_bounds: tuple = (0.1, 0.95),     # uniform over [0.1, 0.95]
    gamma_bounds: tuple = (0.0, 0.6),    # uniform over [0.0, 0.6]
    thr_bounds: tuple = (0.01, 0.99)     # uniform over [0.01, 0.99]
):

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )

    history: List[Dict] = []

    def objective(trial: optuna.trial.Trial):
        # Suggest continuous parameters
        eps = trial.suggest_float("epsilon", eps_bounds[0], eps_bounds[1], log=True)
        lam = trial.suggest_float("lambda", lam_bounds[0], lam_bounds[1])
        gamma = trial.suggest_float("gamma", gamma_bounds[0], gamma_bounds[1])
        thr = trial.suggest_float("threshold", thr_bounds[0], thr_bounds[1])

        # Build weights and train
        w_tr = composite_sample_weights(X_tr, y_tr, eps=eps, lam=lam, gamma=gamma, k=k)
        model = xgb.XGBClassifier(**xgb_params, random_state=seed)
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        yv_prob = model.predict_proba(X_val)[:, 1]

        # Evaluate at the proposed threshold (continuous tuning of thr)
        p, r = precision_recall_at_threshold(y_val, yv_prob, thr)

        if p_min is not None or r_min is not None:
            feasible = True
            if p_min is not None and p < p_min:
                feasible = False
            if r_min is not None and r < r_min:
                feasible = False
            score = (2*p*r/(p+r+1e-12)) if feasible else 0.0
        else:
            # F_beta at this threshold
            beta2 = beta**2
            score = (1+beta2) * (p*r) / (beta2*p + r + 1e-12)

        # Log attributes for later inspection
        trial.set_user_attr("precision", float(p))
        trial.set_user_attr("recall", float(r))
        trial.set_user_attr("score", float(score))

        history.append(dict(eps=eps, lam=lam, gamma=gamma, threshold=thr,
                            precision=p, recall=r, score=score))
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    best_eps = float(best_trial.params["epsilon"])
    best_lam = float(best_trial.params["lambda"])
    best_gamma = float(best_trial.params["gamma"])
    best_thr = float(best_trial.params["threshold"])
    best_p = float(best_trial.user_attrs["precision"])
    best_r = float(best_trial.user_attrs["recall"])
    best_score = float(best_trial.user_attrs["score"])

    # Retrain on FULL training with best (eps, lam, gamma)
    w_full = composite_sample_weights(X_train, y_train, eps=best_eps, lam=best_lam, gamma=best_gamma, k=k)
    final_model = xgb.XGBClassifier(**xgb_params, random_state=seed)
    final_model.fit(X_train, y_train, sample_weight=w_full)

    best = dict(eps=best_eps, lam=best_lam, gamma=best_gamma,
                thr=best_thr, precision=best_p, recall=best_r, score=best_score)

    return best, final_model, history




# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=False, help="Path to Kaggle creditcard.csv")
    parser.add_argument("--plots", action="store_true", help="Show plots for composite model")
    parser.add_argument("--beta", type=float, default=1.0, help="F_beta for selection (default=F1)")
    parser.add_argument("--p_min", type=float, default=None, help="Min precision constraint (overrides beta if set)")
    parser.add_argument("--r_min", type=float, default=None, help="Min recall constraint (overrides beta if set)")
    parser.add_argument("--k", type=int, default=50, help="k for kNN over frauds in mollifier")
    parser.add_argument("--trials", type=int, default=80, help="Optuna trials for continuous tuning")
    parser.add_argument("--eps_min", type=float, default=1e-2, help="Min epsilon (log-uniform)")
    parser.add_argument("--eps_max", type=float, default=2.0, help="Max epsilon (log-uniform)")
    parser.add_argument("--lam_min", type=float, default=0.1, help="Min lambda (uniform)")
    parser.add_argument("--lam_max", type=float, default=0.95, help="Max lambda (uniform)")
    parser.add_argument("--gamma_min", type=float, default=0.0, help="Min gamma (uniform)")
    parser.add_argument("--gamma_max", type=float, default=0.6, help="Max gamma (uniform)")
    parser.add_argument("--thr_min", type=float, default=0.01, help="Min decision threshold (uniform)")
    parser.add_argument("--thr_max", type=float, default=0.99, help="Max decision threshold (uniform)")

    # accept unknown args injected by notebooks
    args, _unknown = parser.parse_known_args()

    # auto-detect CSV if --csv not provided
    if not args.csv:
        for cand in [
            "creditcard.csv",
            "/kaggle/input/creditcardfraud/creditcard.csv",
            "/content/creditcard.csv",
            "/content/drive/MyDrive/creditcard.csv",
        ]:
            if os.path.exists(cand):
                args.csv = cand
                print(f"Auto-detected CSV at: {cand}")
                break

    # final guard: if still missing, try Colab picker or exit nicely
    if not args.csv:
        try:
            from google.colab import files  # type: ignore
            print("No --csv provided and no known path found. Opening file picker…")
            uploaded = files.upload()
            if uploaded:
                args.csv = next(iter(uploaded.keys()))
                print(f"Using uploaded file: {args.csv}")
        except Exception:
            raise SystemExit(
                "Please provide --csv /path/to/creditcard.csv or place creditcard.csv in the working directory."
            )

    SEED = 42
    np.random.seed(SEED)

    print("Loading & preprocessing...")
    X_train, X_test, y_train, y_test = load_and_preprocess(args.csv, seed=SEED)

    xgb_params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=-1,
    )

    print("\nContinuous tuning (Optuna) for (ε, λ, γ, thr) on the COMPOSITE mollifier-weighted XGBoost...")
    best, mol_model, records = tune_eps_lambda_gamma_thr_optuna(
        X_train, y_train,
        xgb_params=xgb_params,
        seed=SEED,
        beta=args.beta, p_min=args.p_min, r_min=args.r_min,
        k=args.k,
        n_trials=args.trials,
        eps_bounds=(args.eps_min, args.eps_max),
        lam_bounds=(args.lam_min, args.lam_max),
        gamma_bounds=(args.gamma_min, args.gamma_max),
        thr_bounds=(args.thr_min, args.thr_max)
    )

    # Evaluate on TEST using tuned threshold
    y_prob_mol = mol_model.predict_proba(X_test)[:, 1]
    title = (
        f"Composite Mollifier (ε={best['eps']:.4f}, λ={best['lam']:.3f}, "
        f"γ={best['gamma']:.3f}, thr={best['thr']:.3f})"
    )
    mol_results = print_metrics(title, y_test, y_prob_mol, thr=best["thr"])

    if args.plots:
        plot_curves(title, y_test, y_prob_mol)

   
if __name__ == "__main__":
    main()
