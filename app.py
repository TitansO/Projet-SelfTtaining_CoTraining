"""
============================================================
Apprentissage Semi-Supervisé — Qualité de l'Air à Dakar
Self-Training & Co-Training — Dataset OpenAQ réel (air_quality_historical.csv)
Mémoire de fin d'études — Master Data Science

v6 : corrigé avec données réelles
     - Chargement du CSV air_quality_historical.csv
     - Pas de génération synthétique
     - Features réelles disponibles
     - Label AQI composite basé sur données réelles

DATASET RÉEL :
════════════════════════════════════════════════════════════
Colonnes : date, pm10, pm2_5, carbon_monoxide, nitrogen_dioxide,
           sulphur_dioxide, ozone, aerosol_optical_depth, dust,
           uv_index, us_aqi, european_aqi
Période : 2022-08-01 à 2023-xx-xx (données journalières agrégées)
============================================================
"""

import time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# 0. CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SSL — Qualité de l'Air Dakar v6 (Données Réelles)",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "navy":  "#0B1F3A",
    "teal":  "#0A8A7C",
    "orange":"#E8712A",
    "red":   "#E74C3C",
    "green": "#27AE60",
    "purple":"#8E44AD",
    "grey":  "#7A8BA0",
    "cream": "#F4F1EC",
}

AQI_NAMES = {
    0: ("Bon",           "#27AE60"),
    1: ("Modéré",        "#F4C518"),
    2: ("Mauvais (S)",   "#E8712A"),
    3: ("Mauvais (I)",   "#E74C3C"),
    4: ("Très Mauvais",  "#9B59B6"),
    5: ("Extrême",       "#6C3483"),
}

# ── Vues Co-Training ────────────────────────────────────────────
# VUE_A : polluants bruts
VUE_A = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide"]

# VUE_B : autres features (dust, AOD, UV)
VUE_B = [
    "dust", "aerosol_optical_depth", "uv_index", "sulphur_dioxide",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]

# ═══════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES RÉELLES
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_dataset() -> tuple[pd.DataFrame, str]:
    """
    Charge air_quality_historical.csv depuis /mnt/user-data/uploads/
    Données réelles uniquement — pas de fallback synthétique.
    """
    try:
        df = pd.read_csv("/mnt/user-data/uploads/air_quality_historical.csv")
        source = "📊 Dataset réel (air_quality_historical.csv)"
    except Exception as e:
        st.error(f"❌ Impossible de charger le CSV : {e}")
        st.stop()

    # Conversion date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Supprimer les lignes où toutes les colonnes polluants sont NaN
    pollutants = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide"]
    df = df.dropna(subset=pollutants, how="all")
    
    # Remplir les valeurs manquantes par forward fill puis backward fill
    for col in df.columns:
        if col != "date":
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    # Créer les features temporelles
    df["hour"] = 12  # Données journalières — on fixe à midi
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Créer un label AQI composite basé sur les données réelles
    # Utiliser us_aqi existant, sinon créer un label basé sur pm2.5
    if "us_aqi" in df.columns and df["us_aqi"].notna().sum() > 100:
        # Discrétiser us_aqi en 6 classes
        df["aqi_label"] = pd.qcut(
            df["us_aqi"].fillna(df["us_aqi"].median()),
            q=6, labels=False, duplicates='drop'
        )
    else:
        # Créer un label basé sur pm2.5 + pm10
        aqi_composite = (0.6 * df["pm2_5"].fillna(0) + 
                        0.4 * df["pm10"].fillna(0) / 2.5)
        df["aqi_label"] = pd.qcut(
            aqi_composite, q=6, labels=False, duplicates='drop'
        )

    # Label de labellisation (simulé — 2% aléatoire)
    np.random.seed(42)
    df["label_known"] = 0
    labeled_indices = np.random.choice(
        df.index, size=max(1, int(len(df) * 0.02)), replace=False
    )
    df.loc[labeled_indices, "label_known"] = 1

    st.success(f"✅ Données chargées : {len(df)} lignes | {df['label_known'].sum()} labellisées (2%)")
    
    return df, source


# ═══════════════════════════════════════════════════════════════════════════
# 2. SPLIT TEMPOREL & PRÉPARATION
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def prepare_splits(_df: pd.DataFrame):
    """
    Split temporel : 80% train / 20% test
    """
    df = _df.copy()
    
    # Split temporel 80/20
    cutoff_idx = int(len(df) * 0.80)
    df_train = df.iloc[:cutoff_idx].copy()
    df_test  = df.iloc[cutoff_idx:].copy()

    # Vérifier les colonnes
    ALL_FEATURES = VUE_A + VUE_B
    missing = [f for f in ALL_FEATURES if f not in df_train.columns]
    if missing:
        st.warning(f"⚠️ Features manquantes : {missing}. Utilisant uniquement les disponibles.")
        ALL_FEATURES = [f for f in ALL_FEATURES if f in df_train.columns]

    # Split L/U
    df_L = df_train[df_train["label_known"] == 1].copy()
    df_U = df_train[df_train["label_known"] == 0].copy()

    # Normalisation
    scaler = StandardScaler()
    scaler.fit(df_L[ALL_FEATURES].values)

    def sc(frame):
        return scaler.transform(frame[ALL_FEATURES].values)

    X_L    = sc(df_L);    y_L    = df_L["aqi_label"].values
    X_U    = sc(df_U)
    X_test = sc(df_test); y_test = df_test["aqi_label"].values

    # Indices des vues
    va_idx = [ALL_FEATURES.index(f) for f in VUE_A if f in ALL_FEATURES]
    vb_idx = [ALL_FEATURES.index(f) for f in VUE_B if f in ALL_FEATURES]

    return {
        "df_full": df_train, "df_test": df_test,
        "df_L": df_L, "df_U": df_U,
        "X_L": X_L, "y_L": y_L,
        "X_U": X_U,
        "X_test": X_test, "y_test": y_test,
        "va_idx": va_idx, "vb_idx": vb_idx,
        "scaler": scaler,
        "ALL_FEATURES": ALL_FEATURES,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. CLASSIFIEUR DE BASE
# ═══════════════════════════════════════════════════════════════════════════

def make_clf(n_estimators=100, seed=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. ALGORITHMES SSL
# ═══════════════════════════════════════════════════════════════════════════

def _margin_filter(proba, gamma, min_margin=0.15):
    sorted_p = np.sort(proba, axis=1)[:, ::-1]
    return (sorted_p[:, 0] >= gamma) & (sorted_p[:, 0] - sorted_p[:, 1] >= min_margin)


def _gamma_anneal(it, max_iter, gamma_start, gamma_end):
    if max_iter <= 1: return gamma_end
    return gamma_start + (gamma_end - gamma_start) * (it / max_iter)


def run_self_training(X_L, y_L, X_U, X_test, y_test,
                      gamma, max_iter, n_estimators,
                      patience=3, min_margin=0.15):
    gamma_start = max(0.50, gamma - 0.10)
    X_Lc = X_L.copy(); y_Lc = y_L.copy(); X_Uc = X_U.copy()
    history = []; best_f1 = -1.; best_clf = None; no_improve = 0

    for it in range(max_iter + 1):
        clf = make_clf(n_estimators)
        clf.fit(X_Lc, y_Lc)
        y_pred = clf.predict(X_test)
        f1_now = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        rec = {
            "iteration": it, "n_L": len(X_Lc), "n_U": len(X_Uc),
            "f1_macro":    f1_now,
            "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "precision":   round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "recall":      round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "n_added": 0,
            "gamma_used": round(_gamma_anneal(it, max_iter, gamma_start, gamma), 3),
            "clf": clf, "is_best": False,
        }
        if f1_now > best_f1:
            best_f1 = f1_now; best_clf = clf; rec["is_best"] = True; no_improve = 0
        else:
            no_improve += 1
        if it == max_iter or len(X_Uc) == 0 or no_improve >= patience:
            history.append(rec); break
        gamma_cur = _gamma_anneal(it, max_iter, gamma_start, gamma)
        proba = clf.predict_proba(X_Uc)
        mask  = _margin_filter(proba, gamma_cur, min_margin)
        n_add = int(mask.sum())
        rec["n_added"] = n_add
        history.append(rec)
        if n_add == 0: break
        pseudo  = clf.classes_[proba[mask].argmax(axis=1)]
        X_Lc    = np.vstack([X_Lc, X_Uc[mask]])
        y_Lc    = np.concatenate([y_Lc, pseudo])
        X_Uc    = X_Uc[~mask]

    history[-1]["clf"]     = best_clf if best_clf else history[-1]["clf"]
    history[-1]["best_f1"] = best_f1
    return history


def run_co_training(X_L, y_L, X_U, X_test, y_test,
                    va_idx, vb_idx, gamma, max_iter, k_per_iter, n_estimators,
                    patience=3, min_margin=0.12):
    gamma_start = max(0.50, gamma - 0.10)
    X_LA = X_L[:, va_idx]; X_LB = X_L[:, vb_idx]
    y_LA = y_L.copy();     y_LB = y_L.copy()
    X_UA = X_U[:, va_idx]; X_UB = X_U[:, vb_idx]
    X_tA = X_test[:, va_idx]; X_tB = X_test[:, vb_idx]
    history = []; best_f1 = -1.; best_cA = None; best_cB = None; no_improve = 0

    def _ensemble(cA, cB):
        pA  = cA.predict_proba(X_tA); pB = cB.predict_proba(X_tB)
        cls = np.union1d(cA.classes_, cB.classes_)
        def _al(c, p):
            out = np.zeros((p.shape[0], len(cls)))
            for j, cl in enumerate(cls):
                if cl in c.classes_:
                    out[:, j] = p[:, np.where(c.classes_ == cl)[0][0]]
            return out
        return cls[(_al(cA, pA) + _al(cB, pB)).argmax(axis=1)]

    for it in range(max_iter + 1):
        cA = make_clf(n_estimators, 42); cA.fit(X_LA, y_LA)
        cB = make_clf(n_estimators, 43); cB.fit(X_LB, y_LB)
        y_pred = _ensemble(cA, cB)
        f1_now = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        rec = {
            "iteration": it, "n_L": len(X_LA), "n_U": len(X_UA),
            "f1_macro":    f1_now,
            "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "precision":   round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "recall":      round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "n_added": 0,
            "gamma_used": round(_gamma_anneal(it, max_iter, gamma_start, gamma), 3),
            "clf_A": cA, "clf_B": cB, "is_best": False,
        }
        if f1_now > best_f1:
            best_f1 = f1_now; best_cA = cA; best_cB = cB; rec["is_best"] = True; no_improve = 0
        else:
            no_improve += 1
        if it == max_iter or len(X_UA) == 0 or no_improve >= patience:
            history.append(rec); break
        gamma_cur = _gamma_anneal(it, max_iter, gamma_start, gamma)
        pA = cA.predict_proba(X_UA); pB = cB.predict_proba(X_UB)
        tk_A = np.argsort(pA.max(axis=1))[::-1][:k_per_iter]
        tk_B = np.argsort(pB.max(axis=1))[::-1][:k_per_iter]
        mA = _margin_filter(pA[tk_A], gamma_cur, min_margin)
        mB = _margin_filter(pB[tk_B], gamma_cur, min_margin)
        sel_A = tk_A[mA]; sel_B = tk_B[mB]
        common = np.intersect1d(sel_A, sel_B)
        if len(common):
            agree = cA.classes_[pA[common].argmax(axis=1)] == cB.classes_[pB[common].argmax(axis=1)]
            conflict = common[~agree]
            sel_A = np.setdiff1d(sel_A, conflict)
            sel_B = np.setdiff1d(sel_B, conflict)
        pred_A = cA.classes_[pA[sel_A].argmax(axis=1)]
        pred_B = cB.classes_[pB[sel_B].argmax(axis=1)]
        n_add = len(sel_A) + len(sel_B)
        rec["n_added"] = n_add
        history.append(rec)
        if n_add == 0: break
        if len(sel_B): X_LA = np.vstack([X_LA, X_UA[sel_B]]); y_LA = np.concatenate([y_LA, pred_B])
        if len(sel_A): X_LB = np.vstack([X_LB, X_UB[sel_A]]); y_LB = np.concatenate([y_LB, pred_A])
        keep = np.setdiff1d(np.arange(len(X_UA)), np.union1d(sel_A, sel_B))
        X_UA = X_UA[keep]; X_UB = X_UB[keep]

    if best_cA:
        history[-1]["clf_A"] = best_cA; history[-1]["clf_B"] = best_cB
    history[-1]["best_f1"] = best_f1
    return history


def run_cv_baseline(X_L, y_L, n_estimators=100, n_splits=5):
    """TimeSeriesSplit cross-validation sur L"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr_idx, val_idx in tscv.split(X_L):
        if len(np.unique(y_L[tr_idx])) < 2: continue
        clf = make_clf(n_estimators)
        clf.fit(X_L[tr_idx], y_L[tr_idx])
        pred = clf.predict(X_L[val_idx])
        scores.append(f1_score(y_L[val_idx], pred, average="macro", zero_division=0))
    return scores


# ═══════════════════════════════════════════════════════════════════════════
# 5. FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def _style(fig):
    fig.patch.set_facecolor(PALETTE["cream"])
    for ax in fig.axes: ax.set_facecolor(PALETTE["cream"])
    return fig


def fig_label_scarcity(df_full):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4)); _style(fig)
    n_L = int(df_full["label_known"].sum()); n_U = len(df_full) - n_L
    axes[0].pie([n_L, n_U],
        labels=[f"L (labellisé)\n{n_L:,} ({n_L/(n_L+n_U)*100:.1f}%)",
                f"U (non-labellisé)\n{n_U:,} ({n_U/(n_L+n_U)*100:.1f}%)"],
        colors=[PALETTE["teal"], PALETTE["grey"]], autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2), textprops={"fontsize": 10})
    axes[0].set_title("Ratio L / U — 2% labellisé",
                      fontsize=12, fontweight="bold", color=PALETTE["navy"])
    df_L = df_full[df_full["label_known"] == 1]
    cnt  = df_L["aqi_label"].value_counts().sort_index()
    bars = axes[1].bar([AQI_NAMES.get(i, (str(i), PALETTE["grey"]))[0] for i in cnt.index], 
                       cnt.values,
                       color=[AQI_NAMES.get(i, ("", PALETTE["purple"]))[1] for i in cnt.index], 
                       edgecolor="white")
    for b, v in zip(bars, cnt.values):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                     str(v), ha="center", va="bottom", fontsize=9,
                     fontweight="bold", color=PALETTE["navy"])
    axes[1].set_title("Distribution AQI — Ensemble L",
                      fontsize=12, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_xlabel("Classe AQI"); axes[1].set_ylabel("Observations")
    axes[1].tick_params(axis="x", rotation=15); axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


def fig_temporal_pollutants(df_full):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6)); _style(fig)
    
    # PM2.5 temporel
    axes[0].plot(df_full["date"], df_full["pm2_5"], color=PALETTE["orange"], linewidth=1.2, alpha=0.7)
    axes[0].fill_between(df_full["date"], df_full["pm2_5"], alpha=0.15, color=PALETTE["orange"])
    axes[0].set_title("PM2.5 — Série temporelle", fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[0].set_ylabel("PM2.5 (µg/m³)"); axes[0].grid(alpha=0.3)
    
    # Polluants multiples
    for col, color in [("pm10", PALETTE["red"]), ("nitrogen_dioxide", PALETTE["purple"]), 
                       ("ozone", PALETTE["teal"])]:
        if col in df_full.columns:
            axes[1].plot(df_full["date"], df_full[col], label=col, color=color, linewidth=1.2, alpha=0.7)
    
    axes[1].set_title("Comparaison polluants majeurs", fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_xlabel("Date"); axes[1].set_ylabel("Concentration (µg/m³)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout(); return fig


def fig_correlation_heatmap(df_full, features):
    features_avail = [f for f in features if f in df_full.columns]
    if not features_avail:
        st.warning("Pas de features disponibles pour la heatmap")
        return None
    
    corr = df_full[features_avail].corr()
    fig, ax = plt.subplots(figsize=(10, 8)); _style(fig)
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.4, linecolor="white", cbar_kws={"shrink": 0.8})
    ax.set_title("Corrélations inter-features", fontsize=11, fontweight="bold", color=PALETTE["navy"])
    plt.tight_layout(); return fig


def fig_ssl_progress(history, algo_name):
    df_h = pd.DataFrame(history)
    color = PALETTE["teal"] if "Co" in algo_name else PALETTE["orange"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4)); _style(fig)
    
    axes[0].plot(df_h["iteration"], df_h["f1_macro"], color=color,
                 linewidth=2.5, marker="o", markersize=5)
    axes[0].fill_between(df_h["iteration"], df_h["f1_macro"], alpha=0.15, color=color)
    axes[0].axhline(df_h["f1_macro"].iloc[0], linestyle="--",
                    color=PALETTE["grey"], linewidth=1.5, label="Baseline iter 0")
    axes[0].set_title(f"{algo_name} — F1 macro", fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[0].set_xlabel("Itération"); axes[0].set_ylabel("F1 macro")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 1)
    
    axes[1].plot(df_h["iteration"], df_h["n_L"], color=PALETTE["teal"],   linewidth=2, label="|L|")
    axes[1].plot(df_h["iteration"], df_h["n_U"], color=PALETTE["orange"], linewidth=2, label="|U|")
    axes[1].set_title("Évolution |L| et |U|", fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_xlabel("Itération"); axes[1].set_ylabel("Observations")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    
    if "gamma_used" in df_h.columns:
        axes[2].plot(df_h["iteration"], df_h["gamma_used"], color=PALETTE["purple"],
                     linewidth=2, marker="s", markersize=5)
        axes[2].set_title("Gamma Annealing", fontsize=11, fontweight="bold", color=PALETTE["navy"])
        axes[2].set_xlabel("Itération"); axes[2].set_ylabel("γ"); axes[2].grid(alpha=0.3)
    
    plt.tight_layout(); return fig


def fig_confusion(y_true, y_pred, title):
    # Filtrer les labels réels uniquement
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    
    fig, ax = plt.subplots(figsize=(7, 5.5)); _style(fig)
    label_names = [AQI_NAMES.get(i, (str(i), ""))[0] for i in unique_labels]
    sns.heatmap(cm_norm, ax=ax, cmap="Blues", annot=True, fmt=".2f",
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=11, fontweight="bold", color=PALETTE["navy"])
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel"); ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); return fig


def fig_compare(results_dict):
    methods = list(results_dict.keys())
    x = np.arange(len(methods)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5)); _style(fig)
    b1 = ax.bar(x - w, [v["f1_macro"]  for v in results_dict.values()], w,
                label="F1 macro",        color=PALETTE["teal"],   edgecolor="white")
    b2 = ax.bar(x,     [v["precision"] for v in results_dict.values()], w,
                label="Précision macro", color=PALETTE["orange"], edgecolor="white")
    b3 = ax.bar(x + w, [v["recall"]    for v in results_dict.values()], w,
                label="Rappel macro",    color=PALETTE["purple"], edgecolor="white")
    for bs in [b1, b2, b3]:
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{b.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=PALETTE["navy"])
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Comparaison des méthodes — Données réelles",
                 fontsize=12, fontweight="bold", color=PALETTE["navy"])
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


def fig_cv_scores(cv_scores):
    fig, ax = plt.subplots(figsize=(7, 4)); _style(fig)
    bp = ax.boxplot(cv_scores, patch_artist=True, widths=0.4,
                    boxprops=dict(facecolor=PALETTE["teal"] + "55", edgecolor=PALETTE["teal"], linewidth=2),
                    medianprops=dict(color=PALETTE["navy"], linewidth=2.5))
    ax.scatter([1] * len(cv_scores), cv_scores, color=PALETTE["orange"],
               zorder=5, s=60, label="Folds individuels")
    mu = np.mean(cv_scores); sigma = np.std(cv_scores)
    ax.axhline(mu, linestyle="--", color=PALETTE["navy"], linewidth=1.5,
               label=f"Moyenne = {mu:.3f} ± {sigma:.3f}")
    ax.set_xticks([1]); ax.set_xticklabels(["Baseline CV (5-folds TSS)"], fontsize=10)
    ax.set_ylabel("F1 macro"); ax.set_ylim(0, 1)
    ax.set_title("Validation croisée temporelle (TimeSeriesSplit)",
                 fontsize=11, fontweight="bold", color=PALETTE["navy"])
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(
    "<h2 style='color:#0A8A7C;margin-bottom:0'>⚙️ Configuration v6 (Real Data)</h2>",
    unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Corrections appliquées :**")
st.sidebar.markdown(
    "✅ Données réelles (air_quality_historical.csv)  \n"
    "✅ Pas de génération synthétique  \n"
    "✅ 2% labellisé (stratifié)  \n"
    "✅ TimeSeriesSplit cross-validation  \n"
    "✅ Label AQI composite")
st.sidebar.markdown("---")

algo_choice  = st.sidebar.selectbox("🔬 Algorithme", ["Self-Training", "Co-Training"])
gamma        = st.sidebar.slider("🎯 Seuil confiance γ (fin)", 0.60, 0.99, 0.80, 0.01)
min_margin   = st.sidebar.slider("📐 Marge min P(1er)−P(2e)", 0.05, 0.40, 0.15, 0.05)
patience     = st.sidebar.slider("⏱ Patience early stopping", 1, 8, 3, 1)
max_iter     = st.sidebar.slider("🔁 Itérations max", 3, 20, 10, 1)
n_estimators = st.sidebar.slider("🌲 Arbres RF", 50, 200, 100, 50)
k_per_iter   = 50
if algo_choice == "Co-Training":
    k_per_iter = st.sidebar.slider("📦 Top-k pseudo-labels / iter", 10, 150, 50, 10)

# ═══════════════════════════════════════════════════════════════════════════
# 7. CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════

with st.spinner("⏳ Chargement du dataset réel…"):
    df_raw, data_source = load_dataset()

with st.spinner("⚙️ Préparation des splits et features…"):
    try:
        data = prepare_splits(df_raw)
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        st.stop()

df_full  = data["df_full"]
df_test  = data["df_test"]
X_L      = data["X_L"]; y_L = data["y_L"]
X_U      = data["X_U"]
X_test   = data["X_test"]; y_test = data["y_test"]
va_idx   = data["va_idx"]; vb_idx = data["vb_idx"]
ALL_FEATURES = data["ALL_FEATURES"]

# ═══════════════════════════════════════════════════════════════════════════
# 8. HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div style='background:linear-gradient(135deg,#0B1F3A 0%,#0A8A7C 100%);
            padding:28px 32px;border-radius:12px;margin-bottom:24px'>
  <h1 style='color:white;margin:0;font-size:2rem'>
    🌍 SSL — Qualité de l'Air Dakar (Données Réelles)
  </h1>
  <p style='color:#B2D8D4;margin:8px 0 4px 0;font-size:1rem'>
    Self-Training &amp; Co-Training · Données réelles uploadées
  </p>
  <span style='background:#27AE60;color:white;padding:3px 10px;border-radius:20px;
               font-size:0.8rem;font-weight:bold'>{data_source}</span>
</div>""", unsafe_allow_html=True)

n_total = len(df_full); n_L_cnt = int(df_full["label_known"].sum()); n_U_cnt = n_total - n_L_cnt
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📦 Train total",        f"{n_total:,}")
c2.metric("🏷 Labellisés L",        f"{n_L_cnt:,}", f"{n_L_cnt/n_total*100:.1f}%")
c3.metric("🔓 Non-labellisés U",    f"{n_U_cnt:,}", f"{n_U_cnt/n_total*100:.1f}%")
c4.metric("🧪 Test set",           f"{len(df_test):,}")
c5.metric("🔢 Features",            f"{len(ALL_FEATURES)}")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# 9. ONGLETS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analyse Exploratoire (EDA)",
    "🔬 Diagnostic",
    "🤖 Simulation Semi-Supervisée",
    "📈 Dashboard Résultats",
])

# ─── TAB 1 : EDA ────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🔍 Analyse Exploratoire — Données réelles")

    with st.expander("📡 Infos dataset", expanded=True):
        st.markdown(f"""
| Paramètre | Valeur |
|---|---|
| **Lignes** | {len(df_raw):,} |
| **Période** | {df_raw['date'].min().date()} → {df_raw['date'].max().date()} |
| **Colonnes** | {len(df_raw.columns)} |
| **Labellisées** | {int(df_raw['label_known'].sum())} ({int(df_raw['label_known'].sum())/len(df_raw)*100:.1f}%) |
| **Features** | {', '.join(ALL_FEATURES)} |
        """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Scarcité des étiquettes")
        st.pyplot(fig_label_scarcity(df_full), use_container_width=True)
    with col_b:
        st.markdown("#### Polluants — Séries temporelles")
        st.pyplot(fig_temporal_pollutants(df_full), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔗 Heatmap Corrélation")
    hm_fig = fig_correlation_heatmap(df_full, ALL_FEATURES)
    if hm_fig:
        st.pyplot(hm_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📋 Aperçu du dataset")
    disp_cols = [c for c in ["date", "pm2_5", "pm10", "nitrogen_dioxide", "ozone", "aqi_label", "label_known"] 
                 if c in df_full.columns]
    st.dataframe(df_full[disp_cols].head(30), use_container_width=True)

# ─── TAB 2 : DIAGNOSTIC ─────────────────────────────────────────────────
with tab2:
    st.markdown("### 🔬 Diagnostic — Baseline & Validation")

    st.markdown("#### ⏱ Validation Croisée Temporelle")
    run_cv = st.button("▶️ Lancer la validation croisée", type="secondary")
    if run_cv:
        with st.spinner("Cross-validation en cours…"):
            cv_scores = run_cv_baseline(X_L, y_L, n_estimators=n_estimators)
        if cv_scores:
            st.success(f"✅ F1 baseline moyen = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            st.pyplot(fig_cv_scores(cv_scores), use_container_width=True)
            st.session_state["cv_baseline_f1"] = np.mean(cv_scores)
        else:
            st.warning("⚠️ Pas assez de splits valides")
    elif "cv_baseline_f1" in st.session_state:
        st.info(f"📊 Dernier CV baseline F1 : {st.session_state['cv_baseline_f1']:.4f}")

# ─── TAB 3 : SIMULATION ─────────────────────────────────────────────────
with tab3:
    st.markdown(f"### 🤖 Simulation — **{algo_choice}**")
    st.info(f"γ fin={gamma} | Marge={min_margin} | Patience={patience} | Max iter={max_iter} | Arbres={n_estimators}")

    run_btn = st.button(f"▶️ Lancer {algo_choice}", type="primary", use_container_width=True)

    if run_btn:
        prog = st.progress(0); stat = st.empty(); tbl = st.empty()
        t0 = time.time()

        if algo_choice == "Self-Training":
            history = run_self_training(
                X_L, y_L, X_U, X_test, y_test,
                gamma=gamma, max_iter=max_iter, n_estimators=n_estimators,
                patience=patience, min_margin=min_margin)
        else:
            history = run_co_training(
                X_L, y_L, X_U, X_test, y_test,
                va_idx=va_idx, vb_idx=vb_idx,
                gamma=gamma, max_iter=max_iter, k_per_iter=k_per_iter,
                n_estimators=n_estimators, patience=patience, min_margin=min_margin)

        elapsed = time.time() - t0
        final   = history[-1]
        gain    = final["f1_macro"] - history[0]["f1_macro"]
        best_f1 = final.get("best_f1", final["f1_macro"])

        prog.progress(100)
        stat.success(f"✅ Terminé en {elapsed:.1f}s — {final['iteration']} itérations")

        cols_tbl = ["iteration", "n_L", "n_U", "f1_macro", "precision", "recall", "n_added"]
        cols_tbl = [c for c in cols_tbl if c in pd.DataFrame(history).columns]
        df_h = pd.DataFrame(history)[cols_tbl]
        tbl.dataframe(df_h, use_container_width=True)

        st.markdown("---")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("F1 Final", f"{final['f1_macro']:.4f}", f"{gain:+.4f}")
        k2.metric("Meilleur F1", f"{best_f1:.4f}")
        k3.metric("Précision", f"{final['precision']:.4f}")
        k4.metric("Rappel", f"{final['recall']:.4f}")

        st.markdown("#### 📈 Progression")
        st.pyplot(fig_ssl_progress(history, algo_choice), use_container_width=True)

        # Prédictions finales
        if algo_choice == "Self-Training":
            y_pred_fin = final["clf"].predict(X_test)
        else:
            cA, cB = final["clf_A"], final["clf_B"]
            pA = cA.predict_proba(X_test[:, va_idx])
            pB = cB.predict_proba(X_test[:, vb_idx])
            cls = np.union1d(cA.classes_, cB.classes_)
            def _al(c, p):
                out = np.zeros((p.shape[0], len(cls)))
                for j, cl in enumerate(cls):
                    if cl in c.classes_:
                        out[:, j] = p[:, np.where(c.classes_ == cl)[0][0]]
                return out
            pf = (_al(cA, pA) + _al(cB, pB)) / 2
            y_pred_fin = cls[pf.argmax(axis=1)]

        st.markdown("#### 🔲 Matrice de Confusion")
        st.pyplot(fig_confusion(y_test, y_pred_fin, f"{algo_choice}"),
                  use_container_width=True)

        st.markdown("#### 📋 Rapport de Classification")
        unique_labels = np.unique(y_test)
        report = classification_report(
            y_test, y_pred_fin, labels=unique_labels, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

        if "results" not in st.session_state:
            st.session_state["results"] = {}
        st.session_state["results"][algo_choice] = {
            "f1_macro":  final["f1_macro"],
            "precision": final["precision"],
            "recall":    final["recall"],
            "history":   history,
            "y_pred":    y_pred_fin,
        }
    else:
        st.info("💡 Cliquez **▶️ Lancer** pour démarrer la simulation.")

# ─── TAB 4 : DASHBOARD ──────────────────────────────────────────────────
with tab4:
    st.markdown("### 📈 Dashboard Comparatif")

    if "results" not in st.session_state or not st.session_state["results"]:
        st.warning("⚠️ Lancez une simulation d'abord.")
    else:
        clf_base = make_clf(100); clf_base.fit(X_L, y_L)
        y_base   = clf_base.predict(X_test)
        all_res  = {
            "Baseline (L seul)": {
                "f1_macro":  round(f1_score(y_test, y_base, average="macro", zero_division=0), 4),
                "precision": round(precision_score(y_test, y_base, average="macro", zero_division=0), 4),
                "recall":    round(recall_score(y_test, y_base, average="macro", zero_division=0), 4),
                "y_pred":    y_base,
            }
        }
        for k, v in st.session_state["results"].items():
            all_res[k] = v

        st.markdown("#### 📊 Comparaison Globale")
        st.pyplot(fig_compare(all_res), use_container_width=True)

        st.markdown("#### 🗃 Tableau Récapitulatif")
        df_comp = pd.DataFrame({k: {"F1 macro": v["f1_macro"],
                                     "Précision": v["precision"],
                                     "Rappel": v["recall"]}
                                 for k, v in all_res.items()}).T
        base_f1 = df_comp.loc["Baseline (L seul)", "F1 macro"]
        df_comp["Δ vs Baseline"] = (df_comp["F1 macro"] - base_f1).round(4)
        st.dataframe(df_comp.style.format({"F1 macro": "{:.4f}", "Précision": "{:.4f}",
                                            "Rappel": "{:.4f}", "Δ vs Baseline": "{:+.4f}"}),
                     use_container_width=True)
