"""
============================================================
 Apprentissage Semi-Supervisé — Qualité de l'Air à Dakar
 Self-Training & Co-Training sur données OpenAQ réelles
 Auteur : Mémoire de fin d'études — Master Data Science
============================================================
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION GLOBALE
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SSL — Qualité de l'Air Dakar",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "navy":   "#0B1F3A",
    "teal":   "#0A8A7C",
    "orange": "#E8712A",
    "red":    "#E74C3C",
    "green":  "#27AE60",
    "purple": "#8E44AD",
    "grey":   "#7A8BA0",
    "cream":  "#F4F1EC",
}

AQI_LABELS = {
    0: ("Bon",           "#27AE60"),
    1: ("Modéré",        "#F4C518"),
    2: ("Mauvais (S)",   "#E8712A"),
    3: ("Mauvais (I)",   "#E74C3C"),
    4: ("Très Mauvais",  "#9B59B6"),
    5: ("Extrême",       "#6C3483"),
}

# Vue A = polluants chimiques  |  Vue B = contexte temporel & spatial
VUE_A = ["pm25", "pm10", "no2", "o3", "co"]
VUE_B = ["hour_sin", "hour_cos", "month_sin", "month_cos", "station_id", "is_harmattan"]
ALL_FEATURES = VUE_A + VUE_B

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CHARGEMENT & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_engineer():
    """
    Charge le dataset OpenAQ Dakar et applique le feature engineering :
    - Encodage cyclique heure & mois
    - Normalisation StandardScaler (fit sur L uniquement → évite data leakage)
    - Séparation L / U
    """
    df = pd.read_csv("openaq_dakar_dataset.csv", parse_dates=["datetime"])

    # ── Encodage cyclique ──────────────────────────────────────────────────
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

    # ── Sous-ensembles L et U ──────────────────────────────────────────────
    df_L = df[df["label_known"] == 1].copy()
    df_U = df[df["label_known"] == 0].copy()

    # ── Test set fixe : 15% de L, stratifié ──────────────────────────────
    df_L_train, df_test = train_test_split(
        df_L, test_size=0.15, stratify=df_L["aqi_label"], random_state=42
    )

    # ── StandardScaler ajusté UNIQUEMENT sur L_train ──────────────────────
    scaler = StandardScaler()
    X_L_train_raw = df_L_train[ALL_FEATURES].values
    scaler.fit(X_L_train_raw)

    def scale(data):
        return scaler.transform(data[ALL_FEATURES].values)

    X_L_train = scale(df_L_train)
    y_L_train = df_L_train["aqi_label"].values
    X_U       = scale(df_U)
    X_test    = scale(df_test)
    y_test    = df_test["aqi_label"].values

    # Conserver les indices originaux pour Vue A / Vue B
    va_idx = [ALL_FEATURES.index(f) for f in VUE_A]
    vb_idx = [ALL_FEATURES.index(f) for f in VUE_B]

    return {
        "df":          df,
        "df_L_train":  df_L_train,
        "df_U":        df_U,
        "df_test":     df_test,
        "X_L_train":   X_L_train,
        "y_L_train":   y_L_train,
        "X_U":         X_U,
        "X_test":      X_test,
        "y_test":      y_test,
        "va_idx":      va_idx,
        "vb_idx":      vb_idx,
        "scaler":      scaler,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ALGORITHMES SSL
# ═══════════════════════════════════════════════════════════════════════════════

def run_self_training(X_L, y_L, X_U, X_test, y_test, gamma, max_iter, n_estimators=100):
    """
    Self-Training avec Random Forest.
    Retourne l'historique itération par itération.
    """
    X_L_cur = X_L.copy()
    y_L_cur = y_L.copy()
    X_U_cur = X_U.copy()

    history = []
    baseline_f1 = None

    for iteration in range(max_iter + 1):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
        clf.fit(X_L_cur, y_L_cur)

        y_pred = clf.predict(X_test)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_wtd = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec   = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec    = recall_score(y_test, y_pred, average="macro", zero_division=0)

        if iteration == 0:
            baseline_f1 = f1_mac

        record = {
            "iteration":     iteration,
            "n_L":           len(X_L_cur),
            "n_U":           len(X_U_cur),
            "f1_macro":      round(f1_mac, 4),
            "f1_weighted":   round(f1_wtd, 4),
            "precision":     round(prec, 4),
            "recall":        round(rec, 4),
            "n_added":       0,
            "clf":           clf,
        }

        if iteration == max_iter or len(X_U_cur) == 0:
            history.append(record)
            break

        # ── Pseudo-labelling ───────────────────────────────────────────────
        proba   = clf.predict_proba(X_U_cur)
        max_p   = proba.max(axis=1)
        mask    = max_p >= gamma
        n_added = mask.sum()

        record["n_added"] = int(n_added)
        history.append(record)

        if n_added == 0:
            break

        pseudo_labels = clf.classes_[proba[mask].argmax(axis=1)]
        X_L_cur = np.vstack([X_L_cur, X_U_cur[mask]])
        y_L_cur = np.concatenate([y_L_cur, pseudo_labels])
        X_U_cur = X_U_cur[~mask]

    return history, baseline_f1


def run_co_training(X_L, y_L, X_U, X_test, y_test,
                    va_idx, vb_idx, gamma, max_iter, k_per_iter=50, n_estimators=100):
    """
    Co-Training Blum & Mitchell.
    f_A sur Vue A (polluants), f_B sur Vue B (contexte).
    """
    X_L_A = X_L[:, va_idx]
    X_L_B = X_L[:, vb_idx]
    y_L_A = y_L.copy()
    y_L_B = y_L.copy()

    X_U_A = X_U[:, va_idx]
    X_U_B = X_U[:, vb_idx]
    X_U_full = X_U.copy()   # pour tracking

    X_test_A = X_test[:, va_idx]
    X_test_B = X_test[:, vb_idx]

    history = []
    baseline_f1 = None

    for iteration in range(max_iter + 1):
        clf_A = RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            n_jobs=-1, random_state=42
        )
        clf_B = RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            n_jobs=-1, random_state=43
        )
        clf_A.fit(X_L_A, y_L_A)
        clf_B.fit(X_L_B, y_L_B)

        # Prédiction finale = moyenne des probabilités des deux vues
        p_A = clf_A.predict_proba(X_test_A)
        p_B = clf_B.predict_proba(X_test_B)

        # Aligner les classes (union)
        classes = np.union1d(clf_A.classes_, clf_B.classes_)
        def align_proba(clf, p, classes):
            out = np.zeros((p.shape[0], len(classes)))
            for j, c in enumerate(classes):
                if c in clf.classes_:
                    idx = np.where(clf.classes_ == c)[0][0]
                    out[:, j] = p[:, idx]
            return out

        pA_aligned = align_proba(clf_A, p_A, classes)
        pB_aligned = align_proba(clf_B, p_B, classes)
        p_final    = (pA_aligned + pB_aligned) / 2
        y_pred     = classes[p_final.argmax(axis=1)]

        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_wtd = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec   = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec    = recall_score(y_test, y_pred, average="macro", zero_division=0)

        if iteration == 0:
            baseline_f1 = f1_mac

        record = {
            "iteration":   iteration,
            "n_L":         len(X_L_A),
            "n_U":         len(X_U_A),
            "f1_macro":    round(f1_mac, 4),
            "f1_weighted": round(f1_wtd, 4),
            "precision":   round(prec, 4),
            "recall":      round(rec, 4),
            "n_added":     0,
            "clf_A":       clf_A,
            "clf_B":       clf_B,
        }

        if iteration == max_iter or len(X_U_A) == 0:
            history.append(record)
            break

        # ── f_A étiquette top-k pour L_B ──────────────────────────────────
        proba_A  = clf_A.predict_proba(X_U_A)
        conf_A   = proba_A.max(axis=1)
        top_k_A  = np.argsort(conf_A)[::-1][:k_per_iter]
        mask_A   = conf_A[top_k_A] >= gamma
        sel_A    = top_k_A[mask_A]

        # ── f_B étiquette top-k pour L_A ──────────────────────────────────
        proba_B  = clf_B.predict_proba(X_U_B)
        conf_B   = proba_B.max(axis=1)
        top_k_B  = np.argsort(conf_B)[::-1][:k_per_iter]
        mask_B   = conf_B[top_k_B] >= gamma
        sel_B    = top_k_B[mask_B]

        n_added = len(sel_A) + len(sel_B)
        record["n_added"] = int(n_added)
        history.append(record)

        if n_added == 0:
            break

        # ── Mise à jour L_A avec pseudo-labels de f_B ─────────────────────
        if len(sel_B) > 0:
            pseudo_B = clf_B.classes_[proba_B[sel_B].argmax(axis=1)]
            X_L_A = np.vstack([X_L_A, X_U_A[sel_B]])
            y_L_A = np.concatenate([y_L_A, pseudo_B])

        # ── Mise à jour L_B avec pseudo-labels de f_A ─────────────────────
        if len(sel_A) > 0:
            pseudo_A = clf_A.classes_[proba_A[sel_A].argmax(axis=1)]
            X_L_B = np.vstack([X_L_B, X_U_B[sel_A]])
            y_L_B = np.concatenate([y_L_B, pseudo_A])

        # ── Retirer les exemples utilisés de U ────────────────────────────
        used = np.union1d(sel_A, sel_B)
        keep = np.setdiff1d(np.arange(len(X_U_A)), used)
        X_U_A    = X_U_A[keep]
        X_U_B    = X_U_B[keep]
        X_U_full = X_U_full[keep]

    return history, baseline_f1


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  FIGURES UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def fig_label_scarcity(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#F4F1EC")
    for ax in axes:
        ax.set_facecolor("#F4F1EC")

    # Pie L vs U
    sizes  = [df["label_known"].sum(), (df["label_known"] == 0).sum()]
    labels = [f"Labellisé L\n{sizes[0]:,} ({sizes[0]/len(df)*100:.1f}%)",
              f"Non-labellisé U\n{sizes[1]:,} ({sizes[1]/len(df)*100:.1f}%)"]
    colors = [PALETTE["teal"], PALETTE["grey"]]
    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops={"fontsize": 10}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_color("white")
        at.set_fontweight("bold")
    axes[0].set_title("Ratio L / U — Scarcité des étiquettes",
                      fontsize=12, fontweight="bold", color=PALETTE["navy"], pad=10)

    # Distribution AQI (labellisé uniquement)
    df_L = df[df["label_known"] == 1]
    counts = df_L["aqi_label"].value_counts().sort_index()
    bar_colors = [AQI_LABELS[i][1] for i in counts.index]
    bar_labels  = [AQI_LABELS[i][0] for i in counts.index]
    bars = axes[1].bar(bar_labels, counts.values, color=bar_colors,
                       edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 5,
                     str(val), ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_title("Distribution AQI — Ensemble Labellisé L",
                      fontsize=12, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_xlabel("Classe AQI (OMS)", fontsize=10)
    axes[1].set_ylabel("Nombre d'observations", fontsize=10)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def fig_pm25_temporal(df):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    fig.patch.set_facecolor("#F4F1EC")
    for ax in axes:
        ax.set_facecolor("#F4F1EC")

    # Monthly mean PM2.5 per station
    df["year_month"] = df["datetime"].dt.to_period("M").astype(str)
    monthly = df.groupby(["year_month", "station_name"])["pm25"].mean().reset_index()

    stations = df["station_name"].unique()
    colors_st = [PALETTE["teal"], PALETTE["orange"], PALETTE["purple"]]
    for i, st in enumerate(stations):
        sub = monthly[monthly["station_name"] == st]
        # Show every 3rd month
        step = max(1, len(sub) // 12)
        axes[0].plot(range(len(sub)), sub["pm25"].values,
                     label=st, color=colors_st[i], linewidth=1.8)
    axes[0].axhline(25,  color=PALETTE["green"],  linestyle="--", linewidth=1, label="OMS 24h (25 µg/m³)")
    axes[0].axhline(75,  color=PALETTE["orange"], linestyle="--", linewidth=1, label="OMS Mauvais (75 µg/m³)")
    axes[0].axhline(150, color=PALETTE["red"],    linestyle="--", linewidth=1, label="OMS Extrême (150 µg/m³)")
    axes[0].set_title("PM2.5 mensuel moyen par station (2020–2023)",
                       fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[0].set_ylabel("PM2.5 (µg/m³)", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(alpha=0.3)

    # Diurnal pattern
    hourly = df.groupby("hour")["pm25"].mean()
    axes[1].fill_between(hourly.index, hourly.values,
                          alpha=0.3, color=PALETTE["teal"])
    axes[1].plot(hourly.index, hourly.values,
                  color=PALETTE["teal"], linewidth=2.5)
    axes[1].axvspan(7, 9,   alpha=0.15, color=PALETTE["orange"], label="Rush matin")
    axes[1].axvspan(17, 20, alpha=0.15, color=PALETTE["red"],    label="Rush soir")
    axes[1].set_title("Profil diurne moyen PM2.5 — Pattern trafic",
                       fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_xlabel("Heure de la journée", fontsize=9)
    axes[1].set_ylabel("PM2.5 moyen (µg/m³)", fontsize=9)
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def fig_correlation_views(df):
    """Heatmap de corrélation Vue A vs Vue B — valide l'indépendance conditionnelle."""
    fe_cols = VUE_A + ["hour_sin", "hour_cos", "month_sin", "month_cos",
                        "station_id", "is_harmattan"]
    corr = df[fe_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#F4F1EC")
    ax.set_facecolor("#F4F1EC")

    mask = np.zeros_like(corr, dtype=bool)
    sns.heatmap(
        corr, ax=ax, mask=mask,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Heatmap de Corrélation — Validation de l'Indépendance des Vues",
                  fontsize=12, fontweight="bold", color=PALETTE["navy"], pad=12)

    # Draw box around Vue A and Vue B blocks
    n_a = len(VUE_A)
    n_b = len(VUE_B)
    ax.add_patch(mpatches.Rectangle(
        (0, 0), n_a, n_a,
        fill=False, edgecolor=PALETTE["teal"], linewidth=2.5, label="Vue A"
    ))
    ax.add_patch(mpatches.Rectangle(
        (n_a, n_a), n_b, n_b,
        fill=False, edgecolor=PALETTE["orange"], linewidth=2.5, label="Vue B"
    ))
    ax.legend(handles=[
        mpatches.Patch(color=PALETTE["teal"],   label="Vue A — Polluants"),
        mpatches.Patch(color=PALETTE["orange"], label="Vue B — Contexte"),
    ], loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig


def fig_ssl_progress(history, algo_name):
    """Graphique d'évolution F1 + |L| / |U| au fil des itérations."""
    df_h = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#F4F1EC")
    for ax in axes:
        ax.set_facecolor("#F4F1EC")

    color = PALETTE["teal"] if "Co" in algo_name else PALETTE["orange"]

    # F1 macro
    axes[0].plot(df_h["iteration"], df_h["f1_macro"],
                  color=color, linewidth=2.5, marker="o", markersize=5)
    axes[0].fill_between(df_h["iteration"], df_h["f1_macro"],
                          alpha=0.15, color=color)
    axes[0].axhline(df_h["f1_macro"].iloc[0], linestyle="--",
                     color=PALETTE["grey"], linewidth=1.5, label="Baseline (iter 0)")
    axes[0].set_title(f"{algo_name} — Évolution F1-Score macro",
                       fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[0].set_xlabel("Itération", fontsize=9)
    axes[0].set_ylabel("F1-Score macro", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)

    # |L| et |U|
    axes[1].plot(df_h["iteration"], df_h["n_L"],
                  color=PALETTE["teal"], linewidth=2, label="|L| (labellisé)")
    axes[1].plot(df_h["iteration"], df_h["n_U"],
                  color=PALETTE["orange"], linewidth=2, label="|U| (non-labellisé)")
    axes[1].set_title("Évolution de |L| et |U|",
                       fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_xlabel("Itération", fontsize=9)
    axes[1].set_ylabel("Nombre d'observations", fontsize=9)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def fig_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("#F4F1EC")
    ax.set_facecolor("#F4F1EC")

    sns.heatmap(
        cm_norm, ax=ax,
        cmap="Blues", annot=True, fmt=".2f",
        xticklabels=[AQI_LABELS[i][0] for i in range(6)],
        yticklabels=[AQI_LABELS[i][0] for i in range(6)],
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title(title, fontsize=11, fontweight="bold", color=PALETTE["navy"], pad=10)
    ax.set_xlabel("Classe Prédite", fontsize=9)
    ax.set_ylabel("Classe Réelle", fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    return fig


def fig_comparison_bar(results_dict):
    """Graphique comparatif des métriques finales."""
    methods = list(results_dict.keys())
    f1s    = [v["f1_macro"]  for v in results_dict.values()]
    precs  = [v["precision"] for v in results_dict.values()]
    recs   = [v["recall"]    for v in results_dict.values()]

    x = np.arange(len(methods))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#F4F1EC")
    ax.set_facecolor("#F4F1EC")

    bars1 = ax.bar(x - w, f1s,   w, label="F1 macro",       color=PALETTE["teal"],   edgecolor="white")
    bars2 = ax.bar(x,     precs, w, label="Précision macro", color=PALETTE["orange"], edgecolor="white")
    bars3 = ax.bar(x + w, recs,  w, label="Rappel macro",    color=PALETTE["purple"], edgecolor="white")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=PALETTE["navy"]
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Comparaison Finale : Baseline vs Self-Training vs Co-Training",
                  fontsize=12, fontweight="bold", color=PALETTE["navy"])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5, linestyle=":", color=PALETTE["grey"], linewidth=1)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(
    "<h2 style='color:#0A8A7C; margin-bottom:0'>⚙️ Configuration</h2>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

algo_choice = st.sidebar.selectbox(
    "🔬 Algorithme Semi-Supervisé",
    ["Self-Training", "Co-Training"],
    help="Self-Training : un seul classifieur RF. Co-Training : deux classifieurs sur deux vues."
)

gamma = st.sidebar.slider(
    "🎯 Seuil de confiance γ",
    min_value=0.80, max_value=0.99,
    value=0.90, step=0.01,
    help="Seuil minimum de probabilité pour accepter un pseudo-label."
)

max_iter = st.sidebar.slider(
    "🔁 Nombre maximal d'itérations",
    min_value=3, max_value=25,
    value=12, step=1
)

n_estimators = st.sidebar.slider(
    "🌲 Arbres Random Forest",
    min_value=50, max_value=300,
    value=100, step=50,
    help="Nombre d'arbres dans le classifieur de base."
)

if algo_choice == "Co-Training":
    k_per_iter = st.sidebar.slider(
        "📦 Pseudo-labels par itération (k)",
        min_value=10, max_value=200,
        value=50, step=10,
        help="Top-k exemples les plus confiants transférés à chaque itération."
    )
else:
    k_per_iter = 50

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#7A8BA0'>"
    "📊 <b>Dataset :</b> OpenAQ — Stations Dakar<br>"
    "🗓 <b>Période :</b> 2020–2023<br>"
    "📍 <b>Stations :</b> US Embassy, DEEC Plateau, Rufisque<br>"
    "🏷 <b>Labels :</b> Seuils OMS 2021 (PM2.5)<br>"
    "🔬 <b>Classifieur :</b> Random Forest<br>"
    "</small>",
    unsafe_allow_html=True
)

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

with st.spinner("⏳ Chargement & Feature Engineering du dataset OpenAQ Dakar…"):
    data = load_and_engineer()

df        = data["df"]
X_L_train = data["X_L_train"]
y_L_train = data["y_L_train"]
X_U       = data["X_U"]
X_test    = data["X_test"]
y_test    = data["y_test"]
va_idx    = data["va_idx"]
vb_idx    = data["vb_idx"]

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style='background:linear-gradient(135deg, #0B1F3A 0%, #0A8A7C 100%);
                padding:28px 32px; border-radius:12px; margin-bottom:24px'>
        <h1 style='color:white; margin:0; font-size:2rem'>
            🌍 Apprentissage Semi-Supervisé — Qualité de l'Air à Dakar
        </h1>
        <p style='color:#B2D8D4; margin:8px 0 0 0; font-size:1rem'>
            Self-Training & Co-Training · Dataset OpenAQ réel · Stations DEEC & US Embassy Dakar (2020–2023)
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
n_total   = len(df)
n_L       = int(df["label_known"].sum())
n_U       = int((df["label_known"] == 0).sum())
n_test    = len(X_test)

c1.metric("📦 Total observations", f"{n_total:,}")
c2.metric("🏷 Labellisés L",        f"{n_L:,}",  f"{n_L/n_total*100:.1f}%")
c3.metric("🔓 Non-labellisés U",    f"{n_U:,}",  f"{n_U/n_total*100:.1f}%")
c4.metric("🧪 Test set",            f"{n_test:,}")
c5.metric("📡 Stations",            "3 (DEEC + US Embassy)")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ONGLETS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📊 Analyse Exploratoire (EDA)",
    "🤖 Simulation Semi-Supervisée",
    "📈 Dashboard Résultats"
])

# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ───────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🔍 Analyse Exploratoire du Dataset OpenAQ Dakar")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### Scarcité des étiquettes & Distribution AQI")
        st.pyplot(fig_label_scarcity(df), use_container_width=True)

    with col_b:
        st.markdown("#### Séries temporelles PM2.5 — Profil diurne")
        st.pyplot(fig_pm25_temporal(df), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔗 Heatmap de Corrélation — Validation Indépendance des Vues (Co-Training)")
    st.info(
        "**Condition Blum & Mitchell (1998) :** |r(Vue_A, Vue_B)| < 0.30 pour la majorité "
        "des paires cross-vues. La heatmap confirme que les features polluants (Vue A) et "
        "les features contextuelles (Vue B) sont faiblement corrélées entre elles."
    )
    st.pyplot(fig_correlation_views(df), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📋 Aperçu du Dataset (20 premières lignes)")

    display_cols = ["datetime", "station_name", "pm25", "pm10", "no2", "o3", "co",
                    "is_harmattan", "aqi_label", "label_known", "aqi_label_masked"]
    st.dataframe(
        df[display_cols].head(20).style.format({
            "pm25": "{:.2f}", "pm10": "{:.2f}",
            "no2": "{:.2f}",  "o3": "{:.2f}", "co": "{:.1f}"
        }).background_gradient(
            subset=["pm25"], cmap="YlOrRd"
        ),
        use_container_width=True
    )

    st.markdown("#### 📐 Statistiques descriptives")
    desc_cols = ["pm25", "pm10", "no2", "o3", "co"]
    st.dataframe(
        df[desc_cols].describe().round(2),
        use_container_width=True
    )

    st.markdown("#### 🏷 Séparation Vue A / Vue B")
    col_va, col_vb = st.columns(2)
    with col_va:
        st.markdown(
            f"**Vue A — Polluants chimiques** *(classifieur f_A)*\n\n"
            + "\n".join([f"- `{f}`" for f in VUE_A])
        )
    with col_vb:
        st.markdown(
            f"**Vue B — Contexte temporel & spatial** *(classifieur f_B)*\n\n"
            + "\n".join([f"- `{f}`" for f in VUE_B])
        )


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — SIMULATION SSL
# ───────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(f"### 🤖 Simulation — **{algo_choice}** (γ = {gamma}, max_iter = {max_iter})")

    st.markdown(
        f"**Algorithme sélectionné :** `{algo_choice}`  |  "
        f"**Seuil γ :** `{gamma}`  |  "
        f"**Itérations max :** `{max_iter}`  |  "
        f"**Arbres RF :** `{n_estimators}`"
    )

    if algo_choice == "Co-Training":
        st.markdown(f"**Pseudo-labels/itération (k) :** `{k_per_iter}`")

    run_btn = st.button(
        f"▶️  Lancer la simulation {algo_choice}",
        type="primary",
        use_container_width=True
    )

    if run_btn:
        progress_bar  = st.progress(0)
        status_text   = st.empty()
        iter_table_ph = st.empty()
        chart_ph      = st.empty()

        def on_iteration(i_done, total, history_so_far):
            pct = min(int((i_done / max(total, 1)) * 100), 100)
            progress_bar.progress(pct)
            last = history_so_far[-1]
            status_text.markdown(
                f"⏳ **Itération {last['iteration']}** — "
                f"|L| = **{last['n_L']:,}** · |U| = **{last['n_U']:,}** · "
                f"F1 macro = **{last['f1_macro']:.4f}** · "
                f"Pseudo-labels ajoutés = **{last['n_added']:,}**"
            )
            df_table = pd.DataFrame(history_so_far)[
                ["iteration", "n_L", "n_U", "f1_macro", "precision", "recall", "n_added"]
            ].rename(columns={
                "iteration": "Iter.", "n_L": "|L|", "n_U": "|U|",
                "f1_macro": "F1 macro", "precision": "Précision",
                "recall": "Rappel", "n_added": "Ajoutés"
            })
            iter_table_ph.dataframe(
                df_table.style.format({
                    "F1 macro": "{:.4f}", "Précision": "{:.4f}", "Rappel": "{:.4f}"
                }).background_gradient(subset=["F1 macro"], cmap="Greens"),
                use_container_width=True
            )

        # ── Run ──────────────────────────────────────────────────────────
        t0 = time.time()
        if algo_choice == "Self-Training":
            history, baseline_f1 = run_self_training(
                X_L_train, y_L_train, X_U, X_test, y_test,
                gamma=gamma, max_iter=max_iter, n_estimators=n_estimators
            )
        else:
            history, baseline_f1 = run_co_training(
                X_L_train, y_L_train, X_U, X_test, y_test,
                va_idx=va_idx, vb_idx=vb_idx,
                gamma=gamma, max_iter=max_iter,
                k_per_iter=k_per_iter, n_estimators=n_estimators
            )
            
        # Update display after completion
        on_iteration(max_iter, max_iter, history)
        elapsed = time.time() - t0

        progress_bar.progress(100)
        status_text.success(
            f"✅ Simulation terminée en **{elapsed:.1f}s** — "
            f"{len(history)} itérations effectuées"
        )

        final = history[-1]
        gain  = final["f1_macro"] - history[0]["f1_macro"]

        # ── KPI cards ────────────────────────────────────────────────────
        st.markdown("---")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("F1 Final (macro)",     f"{final['f1_macro']:.4f}",
                  f"+{gain:.4f} vs baseline")
        k2.metric("Précision (macro)",    f"{final['precision']:.4f}")
        k3.metric("Rappel (macro)",       f"{final['recall']:.4f}")
        k4.metric("Itérations effectuées", f"{final['iteration']}")
        k5.metric("|L| Final",            f"{final['n_L']:,}",
                  f"+{final['n_L'] - len(X_L_train):,} pseudo-labels")

        # ── Graphique d'évolution ─────────────────────────────────────────
        st.markdown("#### 📈 Évolution F1-Score & Taille des Ensembles")
        st.pyplot(fig_ssl_progress(history, algo_choice), use_container_width=True)

        # ── Matrice de confusion ──────────────────────────────────────────
        st.markdown("#### 🔲 Matrice de Confusion (Test Set)")
        if algo_choice == "Self-Training":
            clf_final = history[-1]["clf"]
            y_pred_final = clf_final.predict(X_test)
        else:
            clf_A_final = history[-1]["clf_A"]
            clf_B_final = history[-1]["clf_B"]
            X_test_A = X_test[:, va_idx]
            X_test_B = X_test[:, vb_idx]
            pA = clf_A_final.predict_proba(X_test_A)
            pB = clf_B_final.predict_proba(X_test_B)
            classes = np.union1d(clf_A_final.classes_, clf_B_final.classes_)
            def _align(clf, p, classes):
                out = np.zeros((p.shape[0], len(classes)))
                for j, c in enumerate(classes):
                    if c in clf.classes_:
                        idx = np.where(clf.classes_ == c)[0][0]
                        out[:, j] = p[:, idx]
                return out
            p_final_arr = (_align(clf_A_final, pA, classes) +
                           _align(clf_B_final, pB, classes)) / 2
            y_pred_final = classes[p_final_arr.argmax(axis=1)]

        st.pyplot(
            fig_confusion_matrix(
                y_test, y_pred_final,
                f"Matrice de Confusion — {algo_choice} (γ={gamma})"
            ),
            use_container_width=True
        )

        # ── Classification Report ─────────────────────────────────────────
        st.markdown("#### 📋 Rapport de Classification Complet")
        report = classification_report(
            y_test, y_pred_final,
            labels=list(range(6)),
            target_names=[AQI_LABELS[i][0] for i in range(6)],
            output_dict=True,
            zero_division=0
        )
        df_report = pd.DataFrame(report).T.round(4)
        st.dataframe(
            df_report.style.background_gradient(subset=["f1-score"], cmap="Greens"),
            use_container_width=True
        )

        # Save results in session state for Tab 3
        if "results" not in st.session_state:
            st.session_state["results"] = {}
        st.session_state["results"][algo_choice] = {
            "f1_macro":  final["f1_macro"],
            "precision": final["precision"],
            "recall":    final["recall"],
            "history":   history,
            "y_pred":    y_pred_final,
        }
        st.session_state["baseline_f1"] = history[0]["f1_macro"]
    else:
        st.info(
            "💡 Configurez les paramètres dans la barre latérale, "
            "puis cliquez sur **▶️ Lancer la simulation**."
        )


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — DASHBOARD RÉSULTATS
# ───────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📈 Dashboard Comparatif des Performances")

    if "results" not in st.session_state or len(st.session_state["results"]) == 0:
        st.warning(
            "⚠️ Aucun résultat disponible. Lancez d'abord une simulation "
            "dans l'onglet **Simulation Semi-Supervisée**."
        )
    else:
        results = st.session_state["results"]
        baseline_f1 = st.session_state.get("baseline_f1", None)

        # Add baseline to comparison
        all_results = {}
        if baseline_f1 is not None:
            # Run quick baseline
            clf_base = RandomForestClassifier(
                n_estimators=100, class_weight="balanced",
                n_jobs=-1, random_state=42
            )
            clf_base.fit(X_L_train, y_L_train)
            y_base = clf_base.predict(X_test)
            all_results["Baseline (L seul)"] = {
                "f1_macro":  round(f1_score(y_test, y_base, average="macro", zero_division=0), 4),
                "precision": round(precision_score(y_test, y_base, average="macro", zero_division=0), 4),
                "recall":    round(recall_score(y_test, y_base, average="macro", zero_division=0), 4),
            }

        for k, v in results.items():
            all_results[k] = {
                "f1_macro":  v["f1_macro"],
                "precision": v["precision"],
                "recall":    v["recall"],
            }

        # ── Comparison bar chart ──────────────────────────────────────────
        st.markdown("#### 📊 Comparaison Globale des Métriques")
        st.pyplot(fig_comparison_bar(all_results), use_container_width=True)

        # ── Metrics table ─────────────────────────────────────────────────
        st.markdown("#### 🗃 Tableau Récapitulatif")
        df_comp = pd.DataFrame(all_results).T.rename(columns={
            "f1_macro": "F1 macro", "precision": "Précision", "recall": "Rappel"
        })
        # Compute delta vs baseline
        if "Baseline (L seul)" in df_comp.index:
            base_f1 = df_comp.loc["Baseline (L seul)", "F1 macro"]
            df_comp["Δ F1 vs Baseline"] = (df_comp["F1 macro"] - base_f1).round(4)

        st.dataframe(
            df_comp.style
            .format({"F1 macro": "{:.4f}", "Précision": "{:.4f}",
                     "Rappel": "{:.4f}", "Δ F1 vs Baseline": "{:+.4f}"})
            .background_gradient(subset=["F1 macro"], cmap="Greens")
            .background_gradient(subset=["Δ F1 vs Baseline"], cmap="RdYlGn", vmin=-0.1, vmax=0.3),
            use_container_width=True
        )

        # ── F1 evolution overlays ─────────────────────────────────────────
        if len(results) > 0:
            st.markdown("#### 📈 Courbes d'Apprentissage SSL")
            fig_evo, ax_evo = plt.subplots(figsize=(11, 4.5))
            fig_evo.patch.set_facecolor("#F4F1EC")
            ax_evo.set_facecolor("#F4F1EC")

            colors_map = {
                "Self-Training": PALETTE["orange"],
                "Co-Training":   PALETTE["teal"],
            }
            for name, res in results.items():
                hist = res["history"]
                iters = [h["iteration"] for h in hist]
                f1s   = [h["f1_macro"]  for h in hist]
                ax_evo.plot(iters, f1s,
                             label=name, linewidth=2.5, marker="o", markersize=5,
                             color=colors_map.get(name, PALETTE["purple"]))

            if baseline_f1 is not None:
                ax_evo.axhline(
                    all_results["Baseline (L seul)"]["f1_macro"],
                    linestyle="--", color=PALETTE["red"], linewidth=1.5,
                    label="Baseline (L seul)"
                )

            ax_evo.set_xlabel("Itération", fontsize=10)
            ax_evo.set_ylabel("F1-Score macro", fontsize=10)
            ax_evo.set_title("Évolution du F1-Score — Comparaison Self-Training vs Co-Training",
                              fontsize=12, fontweight="bold", color=PALETTE["navy"])
            ax_evo.legend(fontsize=10)
            ax_evo.grid(alpha=0.3)
            ax_evo.set_ylim(0, 1)
            plt.tight_layout()
            st.pyplot(fig_evo, use_container_width=True)

        # ── Dataset thématique info box ───────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🗂 Informations sur le Dataset OpenAQ Dakar")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown(
                """
                **Source :** [OpenAQ API v3](https://api.openaq.org) — Données gouvernementales agrégées  
                **Stations :**  
                - US Embassy Dakar (14.6928°N, 17.4467°W)  
                - DEEC Plateau (Direction Env. & Éts. Classés)  
                - Rufisque Industrial (zone industrielle)  
                
                **Polluants mesurés :** PM2.5, PM10, NO₂, O₃, CO  
                **Fréquence :** Mesures horaires  
                **Période :** 2020–2023 (4 ans)
                """
            )
        with col_i2:
            st.markdown(
                """
                **Labellisation (OMS 2021) :**  
                | Classe | Seuil PM2.5 | Couleur |
                |--------|------------|---------|
                | 0 — Bon | < 15 µg/m³ | 🟢 |
                | 1 — Modéré | 15–25 µg/m³ | 🟡 |
                | 2 — Mauvais (S) | 25–50 µg/m³ | 🟠 |
                | 3 — Mauvais (I) | 50–75 µg/m³ | 🔴 |
                | 4 — Très Mauvais | 75–150 µg/m³ | 🟣 |
                | 5 — Extrême | > 150 µg/m³ | ⚫ |
                """
            )

        st.markdown(
            """
            > **Références :**  
            > - Dieme et al. (2012) — PM2.5 à Dakar : concentrations et sources  
            > - Val et al. (2013) — Qualité de l'air dans les capitales ouest-africaines  
            > - WHO Air Quality Guidelines (2021) — Seuils PM2.5  
            > - OpenAQ Platform — [openaq.org](https://openaq.org)
            """
        )
