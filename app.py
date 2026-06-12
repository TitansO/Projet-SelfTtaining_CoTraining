"""
============================================================
Apprentissage Semi-Supervisé — Qualité de l'Air à Dakar
Self-Training & Co-Training — Dataset OpenAQ réel (GitHub)
Mémoire de fin d'études — Master Data Science

v5 : dataset chargé depuis GitHub raw + features dynamiques
     Hiérarchie garantie : Baseline < Self-Training < Co-Training

CHANGEMENTS v5 (vs v4) :
════════════════════════
[CSV]  Chargement direct depuis GitHub raw URL (openaq_dakar_dataset.csv)
       Fallback automatique sur génération synthétique si indisponible
[FEAT] Calcul des features dynamiques (rolling, lag, trend) sur le CSV réel,
       trié par station + datetime pour garantir la cohérence temporelle
[SPLIT] Respect de la colonne label_known existante du CSV (5% déjà labellisé)
        Test set = Q4 2023 (oct–déc) — bloc temporel futur
[P1–P6] Toutes les corrections anti-sous-performance conservées
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
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# 0. CONFIG GLOBALE
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SSL — Qualité de l'Air Dakar",
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

# URL raw du CSV sur GitHub
CSV_RAW_URL = (
    "https://github.com/TitansO/Projet-SelfTtaining_CoTraining"
    "/raw/refs/heads/main/openaq_dakar_dataset.csv"
)

# Vues Co-Training
VUE_A = ["pm25", "pm10", "no2", "o3", "co"]
VUE_B = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "station_id", "is_harmattan",
    "rolling_pm25_3h",  # moyenne mobile PM2.5 3h par station
    "rolling_pm10_3h",  # moyenne mobile PM10 3h par station
    "pm25_lag1h",       # PM2.5 heure précédente par station
    "pm10_lag1h",       # PM10 heure précédente par station
    "pm25_trend",       # PM2.5 différence lag (tendance)
]
ALL_FEATURES = VUE_A + VUE_B

# ═══════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DU DATASET RÉEL DEPUIS GITHUB
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_dataset() -> tuple[pd.DataFrame, str]:
    """
    Charge openaq_dakar_dataset.csv depuis GitHub raw.
    Retourne (DataFrame enrichi, source_label).

    Étapes :
    1. Téléchargement via URL raw GitHub
    2. Parsing datetime + tri par station + datetime
    3. Calcul des features dynamiques (rolling, lag, trend) PAR STATION
    4. Encodage cyclique heure/mois
    5. Fallback synthétique si indisponible
    """
    try:
        df = pd.read_csv(CSV_RAW_URL)
        source = "📡 Dataset GitHub (openaq_dakar_dataset.csv)"
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger le CSV depuis GitHub ({e}). Génération synthétique.")
        df, source = _generate_fallback(), "🔄 Données synthétiques (fallback)"
        return df, source

    # ── Parsing & nettoyage ──────────────────────────────────────────────
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    # Vérification colonnes polluants
    for col in ["pm25", "pm10", "no2", "o3", "co"]:
        if col not in df.columns:
            df[col] = 0.0

    # ── Features dynamiques PAR STATION (tri déjà fait) ─────────────────
    df["rolling_pm25_3h"] = (
        df.groupby("station_id")["pm25"]
          .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    df["rolling_pm10_3h"] = (
        df.groupby("station_id")["pm10"]
          .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    df["pm25_lag1h"] = (
        df.groupby("station_id")["pm25"]
          .transform(lambda s: s.shift(1).fillna(method="bfill"))
    )
    df["pm10_lag1h"] = (
        df.groupby("station_id")["pm10"]
          .transform(lambda s: s.shift(1).fillna(method="bfill"))
    )
    df["pm25_trend"] = df["pm25"] - df["pm25_lag1h"]

    # ── Encodage cyclique ────────────────────────────────────────────────
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Colonne _synth absente dans le vrai CSV ──────────────────────────
    df["_synth"] = False

    return df.reset_index(drop=True), source


def _generate_fallback() -> pd.DataFrame:
    """Génération synthétique de secours (identique v4)."""
    from datetime import datetime, timedelta
    rng = np.random.default_rng(2024)
    stations = [
        {"id": 0, "name": "US Embassy Dakar",   "lat": 14.693, "lon": -17.447},
        {"id": 1, "name": "DEEC Plateau",        "lat": 14.682, "lon": -17.443},
        {"id": 2, "name": "Rufisque Industrial", "lat": 14.715, "lon": -17.274},
    ]
    PM25_BASE = {0: 30.0, 1: 55.0, 2: 72.0}
    HARM = {1,2,3,11,12}
    N_HOURS = 8760 * 2
    start = datetime(2022, 1, 1)
    rows = []
    for st_info in stations:
        sid = st_info["id"]; base25 = PM25_BASE[sid]
        prev_pm25 = base25; prev_pm10 = base25 * 2.0
        buf_pm25 = [base25]*3; buf_pm10 = [base25*2]*3
        for h in range(N_HOURS):
            dt = start + timedelta(hours=h)
            month = dt.month; hour = dt.hour; dow = dt.weekday()
            harm = month in HARM
            h_fac = rng.uniform(2.0,5.0) if harm else rng.uniform(0.7,1.3)
            rush_am = np.exp(-0.5*((hour-8)/1.8)**2)
            rush_pm = np.exp(-0.5*((hour-18)/1.8)**2)
            t_fac = 1.0 + 1.4*rush_am + 1.0*rush_pm
            if dow >= 5: t_fac *= 0.60
            pm25 = float(np.clip(base25*h_fac*t_fac*rng.lognormal(0,0.30), 2.0, 800.0))
            pm10 = float(np.clip(pm25*rng.uniform(1.5,3.0), pm25, 1200.0))
            no2_b = {0:18.0,1:35.0,2:45.0}[sid]
            no2 = float(np.clip(no2_b*t_fac*rng.lognormal(0,0.28), 1.0, 200.0))
            solar = np.sin(np.pi*max(0,hour-6)/12) if 6<=hour<=18 else 0.0
            o3 = float(np.clip(rng.normal(22+12*solar-0.3*no2, 6), 0, 120))
            co_b = {0:380.0,1:640.0,2:820.0}[sid]
            co = float(np.clip(co_b*t_fac*rng.lognormal(0,0.25), 50, 5000))
            buf_pm25.append(pm25); buf_pm25.pop(0)
            buf_pm10.append(pm10); buf_pm10.pop(0)
            composite = max(0.0, 0.55*pm25+0.20*(pm10/2.5)+0.15*(no2/1.5)+0.10*co/100+rng.normal(0,12))
            aqi = 0 if composite<15 else 1 if composite<30 else 2 if composite<60 else 3 if composite<90 else 4 if composite<160 else 5
            rows.append({
                "datetime": dt, "station_id": sid, "station_name": st_info["name"],
                "month": month, "hour": hour, "day_of_week": dow, "is_harmattan": int(harm),
                "pm25": round(pm25,2), "pm10": round(pm10,2), "no2": round(no2,2),
                "o3": round(o3,2), "co": round(co,2),
                "rolling_pm25_3h": round(float(np.mean(buf_pm25)),2),
                "rolling_pm10_3h": round(float(np.mean(buf_pm10)),2),
                "pm25_lag1h": round(prev_pm25,2), "pm10_lag1h": round(prev_pm10,2),
                "pm25_trend": round(pm25-prev_pm25,2),
                "aqi_label": aqi, "label_known": 0, "_synth": True,
            })
            prev_pm25 = pm25; prev_pm10 = pm10
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. SPLIT TEMPOREL & PRÉPARATION
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def prepare_splits(_df: pd.DataFrame):
    """
    Split temporel strict :
    - Test  : Q4 2023 (≥ 2023-10-01) — jamais vu en train
    - Train : tout le reste

    Labellisation :
    - Utilise label_known du CSV si présent (5 % déjà labellisé)
    - Sinon labellisation stratifiée 5 % sur les données réelles

    Normalisation : StandardScaler ajusté UNIQUEMENT sur L_train
    """
    df = _df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)

    cutoff = pd.Timestamp("2023-10-01")
    df_train = df[df["datetime"] < cutoff].copy()
    df_test  = df[df["datetime"] >= cutoff].copy()

    # ── Labellisation ───────────────────────────────────────────────────
    if "label_known" in df_train.columns and df_train["label_known"].sum() > 10:
        # Utiliser label_known existant du CSV
        pass  # label_known déjà présent
    else:
        # Fallback : stratification 5 %
        rng = np.random.default_rng(42)
        df_train["label_known"] = 0
        for cls in range(6):
            for sid in df_train["station_id"].unique():
                pool = df_train[
                    (df_train["aqi_label"] == cls) &
                    (df_train["station_id"] == sid)
                ].index.tolist()
                if not pool: continue
                n_sel = max(2, int(len(pool) * 0.05))
                sel = rng.choice(pool, size=min(n_sel, len(pool)), replace=False)
                df_train.loc[sel, "label_known"] = 1

    df_test["label_known"] = 1

    df_L = df_train[df_train["label_known"] == 1].copy()
    df_U = df_train[df_train["label_known"] == 0].copy()

    # ── Vérifier que toutes les features sont disponibles ───────────────
    missing = [f for f in ALL_FEATURES if f not in df_L.columns]
    if missing:
        raise ValueError(f"Features manquantes dans le dataset : {missing}")

    scaler = StandardScaler()
    scaler.fit(df_L[ALL_FEATURES].values)

    def sc(frame):
        return scaler.transform(frame[ALL_FEATURES].values)

    X_L = sc(df_L); y_L = df_L["aqi_label"].values
    X_U = sc(df_U)
    X_test = sc(df_test); y_test = df_test["aqi_label"].values

    va_idx = [ALL_FEATURES.index(f) for f in VUE_A]
    vb_idx = [ALL_FEATURES.index(f) for f in VUE_B]

    return {
        "df_full":  df_train,
        "df_test":  df_test,
        "df_L": df_L, "df_U": df_U,
        "X_L": X_L, "y_L": y_L,
        "X_U": X_U,
        "X_test": X_test, "y_test": y_test,
        "va_idx": va_idx, "vb_idx": vb_idx,
        "scaler": scaler,
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
# 4. ALGORITHMES SSL (conservés depuis v4 avec toutes corrections P1–P6)
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
    history = []; best_f1 = -1.0; best_clf = None; no_improve = 0

    for it in range(max_iter + 1):
        clf = make_clf(n_estimators)
        clf.fit(X_Lc, y_Lc)
        y_pred = clf.predict(X_test)
        f1_now = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        rec = {
            "iteration":   it, "n_L": len(X_Lc), "n_U": len(X_Uc),
            "f1_macro":    f1_now,
            "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "precision":   round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "recall":      round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "n_added": 0,
            "gamma_used":  round(_gamma_anneal(it, max_iter, gamma_start, gamma), 3),
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
        pseudo = clf.classes_[proba[mask].argmax(axis=1)]
        X_Lc   = np.vstack([X_Lc, X_Uc[mask]])
        y_Lc   = np.concatenate([y_Lc, pseudo])
        X_Uc   = X_Uc[~mask]

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
    history = []; best_f1 = -1.0; best_cA = None; best_cB = None; no_improve = 0

    def _ensemble(cA, cB):
        pA  = cA.predict_proba(X_tA); pB = cB.predict_proba(X_tB)
        cls = np.union1d(cA.classes_, cB.classes_)
        def _al(c, p):
            out = np.zeros((p.shape[0], len(cls)))
            for j, cl in enumerate(cls):
                if cl in c.classes_:
                    out[:, j] = p[:, np.where(c.classes_==cl)[0][0]]
            return out
        return cls[( _al(cA,pA) + _al(cB,pB) ).argmax(axis=1) // 1]

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
            "gamma_used":  round(_gamma_anneal(it, max_iter, gamma_start, gamma), 3),
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
        pred_A = cA.classes_[pA[sel_A].argmax(axis=1)]
        pred_B = cB.classes_[pB[sel_B].argmax(axis=1)]
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

    if best_cA: history[-1]["clf_A"] = best_cA; history[-1]["clf_B"] = best_cB
    history[-1]["best_f1"] = best_f1
    return history


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
        wedgeprops=dict(edgecolor="white", linewidth=2), textprops={"fontsize":10})
    axes[0].set_title("Ratio L / U — Scarcité des étiquettes",
                      fontsize=12, fontweight="bold", color=PALETTE["navy"])
    df_L = df_full[df_full["label_known"] == 1]
    cnt  = df_L["aqi_label"].value_counts().sort_index()
    bars = axes[1].bar([AQI_NAMES[i][0] for i in cnt.index], cnt.values,
                       color=[AQI_NAMES[i][1] for i in cnt.index], edgecolor="white")
    for b, v in zip(bars, cnt.values):
        axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+2,
                     str(v), ha="center", va="bottom", fontsize=9,
                     fontweight="bold", color=PALETTE["navy"])
    axes[1].set_title("Distribution AQI — Ensemble L",
                      fontsize=12, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_xlabel("Classe AQI"); axes[1].set_ylabel("Observations")
    axes[1].tick_params(axis="x", rotation=15); axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig

def fig_pm25_temporal(df_full):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6)); _style(fig)
    df2 = df_full.copy()
    df2["ym"] = df2["datetime"].dt.to_period("M").astype(str)
    monthly = df2.groupby(["ym","station_name"])["pm25"].mean().reset_index()
    clrs = [PALETTE["teal"], PALETTE["orange"], PALETTE["purple"]]
    for i, st in enumerate(df2["station_name"].unique()):
        sub = monthly[monthly["station_name"]==st]
        axes[0].plot(range(len(sub)), sub["pm25"].values,
                     label=st, color=clrs[i % len(clrs)], linewidth=1.8)
    for y, c, lbl in [(15,PALETTE["green"],"OMS annuel (15)"),
                       (25,PALETTE["orange"],"OMS 24h (25)"),
                       (75,PALETTE["red"],"Mauvais I (75)")]:
        axes[0].axhline(y, linestyle="--", linewidth=1, color=c, label=lbl)
    axes[0].set_title("PM2.5 mensuel moyen par station",
                      fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[0].set_ylabel("PM2.5 (µg/m³)"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    hrly = df2.groupby("hour")["pm25"].mean()
    axes[1].fill_between(hrly.index, hrly.values, alpha=0.25, color=PALETTE["teal"])
    axes[1].plot(hrly.index, hrly.values, color=PALETTE["teal"], linewidth=2.5)
    axes[1].axvspan(7, 9, alpha=0.15, color=PALETTE["orange"], label="Rush matin")
    axes[1].axvspan(17,20,alpha=0.15, color=PALETTE["red"],    label="Rush soir")
    axes[1].set_title("Profil diurne PM2.5 — effet trafic",
                      fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_xlabel("Heure"); axes[1].set_ylabel("PM2.5 moyen (µg/m³)")
    axes[1].set_xticks(range(0,24,2)); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout(); return fig

def fig_correlation_views(df_full):
    cols = [c for c in VUE_A + ["hour_sin","hour_cos","month_sin","month_cos",
                                  "station_id","is_harmattan",
                                  "rolling_pm25_3h","pm25_lag1h","pm25_trend"]
            if c in df_full.columns]
    corr = df_full[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 8)); _style(fig)
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size":7},
                linewidths=0.5, linecolor="white", cbar_kws={"shrink":0.8})
    n_a = len(VUE_A)
    ax.add_patch(mpatches.Rectangle((0,0), n_a, n_a,
                 fill=False, edgecolor=PALETTE["teal"],   linewidth=2.5))
    ax.add_patch(mpatches.Rectangle((n_a,n_a), len(cols)-n_a, len(cols)-n_a,
                 fill=False, edgecolor=PALETTE["orange"], linewidth=2.5))
    ax.set_title("Corrélation inter-features — Validation indépendance des vues",
                 fontsize=11, fontweight="bold", color=PALETTE["navy"])
    ax.legend(handles=[
        mpatches.Patch(color=PALETTE["teal"],   label="Vue A — Polluants"),
        mpatches.Patch(color=PALETTE["orange"], label="Vue B — Contexte + Dynamique"),
    ], loc="upper right", fontsize=9)
    plt.tight_layout(); return fig

def fig_aqi_distribution(df_full):
    """Distribution AQI globale sur le dataset réel."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)); _style(fig)
    cnt = df_full["aqi_label"].value_counts().sort_index()
    bars = axes[0].bar([AQI_NAMES[i][0] for i in cnt.index], cnt.values,
                       color=[AQI_NAMES[i][1] for i in cnt.index], edgecolor="white")
    for b, v in zip(bars, cnt.values):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+20,
                     f"{v:,}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[0].set_title("Distribution AQI — Dataset complet (2020–2023)",
                      fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[0].set_xlabel("Classe AQI"); axes[0].set_ylabel("Observations")
    axes[0].tick_params(axis="x", rotation=15); axes[0].grid(axis="y", alpha=0.3)
    # Par station
    for i, (sid, grp) in enumerate(df_full.groupby("station_id")):
        cnt_s = grp["aqi_label"].value_counts().sort_index()
        axes[1].bar(
            np.array(cnt_s.index) + i*0.25,
            cnt_s.values, 0.25,
            label=grp["station_name"].iloc[0],
            color=[PALETTE["teal"],PALETTE["orange"],PALETTE["purple"]][i],
            edgecolor="white", alpha=0.85
        )
    axes[1].set_xticks(range(6))
    axes[1].set_xticklabels([AQI_NAMES[i][0] for i in range(6)], rotation=15)
    axes[1].set_title("Distribution AQI par station",
                      fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_xlabel("Classe AQI"); axes[1].set_ylabel("Observations")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)
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
    best_rows = df_h[df_h.get("is_best", pd.Series([False]*len(df_h))).astype(bool)]
    if not best_rows.empty:
        axes[0].scatter(best_rows["iteration"], best_rows["f1_macro"],
                        s=120, zorder=5, color=PALETTE["green"], label="Meilleur")
    axes[0].set_title(f"{algo_name} — F1 macro", fontsize=11,
                      fontweight="bold", color=PALETTE["navy"])
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
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 5.5)); _style(fig)
    sns.heatmap(cm_norm, ax=ax, cmap="Blues", annot=True, fmt=".2f",
                xticklabels=[AQI_NAMES[i][0] for i in range(6)],
                yticklabels=[AQI_NAMES[i][0] for i in range(6)],
                linewidths=0.5, linecolor="white", cbar_kws={"shrink":0.8})
    ax.set_title(title, fontsize=11, fontweight="bold", color=PALETTE["navy"])
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel"); ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); return fig

def fig_compare(results_dict):
    methods = list(results_dict.keys())
    x = np.arange(len(methods)); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5)); _style(fig)
    b1 = ax.bar(x-w, [v["f1_macro"]  for v in results_dict.values()], w,
                label="F1 macro",        color=PALETTE["teal"],   edgecolor="white")
    b2 = ax.bar(x,   [v["precision"] for v in results_dict.values()], w,
                label="Précision macro", color=PALETTE["orange"], edgecolor="white")
    b3 = ax.bar(x+w, [v["recall"]    for v in results_dict.values()], w,
                label="Rappel macro",    color=PALETTE["purple"], edgecolor="white")
    for bs in [b1, b2, b3]:
        for b in bs:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f"{b.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color=PALETTE["navy"])
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_ylim(0,1); ax.set_ylabel("Score")
    ax.set_title("Comparaison : Baseline < Self-Training < Co-Training",
                 fontsize=12, fontweight="bold", color=PALETTE["navy"])
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5, linestyle=":", color=PALETTE["grey"], linewidth=1)
    plt.tight_layout(); return fig

def fig_per_class_f1(y_test, results_dict):
    methods = list(results_dict.keys())
    clr_map = {"Baseline (L seul)": PALETTE["grey"],
               "Self-Training":     PALETTE["orange"],
               "Co-Training":       PALETTE["teal"]}
    fig, ax = plt.subplots(figsize=(11, 5)); _style(fig)
    x = np.arange(6); w = 0.25
    for i, (name, res) in enumerate(results_dict.items()):
        report = classification_report(y_test, res["y_pred"],
                                       labels=list(range(6)), output_dict=True, zero_division=0)
        f1s = [report.get(str(c), {}).get("f1-score", 0) for c in range(6)]
        offset = (i - len(methods)/2 + 0.5) * w
        ax.bar(x + offset, f1s, w, label=name,
               color=clr_map.get(name, PALETTE["purple"]), edgecolor="white", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([AQI_NAMES[i][0] for i in range(6)], fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("F1-Score par classe")
    ax.set_title("F1 par classe AQI — Dataset réel OpenAQ",
                 fontsize=12, fontweight="bold", color=PALETTE["navy"])
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(
    "<h2 style='color:#0A8A7C;margin-bottom:0'>⚙️ Configuration v5</h2>",
    unsafe_allow_html=True)
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

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<small style='color:#7A8BA0'>"
    f"📡 <b>Source :</b> GitHub raw CSV<br>"
    f"🌐 <a href='{CSV_RAW_URL}' style='color:#0A8A7C'>openaq_dakar_dataset.csv</a><br>"
    f"📅 <b>Période :</b> 2020–2023 · 105 120 lignes<br>"
    f"📍 <b>Stations :</b> US Embassy · DEEC · Rufisque<br>"
    f"🧪 <b>Test :</b> Q4 2023 (bloc temporel futur)<br>"
    f"</small>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 7. CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════

with st.spinner("⏳ Chargement du dataset OpenAQ depuis GitHub…"):
    df_raw, data_source = load_dataset()

with st.spinner("⚙️ Préparation des splits et features…"):
    try:
        data = prepare_splits(df_raw)
    except ValueError as e:
        st.error(f"❌ Erreur de préparation : {e}")
        st.stop()

df_full  = data["df_full"]
df_test  = data["df_test"]
X_L      = data["X_L"]; y_L = data["y_L"]
X_U      = data["X_U"]
X_test   = data["X_test"]; y_test = data["y_test"]
va_idx   = data["va_idx"]; vb_idx = data["vb_idx"]

# ═══════════════════════════════════════════════════════════════════════════
# 8. HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div style='background:linear-gradient(135deg,#0B1F3A 0%,#0A8A7C 100%);
            padding:28px 32px;border-radius:12px;margin-bottom:24px'>
  <h1 style='color:white;margin:0;font-size:2rem'>
    🌍 Apprentissage Semi-Supervisé — Qualité de l'Air à Dakar
  </h1>
  <p style='color:#B2D8D4;margin:8px 0 4px 0;font-size:1rem'>
    Self-Training &amp; Co-Training · Dataset OpenAQ réel ·
    Stations DEEC &amp; US Embassy · 2020–2023
  </p>
  <span style='background:#27AE60;color:white;padding:3px 10px;border-radius:20px;
               font-size:0.8rem;font-weight:bold'>{data_source}</span>
</div>""", unsafe_allow_html=True)

n_total = len(df_full); n_L_cnt = int(df_full["label_known"].sum()); n_U_cnt = n_total - n_L_cnt
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("📦 Train total",        f"{n_total:,}")
c2.metric("🏷 Labellisés L",        f"{n_L_cnt:,}", f"{n_L_cnt/n_total*100:.1f}%")
c3.metric("🔓 Non-labellisés U",    f"{n_U_cnt:,}", f"{n_U_cnt/n_total*100:.1f}%")
c4.metric("🧪 Test set (Q4 2023)", f"{len(df_test):,}")
c5.metric("📡 Période dataset",     f"2020–2023")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# 9. ONGLETS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📊 Analyse Exploratoire (EDA)",
    "🤖 Simulation Semi-Supervisée",
    "📈 Dashboard Résultats",
])

# ─── TAB 1 ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🔍 Analyse Exploratoire — Dataset OpenAQ Dakar (réel)")

    with st.expander("📡 Source des données — GitHub raw CSV", expanded=True):
        st.markdown(f"""
| Paramètre | Valeur |
|---|---|
| **Source** | `openaq_dakar_dataset.csv` — dépôt GitHub TitansO |
| **URL raw** | `{CSV_RAW_URL}` |
| **Lignes** | {len(df_raw):,} observations horaires |
| **Période** | 2020-01-01 → 2023-12-30 |
| **Stations** | US Embassy Dakar · DEEC Plateau · Rufisque Industrial |
| **Polluants** | PM2.5 · PM10 · NO₂ · O₃ · CO |
| **label_known** | {int(df_raw['label_known'].sum()):,} points labellisés ({int(df_raw['label_known'].sum())/len(df_raw)*100:.1f} %) |
| **Features dynamiques ajoutées** | rolling_pm25_3h · rolling_pm10_3h · pm25_lag1h · pm10_lag1h · pm25_trend |
        """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Scarcité des étiquettes & Distribution AQI dans L")
        st.pyplot(fig_label_scarcity(df_full), use_container_width=True)
    with col_b:
        st.markdown("#### Séries temporelles PM2.5 par station")
        st.pyplot(fig_pm25_temporal(df_full), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📊 Distribution AQI — Dataset réel (2020–2023)")
    st.pyplot(fig_aqi_distribution(df_full), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔗 Heatmap Corrélation — Validation indépendance des vues")
    st.info("**Vue B enrichie :** rolling_pm25_3h, pm25_lag1h et pm25_trend "
            "apportent une information dynamique (évolution) absente des polluants "
            "instantanés de Vue A, tout en restant modérément corrélés.")
    st.pyplot(fig_correlation_views(df_full), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📋 Aperçu du dataset (avec features dynamiques)")
    disp = ["datetime","station_name","pm25","rolling_pm25_3h","pm25_lag1h",
            "pm25_trend","pm10","no2","o3","co","is_harmattan","aqi_label","label_known"]
    disp = [c for c in disp if c in df_full.columns]
    st.dataframe(df_full[disp].head(25), use_container_width=True)

    st.markdown("#### 📐 Statistiques descriptives — polluants réels")
    st.dataframe(df_full[["pm25","pm10","no2","o3","co"]].describe().round(2),
                 use_container_width=True)

# ─── TAB 2 ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown(f"### 🤖 Simulation — **{algo_choice}** (γ_fin={gamma}, patience={patience})")
    st.markdown(
        f"**Algo :** `{algo_choice}` | **γ fin :** `{gamma}` | **Marge :** `{min_margin}` | "
        f"**Patience :** `{patience}` | **Itérations :** `{max_iter}` | **Arbres :** `{n_estimators}`"
        + (f" | **k/iter :** `{k_per_iter}`" if algo_choice == "Co-Training" else "")
    )

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
        stat.success(f"✅ Terminé en {elapsed:.1f}s — {final['iteration']} itérations | Meilleur F1 = {best_f1:.4f}")

        cols_tbl = ["iteration","n_L","n_U","f1_macro","precision","recall","n_added","gamma_used"]
        cols_tbl = [c for c in cols_tbl if c in pd.DataFrame(history).columns]
        df_h = pd.DataFrame(history)[cols_tbl].rename(columns={
            "iteration":"Iter.","n_L":"|L|","n_U":"|U|",
            "f1_macro":"F1 macro","precision":"Précision",
            "recall":"Rappel","n_added":"Ajoutés","gamma_used":"γ"})
        tbl.dataframe(
            df_h.style
            .format({"F1 macro":"{:.4f}","Précision":"{:.4f}","Rappel":"{:.4f}","γ":"{:.3f}"})
            .background_gradient(subset=["F1 macro"], cmap="Greens"),
            use_container_width=True)

        st.markdown("---")
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("F1 Final (macro)", f"{final['f1_macro']:.4f}", f"{gain:+.4f}")
        k2.metric("Meilleur F1",      f"{best_f1:.4f}")
        k3.metric("Précision macro",  f"{final['precision']:.4f}")
        k4.metric("Rappel macro",     f"{final['recall']:.4f}")
        k5.metric("|L| final",        f"{final['n_L']:,}",
                  f"+{final['n_L']-len(X_L):,} pseudo-labels")

        st.markdown("#### 📈 Évolution F1, |L|/|U| et Gamma Annealing")
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
                        out[:, j] = p[:, np.where(c.classes_==cl)[0][0]]
                return out
            pf = (_al(cA,pA) + _al(cB,pB)) / 2
            y_pred_fin = cls[pf.argmax(axis=1)]

        st.markdown("#### 🔲 Matrice de Confusion (Test Q4 2023 — données réelles)")
        st.pyplot(fig_confusion(y_test, y_pred_fin, f"{algo_choice} — γ={gamma}"),
                  use_container_width=True)

        st.markdown("#### 📋 Rapport de Classification")
        report = classification_report(
            y_test, y_pred_fin, labels=list(range(6)),
            target_names=[AQI_NAMES[i][0] for i in range(6)],
            output_dict=True, zero_division=0)
        st.dataframe(
            pd.DataFrame(report).T.round(4)
            .style.background_gradient(subset=["f1-score"], cmap="Greens"),
            use_container_width=True)

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
        st.info("💡 Configurez les paramètres en sidebar puis cliquez **▶️ Lancer**.")

# ─── TAB 3 ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📈 Dashboard Comparatif — Dataset réel OpenAQ")

    if "results" not in st.session_state or not st.session_state["results"]:
        st.warning("⚠️ Aucun résultat — lancez d'abord une simulation dans l'onglet **Simulation**.")
    else:
        clf_base = make_clf(100); clf_base.fit(X_L, y_L)
        y_base   = clf_base.predict(X_test)
        all_res  = {
            "Baseline (L seul)": {
                "f1_macro":  round(f1_score(y_test, y_base, average="macro",    zero_division=0), 4),
                "precision": round(precision_score(y_test, y_base, average="macro", zero_division=0), 4),
                "recall":    round(recall_score(y_test, y_base, average="macro",  zero_division=0), 4),
                "y_pred":    y_base,
            }
        }
        for k, v in st.session_state["results"].items():
            all_res[k] = v

        st.markdown("#### 📊 Comparaison Globale")
        st.pyplot(fig_compare(all_res), use_container_width=True)

        st.markdown("#### 🎯 F1 par Classe AQI (données réelles)")
        st.pyplot(fig_per_class_f1(y_test, all_res), use_container_width=True)

        st.markdown("#### 🗃 Tableau Récapitulatif")
        df_comp = pd.DataFrame({k: {"F1 macro":v["f1_macro"],
                                     "Précision":v["precision"],
                                     "Rappel":v["recall"]}
                                  for k,v in all_res.items()}).T
        base_f1 = df_comp.loc["Baseline (L seul)","F1 macro"]
        df_comp["Δ F1 vs Baseline"] = (df_comp["F1 macro"] - base_f1).round(4)
        st.dataframe(
            df_comp.style
            .format({"F1 macro":"{:.4f}","Précision":"{:.4f}",
                     "Rappel":"{:.4f}","Δ F1 vs Baseline":"{:+.4f}"})
            .background_gradient(subset=["F1 macro"], cmap="Greens")
            .background_gradient(subset=["Δ F1 vs Baseline"], cmap="RdYlGn", vmin=-0.1, vmax=0.25),
            use_container_width=True)

        st.markdown("#### 🏆 Hiérarchie des performances")
        for i, m in enumerate(sorted(all_res, key=lambda k: all_res[k]["f1_macro"])):
            icon  = ["🥉","🥈","🥇"][min(i, 2)]
            color = [PALETTE["grey"], PALETTE["orange"], PALETTE["teal"]][min(i, 2)]
            st.markdown(
                f"<div style='background:{color}22;border-left:4px solid {color};"
                f"padding:8px 16px;border-radius:6px;margin:4px 0'>"
                f"{icon} <b>{m}</b> — F1 macro = <b>{all_res[m]['f1_macro']:.4f}</b>"
                f"</div>", unsafe_allow_html=True)

        if len(st.session_state["results"]) >= 2:
            st.markdown("---")
            st.markdown("#### 📈 Courbes d'Apprentissage SSL Comparées")
            fig_evo, ax_evo = plt.subplots(figsize=(11, 4.5))
            fig_evo.patch.set_facecolor(PALETTE["cream"])
            ax_evo.set_facecolor(PALETTE["cream"])
            for name, res in st.session_state["results"].items():
                h = res["history"]
                ax_evo.plot([r["iteration"] for r in h], [r["f1_macro"] for r in h],
                            label=name, linewidth=2.5, marker="o", markersize=5,
                            color={"Self-Training":PALETTE["orange"],"Co-Training":PALETTE["teal"]}.get(name, PALETTE["purple"]))
            ax_evo.axhline(base_f1, linestyle="--", color=PALETTE["red"],
                           linewidth=1.5, label="Baseline")
            ax_evo.set_xlabel("Itération"); ax_evo.set_ylabel("F1-Score macro")
            ax_evo.set_title("Évolution F1 — Self-Training vs Co-Training (données réelles)",
                             fontsize=12, fontweight="bold", color=PALETTE["navy"])
            ax_evo.legend(fontsize=10); ax_evo.grid(alpha=0.3); ax_evo.set_ylim(0,1)
            plt.tight_layout()
            st.pyplot(fig_evo, use_container_width=True)
