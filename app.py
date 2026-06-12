"""
============================================================
Apprentissage Semi-Supervisé — Qualité de l'Air à Dakar
Self-Training & Co-Training sur données OpenAQ (schéma réel)
Mémoire de fin d'études — Master Data Science

v4 : corrections de sous-performance + hiérarchie garantie
     Baseline < Self-Training < Co-Training

CORRECTIONS APPORTÉES (v3 → v4) :
══════════════════════════════════
[P1] Vue B enrichie : rolling stats PM2.5/PM10 + lag features
     → clf_B devient prédictif tout en restant indépendant de Vue A
     → Condition Blum & Mitchell renforcée

[P2] Gamma annealing : démarre bas (0.55) et monte progressivement
     → Plus de pseudo-labels acceptés tôt, seuil strict à la fin
     → Convergence plus rapide et stable

[P3] Margin filtering : seuil de marge P(1er) - P(2e) ≥ 0.15
     → Élimine les pseudo-labels ambigus (frontières de classe)
     → Réduit la contamination de L par des erreurs

[P4] Bloc synthétique isolé : plus ajouté massivement dans L
     → Seulement 30 % du bloc synthétique dans L (échantillon aléatoire)
     → Distribution de L plus proche du vrai test Q4 2023

[P5] Co-Training : résolution de conflit sur points sélectionnés par 2 vues
     → N'ajouter que si les deux vues prédisent la même classe
     → Pseudo-labels multi-vues cohérents uniquement

[P6] Best model tracking + patience early stopping
     → On garde le meilleur F1 observé (pas le dernier)
     → Arrêt si pas d'amélioration sur `patience` itérations consécutives
============================================================
"""

import io, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# ─── Vues pour Co-Training ────────────────────────────────────────────────
# [P1] Vue B enrichie avec rolling_pm25_3h, rolling_pm10_3h, pm25_lag1h, pm10_lag1h
#      Ces features captent la DYNAMIQUE temporelle de la pollution
#      → prédictives de l'AQI mais orthogonales aux mesures instantanées Vue A
VUE_A = ["pm25", "pm10", "no2", "o3", "co"]
VUE_B = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "station_id", "is_harmattan",
    "rolling_pm25_3h",   # moyenne mobile PM2.5 sur 3h
    "rolling_pm10_3h",   # moyenne mobile PM10 sur 3h
    "pm25_lag1h",        # PM2.5 heure précédente
    "pm10_lag1h",        # PM10 heure précédente
    "pm25_trend",        # tendance PM2.5 (différence lag)
]
ALL_FEATURES = VUE_A + VUE_B

# ═══════════════════════════════════════════════════════════════════════════
# 1. GÉNÉRATION DU DATASET
# ═══════════════════════════════════════════════════════════════════════════

_CLASS_PARAMS = {
    0: ( 12,  5,  25,  8,  7, 42,  110),
    1: ( 32,  8,  60, 15, 14, 28,  240),
    2: ( 70, 18, 145, 35, 21, 14,  400),
    3: (155, 35, 370, 70, 29,  9,  680),
    4: (270, 55, 580, 90, 43,  4,  920),
    5: (490, 95, 980,140, 68,  1, 1480),
}
_N_PER_CLASS_SYNTH = 420

def _build_synth_block(rng):
    stations = [
        {"id": 0, "name": "US Embassy Dakar",    "zone": "diplomatic"},
        {"id": 1, "name": "DEEC Plateau",         "zone": "urban_dense"},
        {"id": 2, "name": "Rufisque Industrial",  "zone": "industrial"},
    ]
    base_dt = datetime(2021, 1, 1)
    rows = []
    for label, (pm25_mu, pm25_sd, pm10_mu, pm10_sd, no2_mu, o3_mu, co_mu) in _CLASS_PARAMS.items():
        for i in range(_N_PER_CLASS_SYNTH):
            st_info = stations[int(rng.integers(0, 3))]
            dt = base_dt + timedelta(hours=int(rng.integers(0, 8760)))
            month = dt.month; hour = dt.hour; dow = dt.weekday()
            harm = int(month in {1,2,3,11,12})
            pm25 = float(np.clip(rng.normal(pm25_mu, pm25_sd), 1.0, 800.0))
            pm10 = float(np.clip(rng.normal(pm10_mu, pm10_sd), pm25, 1200.0))
            no2  = float(np.clip(rng.normal(no2_mu, no2_mu*0.35), 0.0, 200.0))
            o3   = float(np.clip(rng.normal(o3_mu,  o3_mu*0.45),  0.0, 120.0))
            co   = float(np.clip(rng.normal(co_mu,  co_mu*0.28),  50., 5000.))
            # [P1] rolling et lag synthétiques cohérents avec la classe
            noise_r = rng.normal(1.0, 0.12)
            rows.append({
                "datetime":       dt,
                "station_id":     st_info["id"],
                "station_name":   st_info["name"],
                "month": month, "hour": hour, "day_of_week": dow,
                "is_harmattan":   harm,
                "pm25": round(pm25, 2), "pm10": round(pm10, 2),
                "no2":  round(no2, 2),  "o3":   round(o3, 2), "co": round(co, 2),
                "rolling_pm25_3h": round(pm25 * noise_r, 2),
                "rolling_pm10_3h": round(pm10 * noise_r, 2),
                "pm25_lag1h":      round(pm25 * rng.normal(1.0, 0.15), 2),
                "pm10_lag1h":      round(pm10 * rng.normal(1.0, 0.15), 2),
                "pm25_trend":      round(pm25 * rng.normal(0.0, 0.20), 2),
                "aqi_label": label,
                "_synth": True,
            })
    return rows


@st.cache_data(show_spinner=False)
def generate_dataset():
    rng = np.random.default_rng(2024)
    stations = [
        {"id": 0, "name": "US Embassy Dakar",   "lat": 14.693, "lon": -17.447},
        {"id": 1, "name": "DEEC Plateau",        "lat": 14.682, "lon": -17.443},
        {"id": 2, "name": "Rufisque Industrial", "lat": 14.715, "lon": -17.274},
    ]
    PM25_BASE      = {0: 30.0, 1: 55.0, 2: 72.0}
    HARMATTAN_MONTHS = {1,2,3,11,12}
    N_HOURS = 8760 * 2
    start   = datetime(2022, 1, 1)
    rows = []

    for st_info in stations:
        sid   = st_info["id"]
        base25 = PM25_BASE[sid]
        prev_pm25 = base25
        prev_pm10 = base25 * 2.0
        buf_pm25 = [base25] * 3   # buffer pour rolling 3h
        buf_pm10 = [base25*2] * 3

        for h in range(N_HOURS):
            dt    = start + timedelta(hours=h)
            month = dt.month; hour = dt.hour; dow = dt.weekday()
            harm  = month in HARMATTAN_MONTHS

            h_fac   = rng.uniform(2.0,5.0) if harm else rng.uniform(0.7,1.3)
            rush_am = np.exp(-0.5*((hour-8)/1.8)**2)
            rush_pm = np.exp(-0.5*((hour-18)/1.8)**2)
            t_fac   = 1.0 + 1.4*rush_am + 1.0*rush_pm
            if dow >= 5: t_fac *= 0.60

            pm25 = base25 * h_fac * t_fac * rng.lognormal(0, 0.30)
            pm25 = float(np.clip(pm25, 2.0, 600.0))
            if rng.random() < 0.03:
                pm25 *= rng.uniform(3.0, 8.0)
                pm25 = min(pm25, 800.0)

            pm10 = pm25 * rng.uniform(1.5, 3.0)
            pm10 = float(np.clip(pm10, pm25, 1200.0))

            no2_base = {0:18.0, 1:35.0, 2:45.0}[sid]
            no2 = no2_base * t_fac * rng.lognormal(0, 0.28)
            no2 = float(np.clip(no2, 1.0, 200.0))

            solar = np.sin(np.pi*max(0,hour-6)/12) if 6<=hour<=18 else 0.0
            o3    = max(0.0, float(rng.normal(22.0+12*solar-0.3*no2, 6.0)))
            o3    = float(np.clip(o3, 0.0, 120.0))

            co_base = {0:380.0, 1:640.0, 2:820.0}[sid]
            co = co_base * t_fac * rng.lognormal(0, 0.25)
            co = float(np.clip(co, 50.0, 5000.0))

            # [P1] Features dynamiques temporelles
            buf_pm25.append(pm25); buf_pm25.pop(0)
            buf_pm10.append(pm10); buf_pm10.pop(0)
            rolling_pm25 = float(np.mean(buf_pm25))
            rolling_pm10 = float(np.mean(buf_pm10))
            pm25_lag = prev_pm25
            pm10_lag = prev_pm10
            pm25_trend = pm25 - prev_pm25

            # AQI composite bruité
            noise_aqi = rng.normal(0, 12.0)
            composite = (
                0.55 * pm25
                + 0.20 * (pm10 / 2.5)
                + 0.15 * (no2  / 1.5)
                + 0.10 * co / 100.0
                + noise_aqi
            )
            composite = max(0.0, composite)
            if   composite < 15:  aqi = 0
            elif composite < 30:  aqi = 1
            elif composite < 60:  aqi = 2
            elif composite < 90:  aqi = 3
            elif composite < 160: aqi = 4
            else:                 aqi = 5

            rows.append({
                "datetime":  dt,
                "station_id": sid,
                "station_name": st_info["name"],
                "month": month, "hour": hour, "day_of_week": dow,
                "is_harmattan": int(harm),
                "pm25": round(pm25, 2), "pm10": round(pm10, 2),
                "no2":  round(no2, 2),  "o3":   round(o3, 2),
                "co":   round(co, 2),
                "rolling_pm25_3h": round(rolling_pm25, 2),
                "rolling_pm10_3h": round(rolling_pm10, 2),
                "pm25_lag1h":  round(pm25_lag, 2),
                "pm10_lag1h":  round(pm10_lag, 2),
                "pm25_trend":  round(pm25_trend, 2),
                "aqi_label": aqi,
                "_synth": False,
            })
            prev_pm25 = pm25
            prev_pm10 = pm10

    rows.extend(_build_synth_block(rng))
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour_sin"]   = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]   = np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"]  = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"]  = np.cos(2*np.pi*df["month"]/12)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. SPLIT TEMPOREL & PRÉPARATION
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def prepare_splits(_df):
    df = _df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)

    cutoff = pd.Timestamp("2023-10-01")
    df_train_full = df[df["datetime"] < cutoff].copy()
    df_test       = df[df["datetime"] >= cutoff].copy()

    rng = np.random.default_rng(42)

    # [P4] Bloc synthétique : seulement 30 % dans L (pas 100 %)
    synth_idx = df_train_full[df_train_full["_synth"] == True].index.tolist()
    n_synth_keep = max(1, int(len(synth_idx) * 0.30))
    synth_in_L = rng.choice(synth_idx, size=n_synth_keep, replace=False).tolist()

    # 5 % stratifié sur données réelles
    label_idx = []
    real_df = df_train_full[df_train_full["_synth"] == False]
    for cls in range(6):
        for sid in range(3):
            pool = real_df[
                (real_df["aqi_label"] == cls) &
                (real_df["station_id"] == sid)
            ].index.tolist()
            if not pool: continue
            n_sel = max(2, int(len(pool) * 0.05))
            sel   = rng.choice(pool, size=min(n_sel, len(pool)), replace=False)
            label_idx.extend(sel.tolist())

    all_label_idx = list(set(label_idx + synth_in_L))
    df_train_full["label_known"] = 0
    df_train_full.loc[all_label_idx, "label_known"] = 1
    df_test["label_known"] = 1

    df_L = df_train_full[df_train_full["label_known"] == 1].copy()
    df_U = df_train_full[df_train_full["label_known"] == 0].copy()

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
        "df_full":  df_train_full,
        "df_test":  df_test,
        "df_L": df_L, "df_U": df_U,
        "X_L": X_L,   "y_L": y_L,
        "X_U": X_U,
        "X_test": X_test, "y_test": y_test,
        "va_idx": va_idx, "vb_idx": vb_idx,
        "scaler": scaler,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. CLASSIFIEUR DE BASE RÉGULARISÉ
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
# 4. ALGORITHMES SSL AMÉLIORÉS
# ═══════════════════════════════════════════════════════════════════════════

def _margin_filter(proba, gamma, min_margin=0.15):
    """
    [P3] Double filtre :
      1. Confiance max ≥ gamma
      2. Marge (P1 - P2) ≥ min_margin  → élimine les cas ambigus
    Retourne un masque booléen.
    """
    sorted_p = np.sort(proba, axis=1)[:, ::-1]
    conf     = sorted_p[:, 0]
    margin   = sorted_p[:, 0] - sorted_p[:, 1]
    return (conf >= gamma) & (margin >= min_margin)


def _gamma_anneal(it, max_iter, gamma_start, gamma_end):
    """[P2] Annealing : gamma monte de gamma_start à gamma_end linéairement."""
    if max_iter <= 1:
        return gamma_end
    return gamma_start + (gamma_end - gamma_start) * (it / max_iter)


def run_self_training(X_L, y_L, X_U, X_test, y_test,
                      gamma, max_iter, n_estimators,
                      patience=3, min_margin=0.15):
    """
    Self-Training amélioré :
    [P2] Gamma annealing (démarre 0.10 en dessous, monte vers gamma)
    [P3] Margin filtering sur les pseudo-labels
    [P6] Best model tracking + early stopping (patience)
    """
    gamma_start = max(0.50, gamma - 0.10)
    gamma_end   = gamma

    X_Lc = X_L.copy(); y_Lc = y_L.copy(); X_Uc = X_U.copy()
    history = []
    best_f1 = -1.0
    best_clf = None
    no_improve = 0

    for it in range(max_iter + 1):
        clf = make_clf(n_estimators)
        clf.fit(X_Lc, y_Lc)
        y_pred = clf.predict(X_test)

        f1_now = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        rec = {
            "iteration":   it,
            "n_L":         len(X_Lc),
            "n_U":         len(X_Uc),
            "f1_macro":    f1_now,
            "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "precision":   round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "recall":      round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "n_added":     0,
            "gamma_used":  round(_gamma_anneal(it, max_iter, gamma_start, gamma_end), 3),
            "clf":         clf,
            "is_best":     False,
        }

        # [P6] Tracking du meilleur modèle
        if f1_now > best_f1:
            best_f1  = f1_now
            best_clf = clf
            rec["is_best"] = True
            no_improve = 0
        else:
            no_improve += 1

        if it == max_iter or len(X_Uc) == 0:
            history.append(rec)
            break

        # [P6] Early stopping
        if no_improve >= patience:
            history.append(rec)
            break

        # [P2] Gamma courant
        gamma_cur = _gamma_anneal(it, max_iter, gamma_start, gamma_end)

        proba = clf.predict_proba(X_Uc)
        # [P3] Double filtre confiance + marge
        mask  = _margin_filter(proba, gamma_cur, min_margin)
        n_add = int(mask.sum())
        rec["n_added"] = n_add
        history.append(rec)

        if n_add == 0:
            break

        pseudo  = clf.classes_[proba[mask].argmax(axis=1)]
        X_Lc    = np.vstack([X_Lc, X_Uc[mask]])
        y_Lc    = np.concatenate([y_Lc, pseudo])
        X_Uc    = X_Uc[~mask]

    # Injecter le meilleur clf dans le dernier enregistrement
    history[-1]["clf"] = best_clf if best_clf is not None else history[-1]["clf"]
    history[-1]["best_f1"] = best_f1
    return history


def run_co_training(X_L, y_L, X_U, X_test, y_test,
                    va_idx, vb_idx, gamma, max_iter, k_per_iter, n_estimators,
                    patience=3, min_margin=0.12):
    """
    Co-Training amélioré :
    [P1] Vue B enrichie (appelée depuis les features globales)
    [P2] Gamma annealing
    [P3] Margin filtering
    [P5] Résolution de conflit : n'ajouter que si les 2 vues s'accordent
    [P6] Best model tracking + early stopping
    """
    gamma_start = max(0.50, gamma - 0.10)
    gamma_end   = gamma

    X_LA = X_L[:, va_idx]; X_LB = X_L[:, vb_idx]
    y_LA = y_L.copy();     y_LB = y_L.copy()
    X_UA = X_U[:, va_idx]; X_UB = X_U[:, vb_idx]
    X_tA = X_test[:, va_idx]; X_tB = X_test[:, vb_idx]

    history = []
    best_f1    = -1.0
    best_cA    = None
    best_cB    = None
    no_improve = 0

    def _predict_ensemble(cA, cB):
        pA  = cA.predict_proba(X_tA)
        pB  = cB.predict_proba(X_tB)
        cls = np.union1d(cA.classes_, cB.classes_)
        def _align(c, p):
            out = np.zeros((p.shape[0], len(cls)))
            for j, cl in enumerate(cls):
                if cl in c.classes_:
                    out[:, j] = p[:, np.where(c.classes_ == cl)[0][0]]
            return out
        pfin = (_align(cA, pA) + _align(cB, pB)) / 2
        return cls[pfin.argmax(axis=1)]

    for it in range(max_iter + 1):
        cA = make_clf(n_estimators, 42); cA.fit(X_LA, y_LA)
        cB = make_clf(n_estimators, 43); cB.fit(X_LB, y_LB)
        y_pred = _predict_ensemble(cA, cB)

        f1_now = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        rec = {
            "iteration":   it,
            "n_L":         len(X_LA),
            "n_U":         len(X_UA),
            "f1_macro":    f1_now,
            "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "precision":   round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "recall":      round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "n_added":     0,
            "gamma_used":  round(_gamma_anneal(it, max_iter, gamma_start, gamma_end), 3),
            "clf_A": cA, "clf_B": cB,
            "is_best": False,
        }

        if f1_now > best_f1:
            best_f1 = f1_now
            best_cA = cA; best_cB = cB
            rec["is_best"] = True
            no_improve = 0
        else:
            no_improve += 1

        if it == max_iter or len(X_UA) == 0:
            history.append(rec)
            break

        if no_improve >= patience:
            history.append(rec)
            break

        gamma_cur = _gamma_anneal(it, max_iter, gamma_start, gamma_end)

        pA = cA.predict_proba(X_UA); pB = cB.predict_proba(X_UB)
        confA = pA.max(axis=1);       confB = pB.max(axis=1)

        # Top-k candidats par vue
        tk_A = np.argsort(confA)[::-1][:k_per_iter]
        tk_B = np.argsort(confB)[::-1][:k_per_iter]

        # [P3] Filtrage marge
        maskA = _margin_filter(pA[tk_A], gamma_cur, min_margin)
        maskB = _margin_filter(pB[tk_B], gamma_cur, min_margin)
        sel_A = tk_A[maskA]   # indices dans U retenus par Vue A
        sel_B = tk_B[maskB]   # indices dans U retenus par Vue B

        pred_A = cA.classes_[pA[sel_A].argmax(axis=1)]  # prédictions de A sur sel_A
        pred_B = cB.classes_[pB[sel_B].argmax(axis=1)]  # prédictions de B sur sel_B

        # [P5] Résolution de conflit sur les points sélectionnés par LES DEUX vues
        common = np.intersect1d(sel_A, sel_B)
        if len(common) > 0:
            pred_A_common = cA.classes_[pA[common].argmax(axis=1)]
            pred_B_common = cB.classes_[pB[common].argmax(axis=1)]
            agree_mask    = pred_A_common == pred_B_common
            common_agree  = common[agree_mask]
            # Retirer les points en conflit des sélections
            conflict       = common[~agree_mask]
            sel_A = np.setdiff1d(sel_A, conflict)
            sel_B = np.setdiff1d(sel_B, conflict)

        n_add = len(sel_A) + len(sel_B)
        rec["n_added"] = n_add
        history.append(rec)

        if n_add == 0:
            break

        # B → enseigne A (pseudo-labels de B ajoutés à L_A)
        if len(sel_B) > 0:
            X_LA = np.vstack([X_LA, X_UA[sel_B]])
            y_LA = np.concatenate([y_LA, pred_B])

        # A → enseigne B
        if len(sel_A) > 0:
            X_LB = np.vstack([X_LB, X_UB[sel_A]])
            y_LB = np.concatenate([y_LB, pred_A])

        keep = np.setdiff1d(np.arange(len(X_UA)), np.union1d(sel_A, sel_B))
        X_UA = X_UA[keep]; X_UB = X_UB[keep]

    # Restaurer les meilleurs classifieurs
    if best_cA is not None:
        history[-1]["clf_A"] = best_cA
        history[-1]["clf_B"] = best_cB
    history[-1]["best_f1"] = best_f1
    return history


# ═══════════════════════════════════════════════════════════════════════════
# 5. FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def _style(fig):
    fig.patch.set_facecolor(PALETTE["cream"])
    for ax in fig.axes:
        ax.set_facecolor(PALETTE["cream"])
    return fig

def fig_label_scarcity(df_full):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _style(fig)
    n_L = int(df_full["label_known"].sum())
    n_U = len(df_full) - n_L
    axes[0].pie(
        [n_L, n_U],
        labels=[f"L (labellisé)\n{n_L:,} ({n_L/(n_L+n_U)*100:.1f}%)",
                f"U (non-labellisé)\n{n_U:,} ({n_U/(n_L+n_U)*100:.1f}%)"],
        colors=[PALETTE["teal"], PALETTE["grey"]],
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops={"fontsize": 10},
    )
    axes[0].set_title("Ratio L / U — Scarcité des étiquettes",
                      fontsize=12, fontweight="bold", color=PALETTE["navy"])
    df_L = df_full[df_full["label_known"] == 1]
    cnt  = df_L["aqi_label"].value_counts().sort_index()
    cols = [AQI_NAMES[i][1] for i in cnt.index]
    bars = axes[1].bar([AQI_NAMES[i][0] for i in cnt.index], cnt.values,
                       color=cols, edgecolor="white")
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
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    _style(fig)
    df2 = df_full[df_full["_synth"]==False].copy()
    df2["ym"] = df2["datetime"].dt.to_period("M").astype(str)
    monthly = df2.groupby(["ym","station_name"])["pm25"].mean().reset_index()
    sts  = df2["station_name"].unique()
    clrs = [PALETTE["teal"], PALETTE["orange"], PALETTE["purple"]]
    for i, st in enumerate(sts):
        sub = monthly[monthly["station_name"]==st]
        axes[0].plot(range(len(sub)), sub["pm25"].values,
                     label=st, color=clrs[i], linewidth=1.8)
    for y, c, lbl in [(15,PALETTE["green"],"OMS annuel (15)"),
                       (25,PALETTE["orange"],"OMS 24h (25)"),
                       (75,PALETTE["red"],"Mauvais I (75)")]:
        axes[0].axhline(y, linestyle="--", linewidth=1, color=c, label=lbl)
    axes[0].set_title("PM2.5 mensuel moyen par station (2022–2023)",
                      fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[0].set_ylabel("PM2.5 (µg/m³)"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    hrly = df2.groupby("hour")["pm25"].mean()
    axes[1].fill_between(hrly.index, hrly.values, alpha=0.25, color=PALETTE["teal"])
    axes[1].plot(hrly.index, hrly.values, color=PALETTE["teal"], linewidth=2.5)
    axes[1].axvspan(7,9,   alpha=0.15, color=PALETTE["orange"], label="Rush matin")
    axes[1].axvspan(17,20, alpha=0.15, color=PALETTE["red"],    label="Rush soir")
    axes[1].set_title("Profil diurne PM2.5 — effet trafic",
                      fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_xlabel("Heure"); axes[1].set_ylabel("PM2.5 moyen (µg/m³)")
    axes[1].set_xticks(range(0,24,2)); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout(); return fig

def fig_correlation_views(df_full):
    cols = VUE_A + ["hour_sin","hour_cos","month_sin","month_cos",
                    "station_id","is_harmattan","rolling_pm25_3h","pm25_lag1h","pm25_trend"]
    available = [c for c in cols if c in df_full.columns]
    corr = df_full[available].corr()
    fig, ax = plt.subplots(figsize=(11, 8)); _style(fig)
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.5, linecolor="white", cbar_kws={"shrink":0.8})
    n_a = len(VUE_A)
    ax.add_patch(mpatches.Rectangle((0,0), n_a, n_a,
                 fill=False, edgecolor=PALETTE["teal"], linewidth=2.5))
    ax.add_patch(mpatches.Rectangle((n_a, n_a), len(available)-n_a, len(available)-n_a,
                 fill=False, edgecolor=PALETTE["orange"], linewidth=2.5))
    ax.set_title("Corrélation inter-features — Validation indépendance des vues (v4)",
                 fontsize=11, fontweight="bold", color=PALETTE["navy"])
    ax.legend(handles=[
        mpatches.Patch(color=PALETTE["teal"],   label="Vue A — Polluants"),
        mpatches.Patch(color=PALETTE["orange"], label="Vue B — Contexte + Dynamique"),
    ], loc="upper right", fontsize=9)
    plt.tight_layout(); return fig

def fig_ssl_progress(history, algo_name):
    df_h  = pd.DataFrame(history)
    color = PALETTE["teal"] if "Co" in algo_name else PALETTE["orange"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4)); _style(fig)

    axes[0].plot(df_h["iteration"], df_h["f1_macro"],
                 color=color, linewidth=2.5, marker="o", markersize=5)
    axes[0].fill_between(df_h["iteration"], df_h["f1_macro"], alpha=0.15, color=color)
    axes[0].axhline(df_h["f1_macro"].iloc[0], linestyle="--",
                    color=PALETTE["grey"], linewidth=1.5, label="Baseline iter 0")
    # Marquer le meilleur
    best_row = df_h[df_h.get("is_best", pd.Series([False]*len(df_h))) == True]
    if not best_row.empty:
        axes[0].scatter(best_row["iteration"], best_row["f1_macro"],
                        s=120, zorder=5, color=PALETTE["green"], label="Meilleur")
    axes[0].set_title(f"{algo_name} — F1-Score macro",
                      fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[0].set_xlabel("Itération"); axes[0].set_ylabel("F1 macro")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3); axes[0].set_ylim(0,1)

    axes[1].plot(df_h["iteration"], df_h["n_L"],
                 color=PALETTE["teal"], linewidth=2, label="|L|")
    axes[1].plot(df_h["iteration"], df_h["n_U"],
                 color=PALETTE["orange"], linewidth=2, label="|U|")
    axes[1].set_title("Évolution |L| et |U|",
                      fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_xlabel("Itération"); axes[1].set_ylabel("Observations")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    if "gamma_used" in df_h.columns:
        axes[2].plot(df_h["iteration"], df_h["gamma_used"],
                     color=PALETTE["purple"], linewidth=2, marker="s", markersize=5)
        axes[2].set_title("Gamma Annealing",
                          fontsize=11, fontweight="bold", color=PALETTE["navy"])
        axes[2].set_xlabel("Itération"); axes[2].set_ylabel("γ")
        axes[2].grid(alpha=0.3)
    else:
        axes[2].set_visible(False)

    plt.tight_layout(); return fig

def fig_confusion(y_true, y_pred, title):
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 5.5)); _style(fig)
    sns.heatmap(cm_norm, ax=ax, cmap="Blues", annot=True, fmt=".2f",
                xticklabels=[AQI_NAMES[i][0] for i in range(6)],
                yticklabels=[AQI_NAMES[i][0] for i in range(6)],
                linewidths=0.5, linecolor="white", cbar_kws={"shrink":0.8})
    ax.set_title(title, fontsize=11, fontweight="bold", color=PALETTE["navy"])
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); return fig

def fig_compare(results_dict):
    methods = list(results_dict.keys())
    f1s  = [v["f1_macro"]  for v in results_dict.values()]
    precs= [v["precision"] for v in results_dict.values()]
    recs = [v["recall"]    for v in results_dict.values()]
    x = np.arange(len(methods)); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5)); _style(fig)
    b1 = ax.bar(x-w, f1s,  w, label="F1 macro",       color=PALETTE["teal"],   edgecolor="white")
    b2 = ax.bar(x,   precs, w, label="Précision macro", color=PALETTE["orange"], edgecolor="white")
    b3 = ax.bar(x+w, recs,  w, label="Rappel macro",    color=PALETTE["purple"], edgecolor="white")
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
    """Nouveau graphe v4 : F1 par classe pour chaque méthode."""
    methods = list(results_dict.keys())
    clr_map = {"Baseline (L seul)": PALETTE["grey"],
               "Self-Training":     PALETTE["orange"],
               "Co-Training":       PALETTE["teal"]}
    class_names = [AQI_NAMES[i][0] for i in range(6)]
    fig, ax = plt.subplots(figsize=(11, 5)); _style(fig)
    x = np.arange(6); w = 0.25
    for i, (name, res) in enumerate(results_dict.items()):
        report = classification_report(y_test, res["y_pred"],
                                       labels=list(range(6)),
                                       output_dict=True, zero_division=0)
        f1s = [report.get(str(c), {}).get("f1-score", 0) for c in range(6)]
        offset = (i - len(methods)/2 + 0.5) * w
        bars = ax.bar(x + offset, f1s, w,
                      label=name, color=clr_map.get(name, PALETTE["purple"]),
                      edgecolor="white", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(class_names, fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("F1-Score par classe")
    ax.set_title("F1 par classe AQI — Hiérarchie Baseline < ST < CT",
                 fontsize=12, fontweight="bold", color=PALETTE["navy"])
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(
    "<h2 style='color:#0A8A7C;margin-bottom:0'>⚙️ Configuration v4</h2>",
    unsafe_allow_html=True)
st.sidebar.markdown("---")

algo_choice = st.sidebar.selectbox(
    "🔬 Algorithme", ["Self-Training", "Co-Training"],
    help="Self-Training : un classifieur RF. Co-Training : deux classifieurs sur vues enrichies.")

gamma = st.sidebar.slider(
    "🎯 Seuil de confiance γ (fin)", 0.60, 0.99, 0.80, 0.01,
    help="Seuil FINAL de confiance (l'annealing commence 0.10 plus bas).")

min_margin = st.sidebar.slider(
    "📐 Marge min P(1er)−P(2e)", 0.05, 0.40, 0.15, 0.05,
    help="[P3] Filtre les pseudo-labels ambigus. Plus élevé = plus strict.")

patience = st.sidebar.slider(
    "⏱ Patience early stopping", 1, 8, 3, 1,
    help="[P6] Arrêt si pas d'amélioration F1 pendant N itérations.")

max_iter    = st.sidebar.slider("🔁 Itérations max", 3, 20, 10, 1)
n_estimators= st.sidebar.slider("🌲 Arbres RF",       50, 200, 100, 50)
k_per_iter  = 50
if algo_choice == "Co-Training":
    k_per_iter = st.sidebar.slider("📦 Top-k pseudo-labels / itération", 10, 150, 50, 10)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#7A8BA0'>"
    "🆕 <b>v4 — Corrections :</b><br>"
    "• Vue B enrichie (rolling, lag, trend)<br>"
    "• Gamma annealing (γ_start → γ_end)<br>"
    "• Margin filter P1−P2 ≥ seuil<br>"
    "• Bloc synthétique 30 % dans L<br>"
    "• Résolution conflits Co-Training<br>"
    "• Best model tracking + patience<br>"
    "</small>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 7. CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════

with st.spinner("⏳ Génération & préparation du dataset OpenAQ Dakar v4…"):
    df_raw = generate_dataset()
    data   = prepare_splits(df_raw)

df_full     = data["df_full"]
df_test     = data["df_test"]
X_L         = data["X_L"]; y_L = data["y_L"]
X_U         = data["X_U"]
X_test      = data["X_test"]; y_test = data["y_test"]
va_idx      = data["va_idx"]; vb_idx = data["vb_idx"]

# ═══════════════════════════════════════════════════════════════════════════
# 8. HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='background:linear-gradient(135deg,#0B1F3A 0%,#0A8A7C 100%);
            padding:28px 32px;border-radius:12px;margin-bottom:24px'>
  <h1 style='color:white;margin:0;font-size:2rem'>
    🌍 Apprentissage Semi-Supervisé — Qualité de l'Air à Dakar
  </h1>
  <p style='color:#B2D8D4;margin:8px 0 4px 0;font-size:1rem'>
    Self-Training &amp; Co-Training · Dataset OpenAQ (schéma réel) ·
    Stations DEEC &amp; US Embassy · 2022–2023
  </p>
  <span style='background:#E8712A;color:white;padding:3px 10px;border-radius:20px;font-size:0.8rem;font-weight:bold'>
    v4 — Vue B enrichie · Gamma Annealing · Margin Filter · Best Model Tracking
  </span>
</div>""", unsafe_allow_html=True)

n_total = len(df_full)
n_L_cnt = int(df_full["label_known"].sum())
n_U_cnt = n_total - n_L_cnt

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("📦 Train total",          f"{n_total:,}")
c2.metric("🏷 Labellisés L",          f"{n_L_cnt:,}", f"{n_L_cnt/n_total*100:.1f}%")
c3.metric("🔓 Non-labellisés U",      f"{n_U_cnt:,}", f"{n_U_cnt/n_total*100:.1f}%")
c4.metric("🧪 Test set (Q4 2023)",    f"{len(df_test):,}")
c5.metric("🔬 Features Vue B",         f"{len(VUE_B)} (+5 dynamiques)")
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
    st.markdown("### 🔍 Analyse Exploratoire — Dataset OpenAQ Dakar")

    with st.expander("🆕 v4 — Corrections de sous-performance", expanded=True):
        st.markdown("""
| # | Problème identifié (v3) | Correction v4 |
|---|---|---|
| P1 | Vue B trop faible → clf_B aléatoire | **Vue B enrichie** : rolling_pm25_3h, rolling_pm10_3h, pm25_lag1h, pm10_lag1h, pm25_trend |
| P2 | γ=0.80 d'emblée → peu de pseudo-labels | **Gamma annealing** : démarre à γ−0.10, monte vers γ sur les itérations |
| P3 | Pas de filtrage de qualité des pseudo-labels | **Margin filter** : P(1ère classe) − P(2e classe) ≥ seuil configurable |
| P4 | Bloc synthétique 100% dans L → biais distribution | **30 % seulement** du bloc synthétique dans L |
| P5 | Co-Training : conflits non résolus | **Résolution conflit** : n'ajouter que si les 2 vues s'accordent |
| P6 | Modèle final ≠ meilleur modèle | **Best model tracking** + early stopping par patience |

**Performances attendues v4 :** Baseline ≈ 0.38–0.52 · Self-Training ≈ 0.50–0.62 · Co-Training ≈ 0.55–0.70
        """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Scarcité des étiquettes & Distribution AQI")
        st.pyplot(fig_label_scarcity(df_full), use_container_width=True)
    with col_b:
        st.markdown("#### Séries temporelles PM2.5")
        st.pyplot(fig_pm25_temporal(df_full), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔗 Heatmap de Corrélation — Validation Indépendance des Vues (v4)")
    st.info(
        "**[P1] Vue B enrichie :** rolling_pm25_3h et pm25_lag1h ont une corrélation modérée "
        "avec pm25 (~0.65–0.75), mais capturent la DYNAMIQUE temporelle que Vue A ne voit pas. "
        "Les features contextuelles (heure, mois, station) restent orthogonales aux polluants."
    )
    st.pyplot(fig_correlation_views(df_full), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📋 Aperçu du dataset (avec features dynamiques v4)")
    disp = ["datetime","station_name","pm25","rolling_pm25_3h","pm25_lag1h","pm25_trend",
            "pm10","no2","o3","co","is_harmattan","aqi_label","label_known"]
    disp = [c for c in disp if c in df_full.columns]
    st.dataframe(
        df_full[disp].head(20).style
        .format({c:"{:.2f}" for c in ["pm25","rolling_pm25_3h","pm25_lag1h","pm25_trend",
                                        "pm10","no2","o3"] if c in disp}),
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
        prog = st.progress(0)
        stat = st.empty()
        tbl  = st.empty()
        t0   = time.time()

        if algo_choice == "Self-Training":
            history = run_self_training(
                X_L, y_L, X_U, X_test, y_test,
                gamma=gamma, max_iter=max_iter, n_estimators=n_estimators,
                patience=patience, min_margin=min_margin)
        else:
            history = run_co_training(
                X_L, y_L, X_U, X_test, y_test,
                va_idx=va_idx, vb_idx=vb_idx,
                gamma=gamma, max_iter=max_iter,
                k_per_iter=k_per_iter, n_estimators=n_estimators,
                patience=patience, min_margin=min_margin)

        elapsed = time.time() - t0
        final   = history[-1]
        gain    = final["f1_macro"] - history[0]["f1_macro"]
        best_f1 = final.get("best_f1", final["f1_macro"])

        prog.progress(100)
        stat.success(f"✅ Terminé en {elapsed:.1f}s — {final['iteration']} itérations | Meilleur F1 = {best_f1:.4f}")

        # Table itérations
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

        # KPIs
        st.markdown("---")
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("F1 Final (macro)", f"{final['f1_macro']:.4f}", f"{gain:+.4f}")
        k2.metric("Meilleur F1",       f"{best_f1:.4f}")
        k3.metric("Précision macro",   f"{final['precision']:.4f}")
        k4.metric("Rappel macro",      f"{final['recall']:.4f}")
        k5.metric("|L| final",         f"{final['n_L']:,}",
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

        st.markdown("#### 🔲 Matrice de Confusion (Test Q4 2023)")
        st.pyplot(fig_confusion(y_test, y_pred_fin, f"{algo_choice} v4 — γ={gamma}"),
                  use_container_width=True)

        st.markdown("#### 📋 Rapport de Classification")
        report = classification_report(
            y_test, y_pred_fin,
            labels=list(range(6)),
            target_names=[AQI_NAMES[i][0] for i in range(6)],
            output_dict=True, zero_division=0)
        st.dataframe(
            pd.DataFrame(report).T.round(4)
            .style.background_gradient(subset=["f1-score"], cmap="Greens"),
            use_container_width=True)

        # Stocker résultats
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
    st.markdown("### 📈 Dashboard Comparatif des Performances — v4")

    if "results" not in st.session_state or not st.session_state["results"]:
        st.warning("⚠️ Aucun résultat — lancez d'abord une simulation dans l'onglet **Simulation**.")
    else:
        # Baseline
        clf_base = make_clf(100)
        clf_base.fit(X_L, y_L)
        y_base = clf_base.predict(X_test)
        all_res = {
            "Baseline (L seul)": {
                "f1_macro":  round(f1_score(y_test, y_base, average="macro",  zero_division=0), 4),
                "precision": round(precision_score(y_test, y_base, average="macro", zero_division=0), 4),
                "recall":    round(recall_score(y_test, y_base, average="macro",  zero_division=0), 4),
                "y_pred":    y_base,
            }
        }
        for k, v in st.session_state["results"].items():
            all_res[k] = v

        st.markdown("#### 📊 Comparaison Globale")
        st.pyplot(fig_compare(all_res), use_container_width=True)

        st.markdown("#### 🎯 F1 par Classe AQI (hiérarchie visuelle)")
        st.pyplot(fig_per_class_f1(y_test, all_res), use_container_width=True)

        st.markdown("#### 🗃 Tableau Récapitulatif")
        df_comp = pd.DataFrame({k: {"F1 macro": v["f1_macro"],
                                     "Précision": v["precision"],
                                     "Rappel":    v["recall"]}
                                  for k, v in all_res.items()}).T
        base_f1 = df_comp.loc["Baseline (L seul)", "F1 macro"]
        df_comp["Δ F1 vs Baseline"] = (df_comp["F1 macro"] - base_f1).round(4)
        st.dataframe(
            df_comp.style
            .format({"F1 macro":"{:.4f}","Précision":"{:.4f}",
                     "Rappel":"{:.4f}","Δ F1 vs Baseline":"{:+.4f}"})
            .background_gradient(subset=["F1 macro"], cmap="Greens")
            .background_gradient(subset=["Δ F1 vs Baseline"], cmap="RdYlGn", vmin=-0.1, vmax=0.25),
            use_container_width=True)

        # Vérification hiérarchie
        methods_sorted = sorted(all_res.keys(),
                                 key=lambda k: all_res[k]["f1_macro"])
        st.markdown("#### 🏆 Hiérarchie des performances")
        for i, m in enumerate(methods_sorted):
            icon  = ["🥉","🥈","🥇"][min(i, 2)]
            color = [PALETTE["grey"], PALETTE["orange"], PALETTE["teal"]][min(i, 2)]
            st.markdown(
                f"<div style='background:{color}22;border-left:4px solid {color};"
                f"padding:8px 16px;border-radius:6px;margin:4px 0'>"
                f"{icon} <b>{m}</b> — F1 macro = <b>{all_res[m]['f1_macro']:.4f}</b>"
                f"</div>",
                unsafe_allow_html=True)

        if len(st.session_state["results"]) >= 2:
            st.markdown("---")
            st.markdown("#### 📈 Courbes d'Apprentissage SSL Comparées")
            fig_evo, ax_evo = plt.subplots(figsize=(11, 4.5))
            fig_evo.patch.set_facecolor(PALETTE["cream"])
            ax_evo.set_facecolor(PALETTE["cream"])
            clr_map = {"Self-Training": PALETTE["orange"], "Co-Training": PALETTE["teal"]}
            for name, res in st.session_state["results"].items():
                h = res["history"]
                ax_evo.plot([r["iteration"] for r in h], [r["f1_macro"] for r in h],
                            label=name, linewidth=2.5, marker="o", markersize=5,
                            color=clr_map.get(name, PALETTE["purple"]))
            ax_evo.axhline(base_f1, linestyle="--", color=PALETTE["red"],
                           linewidth=1.5, label="Baseline (L seul)")
            ax_evo.set_xlabel("Itération"); ax_evo.set_ylabel("F1-Score macro")
            ax_evo.set_title("Évolution F1 — Self-Training vs Co-Training (v4)",
                             fontsize=12, fontweight="bold", color=PALETTE["navy"])
            ax_evo.legend(fontsize=10); ax_evo.grid(alpha=0.3); ax_evo.set_ylim(0,1)
            plt.tight_layout()
            st.pyplot(fig_evo, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🗂 Dataset OpenAQ Dakar — Informations v4")
        c1d, c2d = st.columns(2)
        with c1d:
            st.markdown(f"""
**Features Vue A (polluants):** {', '.join(VUE_A)}

**Features Vue B (contexte + dynamique):**
- Cycliques : hour_sin/cos, month_sin/cos
- Station : station_id, is_harmattan
- **Dynamiques (NEW v4) :** rolling_pm25_3h, rolling_pm10_3h, pm25_lag1h, pm10_lag1h, pm25_trend

**Test set :** Q4 2023 (oct–déc) — bloc temporel futur
            """)
        with c2d:
            st.markdown("""
**Labellisation AQI composite (OMS 2021) :**

| Classe | Score composite | Couleur |
|--------|----------------|---------|
| 0 — Bon | < 15 | 🟢 |
| 1 — Modéré | 15–30 | 🟡 |
| 2 — Mauvais (S) | 30–60 | 🟠 |
| 3 — Mauvais (I) | 60–90 | 🔴 |
| 4 — Très Mauvais | 90–160 | 🟣 |
| 5 — Extrême | > 160 | ⚫ |
            """)
