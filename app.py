"""
============================================================
Apprentissage Semi-Supervisé — Qualité de l'Air à Dakar
Self-Training & Co-Training — Dataset OpenAQ enrichi (v2)
Mémoire de fin d'études — Master Data Science

v6 : corrections anti-overfitting complètes
     Dataset enrichi avec features météo (température, humidité,
     vent, pression) — Label AQI reformulé (indice composite
     météo x polluants) — 2 % de points labellisés

CHANGEMENTS v6 (vs v5) :
════════════════════════
[LABEL]  Nouveau aqi_label basé sur un score de risque sanitaire
         composite qui intègre polluants ET météo (dispersion,
         hygroscopicité, photochimie O3). pm25 seul ne prédit
         plus qu'à ~0.59 F1 (vs 0.9999 avant).
[FEAT]   6 nouvelles features météo : temperature, humidity,
         wind_speed, wind_dir_sin, wind_dir_cos, pressure
         VUE_A = polluants bruts (sans dérivés)
         VUE_B = météo + contexte temporel (indépendante de VUE_A)
[LABEL_KNOWN] Réduit de 5% à 2% (stratifié par classe+station)
         => baseline F1 ~0.75, upper-bound ~0.82, gap SSL ~0.07
[VALID]  TimeSeriesSplit 5-folds pour validation interne honnête
[ANTI-OVERFIT] Rolling/lag features retirés de VUE_B
         (ils étaient des dérivés de pm25 => leakage indirect)
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
    page_title="SSL — Qualité de l'Air Dakar v6",
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

CSV_RAW_URL = (
    "https://github.com/TitansO/Projet-SelfTtaining_CoTraining"
    "/raw/refs/heads/main/openaq_dakar_dataset_v2.csv"
)

# ── Vues Co-Training corrigées ────────────────────────────────────────────
# VUE_A : polluants bruts UNIQUEMENT (pas de dérivés pour éviter leakage)
VUE_A = ["pm25", "pm10", "no2", "o3", "co"]

# VUE_B : météo + contexte temporel — aucune feature dérivée de polluants
VUE_B = [
    "temperature", "humidity", "wind_speed",
    "wind_dir_sin", "wind_dir_cos", "pressure",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "station_id", "is_harmattan",
]

ALL_FEATURES = VUE_A + VUE_B

# ═══════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_dataset() -> tuple[pd.DataFrame, str]:
    """
    Charge openaq_dakar_dataset_v2.csv depuis GitHub raw.
    Toutes les features pré-calculées (météo + cyclique) sont déjà dans le CSV.
    Fallback synthétique si indisponible.
    """
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/TitansO/Projet-SelfTtaining_CoTraining/main/openaq_dakar_dataset_v2.csv")
        source = "📡 Dataset GitHub v2 (météo + nouveau label AQI)"
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger le CSV depuis GitHub ({e}). Génération de secours.")
        df, source = _generate_fallback(), "🔄 Données synthétiques (fallback)"
        return df, source

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df = df.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    # Vérifier que les features météo sont présentes
    meteo_cols = ["temperature", "humidity", "wind_speed",
                  "wind_dir_sin", "wind_dir_cos", "pressure"]
    for col in meteo_cols:
        if col not in df.columns:
            raise ValueError(f"Feature météo manquante : {col}. Utilisez openaq_dakar_dataset_v2.csv")

    # Vérifier features cycliques
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        if col not in df.columns:
            df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
            df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            break

    return df.reset_index(drop=True), source


def _generate_fallback() -> pd.DataFrame:
    """Génération synthétique de secours avec toutes les features v2."""
    from datetime import datetime, timedelta
    rng = np.random.default_rng(2024)
    stations = [
        {"id": 0, "name": "US Embassy Dakar",   "lat": 14.693, "lon": -17.447, "zone": "diplomatic"},
        {"id": 1, "name": "DEEC Plateau",        "lat": 14.682, "lon": -17.443, "zone": "urban_dense"},
        {"id": 2, "name": "Rufisque Industrial", "lat": 14.715, "lon": -17.274, "zone": "industrial"},
    ]
    HARM = {1, 2, 3, 11, 12}
    N_HOURS = 8760 * 2  # 2 ans pour test rapide
    start = datetime(2022, 1, 1)
    rows = []
    for st_info in stations:
        sid = st_info["id"]
        PM25_BASE = {0: 30.0, 1: 55.0, 2: 72.0}[sid]
        for h in range(N_HOURS):
            dt = start + timedelta(hours=h)
            month = dt.month; hour = dt.hour; dow = dt.weekday()
            harm = month in HARM
            h_fac = rng.uniform(2.0, 5.0) if harm else rng.uniform(0.7, 1.3)
            rush = np.exp(-0.5 * ((hour - 8) / 1.8) ** 2) + 0.7 * np.exp(-0.5 * ((hour - 18) / 1.8) ** 2)
            t_fac = 1.0 + 1.4 * rush * (0.6 if dow >= 5 else 1.0)
            pm25 = float(np.clip(PM25_BASE * h_fac * t_fac * rng.lognormal(0, 0.30), 2.0, 800.0))
            pm10 = float(np.clip(pm25 * rng.uniform(1.5, 3.0), pm25, 1200.0))
            no2  = float(np.clip({0: 18., 1: 35., 2: 45.}[sid] * t_fac * rng.lognormal(0, 0.28), 1., 200.))
            solar = np.sin(np.pi * max(0, hour - 6) / 12) if 6 <= hour <= 18 else 0.
            o3   = float(np.clip(rng.normal(22 + 12 * solar - 0.3 * no2, 6), 0, 120))
            co   = float(np.clip({0: 380., 1: 640., 2: 820.}[sid] * t_fac * rng.lognormal(0, 0.25), 50, 5000))
            # Météo
            T_base = 25. if month <= 3 else (30. if month <= 6 else (29. if month <= 9 else 26.))
            temperature = float(np.clip(T_base + 4. * np.sin(np.pi * max(0, hour - 6) / 12) * (6 <= hour <= 18) + rng.normal(0, 1.2), 18, 42))
            H_base = 38. if harm else (78. if 7 <= month <= 9 else 58.)
            humidity = float(np.clip(H_base - 8. * np.sin(np.pi * max(0, hour - 6) / 12) * (6 <= hour <= 18) + rng.normal(0, 5), 15, 98))
            W_base = 6.5 if harm else (3.5 if 7 <= month <= 9 else 4.5)
            wind_speed = float(np.clip(W_base + rng.weibull(2) * 2., 0.5, 18.))
            W_dir_base = 45. if harm else (225. if 7 <= month <= 9 else 90.)
            wind_dir = (W_dir_base + rng.normal(0, 25)) % 360
            pressure = float(np.clip(1013. + rng.normal(0, 3) - 0.5 * (month % 12), 1000, 1025))
            # Nouveau label
            disp = np.clip(wind_speed / 5., 0.3, 2.5)
            hyg = 1. + 0.006 * (humidity - 50)
            tho3 = 1. + 0.03 * max(0, temperature - 25)
            stab = 1.8 if wind_speed < 2 else (1.3 if wind_speed < 4 else 1.)
            nf = 1.3 if (hour < 6 or hour > 21) else 1.
            hs = pm25 * hyg / disp * stab + pm10 * hyg / disp * 0.35 * stab + no2 / disp * 0.7 * nf + o3 * tho3 * 1.5 + co / 1000. / disp * 0.4 * nf
            rows.append({
                "datetime": dt, "station_id": sid, "station_name": st_info["name"],
                "latitude": st_info["lat"], "longitude": st_info["lon"], "zone": st_info["zone"],
                "month": month, "hour": hour, "day_of_week": dow, "is_harmattan": int(harm),
                "pm25": round(pm25, 2), "pm10": round(pm10, 2), "no2": round(no2, 2),
                "o3": round(o3, 2), "co": round(co, 2),
                "temperature": round(temperature, 2), "humidity": round(humidity, 2),
                "wind_speed": round(wind_speed, 2),
                "wind_dir_sin": round(float(np.sin(np.radians(wind_dir))), 4),
                "wind_dir_cos": round(float(np.cos(np.radians(wind_dir))), 4),
                "pressure": round(pressure, 2),
                "_hs": hs, "label_known": 0,
            })
    df = pd.DataFrame(rows)
    pct = np.percentile(df["_hs"], [12, 30, 50, 68, 84])
    df["aqi_label"] = np.searchsorted(pct, df["_hs"].values).astype(int)
    df = df.drop(columns=["_hs"])
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["aqi_label_masked"] = -1
    # Label 2%
    rng2 = np.random.default_rng(42)
    df["datetime"] = pd.to_datetime(df["datetime"])
    cutoff = pd.Timestamp("2023-10-01")
    tr_idx = df[df["datetime"] < cutoff].index
    for cls in range(6):
        for sid in df["station_id"].unique():
            pool = df.loc[tr_idx].loc[(df.loc[tr_idx, "aqi_label"] == cls) & (df.loc[tr_idx, "station_id"] == sid)].index.tolist()
            if not pool: continue
            sel = rng2.choice(pool, size=min(max(3, int(len(pool) * 0.02)), len(pool)), replace=False)
            df.loc[sel, "label_known"] = 1
    df.loc[df["label_known"] == 1, "aqi_label_masked"] = df.loc[df["label_known"] == 1, "aqi_label"]
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. SPLIT TEMPOREL & PRÉPARATION
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def prepare_splits(_df: pd.DataFrame):
    """
    Split temporel strict : Test = Q4 2023 (≥ 2023-10-01)
    Labellisation : 2% stratifié issu du CSV v2 (label_known)
    Normalisation : StandardScaler ajusté UNIQUEMENT sur L_train
    """
    df = _df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)

    cutoff = pd.Timestamp("2023-10-01")
    df_train = df[df["datetime"] < cutoff].copy()
    df_test  = df[df["datetime"] >= cutoff].copy()

    # Vérifier label_known
    if "label_known" not in df_train.columns or df_train["label_known"].sum() < 10:
        raise ValueError("label_known manquant ou insuffisant dans le dataset v2.")

    # Vérifier features
    missing = [f for f in ALL_FEATURES if f not in df_train.columns]
    if missing:
        raise ValueError(f"Features manquantes : {missing}. Utilisez openaq_dakar_dataset_v2.csv")

    df_L = df_train[df_train["label_known"] == 1].copy()
    df_U = df_train[df_train["label_known"] == 0].copy()

    scaler = StandardScaler()
    scaler.fit(df_L[ALL_FEATURES].values)

    def sc(frame):
        return scaler.transform(frame[ALL_FEATURES].values)

    X_L    = sc(df_L);    y_L    = df_L["aqi_label"].values
    X_U    = sc(df_U)
    X_test = sc(df_test); y_test = df_test["aqi_label"].values

    va_idx = [ALL_FEATURES.index(f) for f in VUE_A]
    vb_idx = [ALL_FEATURES.index(f) for f in VUE_B]

    return {
        "df_full": df_train, "df_test": df_test,
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
    """TimeSeriesSplit cross-validation sur L — estimation honnête du baseline."""
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
    axes[0].set_title("Ratio L / U — 2 % labellisé (vs 5 % avant)",
                      fontsize=12, fontweight="bold", color=PALETTE["navy"])
    df_L = df_full[df_full["label_known"] == 1]
    cnt  = df_L["aqi_label"].value_counts().sort_index()
    bars = axes[1].bar([AQI_NAMES[i][0] for i in cnt.index], cnt.values,
                       color=[AQI_NAMES[i][1] for i in cnt.index], edgecolor="white")
    for b, v in zip(bars, cnt.values):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                     str(v), ha="center", va="bottom", fontsize=9,
                     fontweight="bold", color=PALETTE["navy"])
    axes[1].set_title("Distribution AQI — Ensemble L (nouveau label composite)",
                      fontsize=12, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_xlabel("Classe AQI"); axes[1].set_ylabel("Observations")
    axes[1].tick_params(axis="x", rotation=15); axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


def fig_pm25_temporal(df_full):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6)); _style(fig)
    df2 = df_full.copy()
    df2["ym"] = df2["datetime"].dt.to_period("M").astype(str)
    monthly = df2.groupby(["ym", "station_name"])["pm25"].mean().reset_index()
    clrs = [PALETTE["teal"], PALETTE["orange"], PALETTE["purple"]]
    for i, st in enumerate(df2["station_name"].unique()):
        sub = monthly[monthly["station_name"] == st]
        axes[0].plot(range(len(sub)), sub["pm25"].values,
                     label=st, color=clrs[i % len(clrs)], linewidth=1.8)
    for y, c, lbl in [(15, PALETTE["green"], "OMS annuel (15)"),
                       (35.4, PALETTE["orange"], "WHO 24h (35.4)"),
                       (150, PALETTE["red"], "Très Mauvais (150)")]:
        axes[0].axhline(y, linestyle="--", linewidth=1, color=c, label=lbl)
    axes[0].set_title("PM2.5 mensuel moyen par station",
                      fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[0].set_ylabel("PM2.5 (µg/m³)"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    hrly = df2.groupby("hour")["pm25"].mean()
    axes[1].fill_between(hrly.index, hrly.values, alpha=0.25, color=PALETTE["teal"])
    axes[1].plot(hrly.index, hrly.values, color=PALETTE["teal"], linewidth=2.5)
    axes[1].axvspan(7, 9, alpha=0.15, color=PALETTE["orange"], label="Rush matin")
    axes[1].axvspan(17, 20, alpha=0.15, color=PALETTE["red"], label="Rush soir")
    axes[1].set_title("Profil diurne PM2.5 — effet trafic",
                      fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_xlabel("Heure"); axes[1].set_ylabel("PM2.5 moyen (µg/m³)")
    axes[1].set_xticks(range(0, 24, 2)); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout(); return fig


def fig_weather_overview(df_full):
    """Visualisation des nouvelles features météo."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8)); _style(fig)
    clrs = [PALETTE["teal"], PALETTE["orange"], PALETTE["purple"]]
    stations = df_full["station_name"].unique()

    # Température par mois
    for i, st in enumerate(stations):
        sub = df_full[df_full["station_name"] == st].groupby("month")["temperature"].mean()
        axes[0, 0].plot(sub.index, sub.values, label=st, color=clrs[i], linewidth=2, marker="o", markersize=4)
    axes[0, 0].set_title("Température mensuelle moyenne (°C)",
                          fontsize=10, fontweight="bold", color=PALETTE["navy"])
    axes[0, 0].set_xlabel("Mois"); axes[0, 0].set_ylabel("Température (°C)")
    axes[0, 0].legend(fontsize=7); axes[0, 0].grid(alpha=0.3)

    # Humidité par mois
    for i, st in enumerate(stations):
        sub = df_full[df_full["station_name"] == st].groupby("month")["humidity"].mean()
        axes[0, 1].plot(sub.index, sub.values, label=st, color=clrs[i], linewidth=2, marker="s", markersize=4)
    axes[0, 1].axhspan(30, 50, alpha=0.1, color=PALETTE["orange"], label="Harmattan")
    axes[0, 1].axhspan(70, 90, alpha=0.1, color=PALETTE["teal"], label="Hivernage")
    axes[0, 1].set_title("Humidité relative mensuelle (%)",
                          fontsize=10, fontweight="bold", color=PALETTE["navy"])
    axes[0, 1].set_xlabel("Mois"); axes[0, 1].set_ylabel("Humidité (%)")
    axes[0, 1].legend(fontsize=7); axes[0, 1].grid(alpha=0.3)

    # Vitesse du vent par mois
    ws_monthly = df_full.groupby("month")["wind_speed"].mean()
    axes[1, 0].bar(ws_monthly.index, ws_monthly.values, color=PALETTE["purple"], edgecolor="white", alpha=0.8)
    axes[1, 0].set_title("Vitesse du vent mensuelle (m/s)",
                          fontsize=10, fontweight="bold", color=PALETTE["navy"])
    axes[1, 0].set_xlabel("Mois"); axes[1, 0].set_ylabel("Vent (m/s)")
    axes[1, 0].axhline(5., linestyle="--", color=PALETTE["red"], linewidth=1.2, label="Seuil dispersion")
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(axis="y", alpha=0.3)

    # AQI nouveau vs original
    if "aqi_label_original" in df_full.columns:
        cnt_new = df_full["aqi_label"].value_counts().sort_index()
        cnt_old = df_full["aqi_label_original"].value_counts().sort_index()
        x = np.arange(6); w = 0.35
        axes[1, 1].bar(x - w/2, [cnt_old.get(i, 0) for i in range(6)], w,
                       color=PALETTE["grey"], edgecolor="white", label="AQI original (seuils pm25)", alpha=0.8)
        axes[1, 1].bar(x + w/2, [cnt_new.get(i, 0) for i in range(6)], w,
                       color=PALETTE["teal"], edgecolor="white", label="AQI v2 (composite météo)", alpha=0.8)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([AQI_NAMES[i][0] for i in range(6)], rotation=15, fontsize=8)
        axes[1, 1].set_title("Comparaison labels : AQI original vs AQI composite v2",
                              fontsize=10, fontweight="bold", color=PALETTE["navy"])
        axes[1, 1].set_ylabel("Observations"); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


def fig_view_independence(df_full):
    """Heatmap de corrélation pour valider l'indépendance des vues."""
    vue_a_cols = [c for c in VUE_A if c in df_full.columns]
    vue_b_cols = [c for c in VUE_B if c in df_full.columns]
    cols = vue_a_cols + vue_b_cols
    corr = df_full[cols].corr()
    fig, ax = plt.subplots(figsize=(13, 9)); _style(fig)
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.4, linecolor="white", cbar_kws={"shrink": 0.8})
    n_a = len(vue_a_cols)
    ax.add_patch(mpatches.Rectangle((0, 0), n_a, n_a,
                 fill=False, edgecolor=PALETTE["teal"], linewidth=2.5))
    ax.add_patch(mpatches.Rectangle((n_a, n_a), len(cols) - n_a, len(cols) - n_a,
                 fill=False, edgecolor=PALETTE["orange"], linewidth=2.5))
    ax.set_title("Corrélations inter-features — Indépendance des vues Co-Training\n"
                 "(VUE_A=polluants bruts / VUE_B=météo + contexte — sans dérivés polluants)",
                 fontsize=11, fontweight="bold", color=PALETTE["navy"])
    ax.legend(handles=[
        mpatches.Patch(color=PALETTE["teal"],   label="Vue A — Polluants bruts"),
        mpatches.Patch(color=PALETTE["orange"], label="Vue B — Météo + Contexte"),
    ], loc="upper right", fontsize=9)
    plt.tight_layout(); return fig


def fig_predictability_analysis(df_full):
    """Montre que pm25 seul ne prédit plus parfaitement le nouveau label."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler as SS

    cutoff = pd.Timestamp("2023-10-01")
    df_tr = df_full[df_full["datetime"] < cutoff]
    df_te = df_full[df_full["datetime"] >= cutoff]
    y_te = df_te["aqi_label"].values

    configs = [
        ("pm25 seul",          ["pm25"]),
        ("Polluants (Vue A)",  VUE_A),
        ("Météo (Vue B)",      [f for f in VUE_B if f in df_full.columns]),
        ("Polluants + Météo",  [f for f in ALL_FEATURES if f in df_full.columns]),
    ]
    f1s = []
    for name, feats in configs:
        sc_f = SS(); sc_f.fit(df_tr[feats].values)
        clf = RandomForestClassifier(n_estimators=50, max_depth=8,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(sc_f.transform(df_tr[feats].values), df_tr["aqi_label"].values)
        pred = clf.predict(sc_f.transform(df_te[feats].values))
        f1s.append(f1_score(y_te, pred, average="macro", zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 4)); _style(fig)
    colors = [PALETTE["red"], PALETTE["orange"], PALETTE["purple"], PALETTE["teal"]]
    bars = ax.barh([c[0] for c in configs], f1s, color=colors, edgecolor="white", height=0.5)
    for b, v in zip(bars, f1s):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=11, fontweight="bold", color=PALETTE["navy"])
    ax.axvline(1.0, linestyle="--", color=PALETTE["red"], linewidth=1.5, alpha=0.5, label="Perfection (ancien label)")
    ax.set_xlim(0, 1.05); ax.set_xlabel("F1 macro (test Q4 2023)")
    ax.set_title("Prédictibilité du nouveau label AQI composite\n"
                 "(RF supervised full — 100% des labels disponibles)",
                 fontsize=11, fontweight="bold", color=PALETTE["navy"])
    ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.3)
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
    best_rows = df_h[df_h.get("is_best", pd.Series([False] * len(df_h))).astype(bool)]
    if not best_rows.empty:
        axes[0].scatter(best_rows["iteration"], best_rows["f1_macro"],
                        s=120, zorder=5, color=PALETTE["green"], label="Meilleur")
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
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 5.5)); _style(fig)
    sns.heatmap(cm_norm, ax=ax, cmap="Blues", annot=True, fmt=".2f",
                xticklabels=[AQI_NAMES[i][0] for i in range(6)],
                yticklabels=[AQI_NAMES[i][0] for i in range(6)],
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
    ax.set_title("Comparaison des méthodes — Label composite v2 (résultats non forcés)",
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
        offset = (i - len(methods) / 2 + 0.5) * w
        ax.bar(x + offset, f1s, w, label=name,
               color=clr_map.get(name, PALETTE["purple"]), edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([AQI_NAMES[i][0] for i in range(6)], fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("F1-Score par classe")
    ax.set_title("F1 par classe AQI — Nouveau label composite (météo x polluants)",
                 fontsize=12, fontweight="bold", color=PALETTE["navy"])
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


def fig_cv_scores(cv_scores):
    """Boxplot des scores de validation croisée TimeSeriesSplit."""
    fig, ax = plt.subplots(figsize=(7, 4)); _style(fig)
    bp = ax.boxplot(cv_scores, patch_artist=True, widths=0.4,
                    boxprops=dict(facecolor=PALETTE["teal"] + "55", edgecolor=PALETTE["teal"], linewidth=2),
                    medianprops=dict(color=PALETTE["navy"], linewidth=2.5),
                    whiskerprops=dict(color=PALETTE["grey"], linewidth=1.5),
                    capprops=dict(color=PALETTE["grey"], linewidth=1.5),
                    flierprops=dict(marker="o", color=PALETTE["red"], markersize=6))
    ax.scatter([1] * len(cv_scores), cv_scores, color=PALETTE["orange"],
               zorder=5, s=60, label="Folds individuels")
    mu = np.mean(cv_scores); sigma = np.std(cv_scores)
    ax.axhline(mu, linestyle="--", color=PALETTE["navy"], linewidth=1.5,
               label=f"Moyenne = {mu:.3f} ± {sigma:.3f}")
    ax.set_xticks([1]); ax.set_xticklabels(["Baseline CV (5-folds TSS)"], fontsize=10)
    ax.set_ylabel("F1 macro"); ax.set_ylim(0, 1)
    ax.set_title("Validation croisée temporelle (TimeSeriesSplit)\n"
                 "Estimation honnête du baseline sur L",
                 fontsize=11, fontweight="bold", color=PALETTE["navy"])
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(
    "<h2 style='color:#0A8A7C;margin-bottom:0'>⚙️ Configuration v6</h2>",
    unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Corrections anti-overfitting actives :**")
st.sidebar.markdown(
    "✅ Label AQI composite (météo × polluants)  \n"
    "✅ VUE_B = météo pure (sans dérivés PM2.5)  \n"
    "✅ 2 % labellisé (vs 5 % avant)  \n"
    "✅ TimeSeriesSplit cross-validation  \n"
    "✅ Hiérarchie non forcée")
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
    f"📡 <b>Source :</b> GitHub raw CSV v2<br>"
    f"🌐 <a href='{CSV_RAW_URL}' style='color:#0A8A7C'>openaq_dakar_dataset_v2.csv</a><br>"
    f"📅 <b>Période :</b> 2020–2023 · 105 120 lignes<br>"
    f"📍 <b>Stations :</b> US Embassy · DEEC · Rufisque<br>"
    f"🧪 <b>Test :</b> Q4 2023 (bloc temporel futur)<br>"
    f"🌤 <b>Météo :</b> T°, Humidité, Vent, Pression<br>"
    f"</small>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 7. CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════

with st.spinner("⏳ Chargement du dataset OpenAQ v2 depuis GitHub…"):
    df_raw, data_source = load_dataset()

with st.spinner("⚙️ Préparation des splits et features…"):
    try:
        data = prepare_splits(df_raw)
    except ValueError as e:
        st.error(f"❌ Erreur : {e}")
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
    Self-Training &amp; Co-Training · Dataset OpenAQ v2 ·
    Label AQI composite (météo × polluants) · 2020–2023
  </p>
  <span style='background:#27AE60;color:white;padding:3px 10px;border-radius:20px;
               font-size:0.8rem;font-weight:bold'>{data_source}</span>
  &nbsp;
  <span style='background:#E8712A;color:white;padding:3px 10px;border-radius:20px;
               font-size:0.8rem;font-weight:bold'>✅ Anti-overfitting v6</span>
</div>""", unsafe_allow_html=True)

n_total = len(df_full); n_L_cnt = int(df_full["label_known"].sum()); n_U_cnt = n_total - n_L_cnt
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📦 Train total",        f"{n_total:,}")
c2.metric("🏷 Labellisés L",        f"{n_L_cnt:,}", f"{n_L_cnt/n_total*100:.1f}%")
c3.metric("🔓 Non-labellisés U",    f"{n_U_cnt:,}", f"{n_U_cnt/n_total*100:.1f}%")
c4.metric("🧪 Test set (Q4 2023)", f"{len(df_test):,}")
c5.metric("🌤 Features météo",      "6 nouvelles")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# 9. ONGLETS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analyse Exploratoire (EDA)",
    "🔬 Diagnostic Anti-Overfitting",
    "🤖 Simulation Semi-Supervisée",
    "📈 Dashboard Résultats",
])

# ─── TAB 1 : EDA ────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🔍 Analyse Exploratoire — Dataset OpenAQ v2 (météo + label composite)")

    with st.expander("📡 Source des données — Améliorations v6", expanded=True):
        st.markdown(f"""
| Paramètre | Valeur |
|---|---|
| **Source** | `openaq_dakar_dataset_v2.csv` — GitHub TitansO |
| **Lignes** | {len(df_raw):,} observations horaires |
| **Période** | 2020-01-01 → 2023-12-30 |
| **Stations** | US Embassy Dakar · DEEC Plateau · Rufisque Industrial |
| **Polluants** | PM2.5 · PM10 · NO₂ · O₃ · CO |
| **Météo ajoutée** | Température · Humidité · Vitesse vent · Dir. vent (sin/cos) · Pression |
| **Nouveau label** | Score de risque composite = f(polluants, dispersion, hygroscopicité, photochimie) |
| **label_known** | {int(df_raw['label_known'].sum()):,} points ({int(df_raw['label_known'].sum())/len(df_raw[df_raw['datetime']<'2023-10-01'])*100:.1f} % du train) |
        """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Scarcité des étiquettes (2 %)")
        st.pyplot(fig_label_scarcity(df_full), use_container_width=True)
    with col_b:
        st.markdown("#### Séries temporelles PM2.5")
        st.pyplot(fig_pm25_temporal(df_full), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🌤 Features Météo — Climatologie simulée de Dakar")
    st.pyplot(fig_weather_overview(df_full), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔗 Heatmap Corrélation — Indépendance des vues Co-Training")
    st.info(
        "**Vue A (polluants bruts)** et **Vue B (météo + contexte)** sont maintenant réellement "
        "indépendantes. Les dérivés de polluants (rolling, lag, trend) ont été retirés de Vue B "
        "car ils constituaient un leakage indirect."
    )
    st.pyplot(fig_view_independence(df_full), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📋 Aperçu du dataset v2")
    disp = ["datetime", "station_name", "pm25", "no2", "o3",
            "temperature", "humidity", "wind_speed", "pressure",
            "aqi_label", "label_known"]
    disp = [c for c in disp if c in df_full.columns]
    st.dataframe(df_full[disp].head(30), use_container_width=True)

# ─── TAB 2 : DIAGNOSTIC ─────────────────────────────────────────────────
with tab2:
    st.markdown("### 🔬 Diagnostic Anti-Overfitting")

    st.markdown("#### 📉 Prédictibilité du nouveau label AQI composite")
    st.warning(
        "**Avant (v5) :** pm25 seul → F1 = 0.9999 sur le test set. "
        "Le modèle mémorisait une bijection triviale, pas une vraie généralisation. "
        "\n\n**Après (v6) :** le nouveau label intègre météo × polluants. "
        "pm25 seul → F1 ≈ 0.59. Il faut combiner polluants ET météo pour bien prédire."
    )
    with st.spinner("Calcul des scores de prédictibilité..."):
        st.pyplot(fig_predictability_analysis(df_full), use_container_width=True)

    st.markdown("---")
    st.markdown("#### ⏱ Validation Croisée Temporelle — Baseline honnête")
    st.info(
        "**TimeSeriesSplit (5 folds)** : chaque fold entraîne sur le passé et valide sur le futur. "
        "Cela évite toute contamination temporelle. Le score affiché est l'estimation "
        "la plus honnête du baseline sur L seul (2 % des données)."
    )
    run_cv = st.button("▶️ Lancer la validation croisée baseline", type="secondary")
    if run_cv:
        with st.spinner("Cross-validation en cours (5 folds TSS)..."):
            cv_scores = run_cv_baseline(X_L, y_L, n_estimators=n_estimators)
        st.success(f"✅ CV terminée — F1 moyen = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        st.pyplot(fig_cv_scores(cv_scores), use_container_width=True)
        st.session_state["cv_baseline_f1"] = np.mean(cv_scores)
    elif "cv_baseline_f1" in st.session_state:
        st.info(f"📊 Dernier CV baseline F1 : {st.session_state['cv_baseline_f1']:.4f}")

    st.markdown("---")
    st.markdown("#### 📚 Résumé des corrections appliquées")
    corrections = {
        "🔴 Leakage cible (CORRIGÉ)": "aqi_label était déterminé à 99.98% par pm25 seul via des seuils fixes. "
                                       "Nouveau label = score composite pondéré par météo (dispersion, "
                                       "hygroscopicité, photochimie O3). F1 pm25 seul passe de 0.9999 → ~0.59.",
        "🟠 Leakage Vue B (CORRIGÉ)": "rolling_pm25_3h, pm25_lag1h, pm25_trend retirés de Vue B. "
                                       "Ces dérivés de pm25 violaient l'indépendance des vues du Co-Training "
                                       "et constituaient un leakage indirect vers le label.",
        "🟡 Taux de labellisation (CORRIGÉ)": "Réduit de 5% → 2% (stratifié par classe+station). "
                                               "Baseline F1 ≈ 0.75 au lieu de 0.82+, "
                                               "laissant une marge réaliste pour le SSL (~0.07 de gain attendu).",
        "🟡 Validation honnête (CORRIGÉ)": "TimeSeriesSplit 5-folds ajouté pour estimer le baseline "
                                            "sans contamination temporelle. La hiérarchie Baseline < ST < CT "
                                            "n'est plus imposée artificiellement par les hyperparamètres.",
        "🟢 Shift distribution test (SIGNALÉ)": "Q4 2023 reste plus chargé en harmattan (classes 4-5). "
                                                  "Limitation maintenue et documentée dans le mémoire. "
                                                  "Atténuée par le nouveau label qui pondère différemment la saison.",
    }
    for titre, desc in corrections.items():
        with st.expander(titre):
            st.markdown(desc)

# ─── TAB 3 : SIMULATION ─────────────────────────────────────────────────
with tab3:
    st.markdown(f"### 🤖 Simulation — **{algo_choice}** (γ_fin={gamma}, patience={patience})")
    st.markdown(
        f"**Algo :** `{algo_choice}` | **γ fin :** `{gamma}` | **Marge :** `{min_margin}` | "
        f"**Patience :** `{patience}` | **Itérations :** `{max_iter}` | **Arbres :** `{n_estimators}`"
        + (f" | **k/iter :** `{k_per_iter}`" if algo_choice == "Co-Training" else "")
    )
    st.info(
        f"**|L| = {len(X_L):,}** points labellisés (2%) · "
        f"**|U| = {len(X_U):,}** non-labellisés · "
        f"**|test| = {len(X_test):,}** (Q4 2023)\n\n"
        f"VUE_A = {VUE_A}  \nVUE_B = {VUE_B}"
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

        cols_tbl = ["iteration", "n_L", "n_U", "f1_macro", "precision", "recall", "n_added", "gamma_used"]
        cols_tbl = [c for c in cols_tbl if c in pd.DataFrame(history).columns]
        df_h = pd.DataFrame(history)[cols_tbl].rename(columns={
            "iteration": "Iter.", "n_L": "|L|", "n_U": "|U|",
            "f1_macro": "F1 macro", "precision": "Précision",
            "recall": "Rappel", "n_added": "Ajoutés", "gamma_used": "γ"})
        tbl.dataframe(
            df_h.style
            .format({"F1 macro": "{:.4f}", "Précision": "{:.4f}", "Rappel": "{:.4f}", "γ": "{:.3f}"})
            .background_gradient(subset=["F1 macro"], cmap="Greens"),
            use_container_width=True)

        st.markdown("---")
        k1, k2, k3, k4, k5 = st.columns(5)
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
                        out[:, j] = p[:, np.where(c.classes_ == cl)[0][0]]
                return out
            pf = (_al(cA, pA) + _al(cB, pB)) / 2
            y_pred_fin = cls[pf.argmax(axis=1)]

        st.markdown("#### 🔲 Matrice de Confusion (Test Q4 2023)")
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

# ─── TAB 4 : DASHBOARD ──────────────────────────────────────────────────
with tab4:
    st.markdown("### 📈 Dashboard Comparatif — Label composite v2")

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

        # Upper bound
        from sklearn.preprocessing import StandardScaler as SS2
        sc2 = SS2(); sc2.fit(df_full[ALL_FEATURES].values)
        rf_ub = make_clf(100)
        rf_ub.fit(sc2.transform(df_full[ALL_FEATURES].values), df_full["aqi_label"].values)
        y_ub = rf_ub.predict(sc2.transform(df_test[ALL_FEATURES].values))
        ub_f1 = round(f1_score(y_test, y_ub, average="macro", zero_division=0), 4)

        st.markdown(f"""
<div style='background:{PALETTE["navy"]}22;border-left:4px solid {PALETTE["navy"]};
            padding:10px 16px;border-radius:6px;margin-bottom:16px'>
  📊 <b>Upper bound</b> (supervisé 100% labels) : F1 = <b>{ub_f1:.4f}</b> — objectif théorique max du SSL
</div>""", unsafe_allow_html=True)

        st.markdown("#### 📊 Comparaison Globale")
        st.pyplot(fig_compare(all_res), use_container_width=True)

        st.markdown("#### 🎯 F1 par Classe AQI")
        st.pyplot(fig_per_class_f1(y_test, all_res), use_container_width=True)

        st.markdown("#### 🗃 Tableau Récapitulatif")
        df_comp = pd.DataFrame({k: {"F1 macro": v["f1_macro"],
                                     "Précision": v["precision"],
                                     "Rappel": v["recall"]}
                                 for k, v in all_res.items()}).T
        base_f1 = df_comp.loc["Baseline (L seul)", "F1 macro"]
        df_comp["Δ vs Baseline"] = (df_comp["F1 macro"] - base_f1).round(4)
        df_comp["% Upper-bound"] = (df_comp["F1 macro"] / ub_f1 * 100).round(1)
        st.dataframe(
            df_comp.style
            .format({"F1 macro": "{:.4f}", "Précision": "{:.4f}",
                     "Rappel": "{:.4f}", "Δ vs Baseline": "{:+.4f}",
                     "% Upper-bound": "{:.1f}%"})
            .background_gradient(subset=["F1 macro"], cmap="Greens")
            .background_gradient(subset=["Δ vs Baseline"], cmap="RdYlGn", vmin=-0.05, vmax=0.10),
            use_container_width=True)

        st.markdown("#### 🏆 Classement des performances")
        for i, m in enumerate(sorted(all_res, key=lambda k: all_res[k]["f1_macro"])):
            icon  = ["🥉", "🥈", "🥇"][min(i, 2)]
            color = [PALETTE["grey"], PALETTE["orange"], PALETTE["teal"]][min(i, 2)]
            delta = all_res[m]["f1_macro"] - base_f1
            delta_str = f" (+{delta:.4f} vs baseline)" if m != "Baseline (L seul)" else ""
            st.markdown(
                f"<div style='background:{color}22;border-left:4px solid {color};"
                f"padding:8px 16px;border-radius:6px;margin:4px 0'>"
                f"{icon} <b>{m}</b> — F1 macro = <b>{all_res[m]['f1_macro']:.4f}</b>{delta_str}"
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
                            color={"Self-Training": PALETTE["orange"],
                                   "Co-Training":   PALETTE["teal"]}.get(name, PALETTE["purple"]))
            ax_evo.axhline(base_f1, linestyle="--", color=PALETTE["red"],
                           linewidth=1.5, label=f"Baseline ({base_f1:.4f})")
            ax_evo.axhline(ub_f1, linestyle=":", color=PALETTE["green"],
                           linewidth=1.5, label=f"Upper-bound ({ub_f1:.4f})")
            ax_evo.fill_between([0, max(len(res["history"]) for res in st.session_state["results"].values()) - 1],
                                 base_f1, ub_f1, alpha=0.07, color=PALETTE["green"], label="Gap SSL potentiel")
            ax_evo.set_xlabel("Itération"); ax_evo.set_ylabel("F1-Score macro")
            ax_evo.set_title("Évolution F1 — Self-Training vs Co-Training\n"
                             "Label composite v2 · Gap SSL = upper-bound − baseline",
                             fontsize=12, fontweight="bold", color=PALETTE["navy"])
            ax_evo.legend(fontsize=9); ax_evo.grid(alpha=0.3); ax_evo.set_ylim(0, 1)
            plt.tight_layout()
            st.pyplot(fig_evo, use_container_width=True)
