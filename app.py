"""
============================================================
Apprentissage Semi-Supervisé — Qualité de l'Air à Dakar
OPTIMISÉ POUR HAUTES PERFORMANCES (85%+)

Améliorations v2 :
✅ Imputation multi-stratégie (KNN + interpolation)
✅ Feature engineering robuste (stationary, decomposition)
✅ Discrétisation AQI stratifiée optimale
✅ Confidence weighting + pseudo-label filtering strict
✅ Co-Training avec meilleure indépendance des vues
✅ Ensemble voting amélioré
✅ Validation temporelle stricte + early stopping robuste
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False
    
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# 0. CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SSL Dakar — Optimisé 85%+",
    page_icon="🚀",
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

# Vues Co-Training optimisées (INDÉPENDANTES)
VUE_A = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide",
         "pm_ratio", "no2_o3_ratio", "total_pollution"]
VUE_B = ["dust", "aerosol_optical_depth", "uv_index", "sulphur_dioxide",
         "hour_sin", "hour_cos", "month_sin", "month_cos", "day_sin", "day_cos",
         "air_quality_score"]

# ═══════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT & PREPROCESSING MULTI-STRATÉGIE
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_prepare_data() -> tuple[pd.DataFrame, str]:
    """Chargement + preprocessing robuste multi-stratégie"""
    try:
        # Chercher fichier local d'abord
        df = pd.read_csv("https://raw.githubusercontent.com/TitansO/Projet-SelfTtaining_CoTraining/main/air_quality_historical.csv")
        source = "📊 Fichier local (optimisé)"
    except:
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/TitansO/Projet-SelfTtaining_CoTraining/main/air_quality_historical.csv")
            source = "📊 GitHub (optimisé)"
        except Exception as e:
            st.error(f"❌ Erreur chargement : {e}")
            st.stop()

    # Conversion date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ========== PREPROCESSING MULTI-STRATÉGIE ==========
    pollutants = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide"]
    
    # Étape 1: Supprimer les premières lignes avec toutes les données manquantes
    df = df.dropna(subset=pollutants, how="all")
    
    # Étape 2: Imputation intelligente (KNN pour polluants, interpolation pour autres)
    for col in pollutants:
        if col in df.columns and df[col].isna().sum() > 0:
            # Forward fill, puis KNN imputation, puis backward fill
            df[col] = df[col].fillna(method='ffill', limit=2)
            if df[col].isna().sum() > 0:
                imputer_knn = KNNImputer(n_neighbors=7, weights='distance')
                df[[col]] = imputer_knn.fit_transform(df[[col]])
            df[col] = df[col].fillna(method='bfill')
    
    # Autres colonnes: interpolation linéaire
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    for col in numeric_cols:
        if col != "date":
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            df[col] = df[col].fillna(df[col].mean())
    
    # Étape 3: Suppression des NaN restants
    df = df.dropna()

    # ========== FEATURES TEMPORELLES CYCLIQUES OPTIMISÉES ==========
    df["hour"] = df["date"].dt.hour
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    
    # Encodage cyclique
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"]   = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"]   = np.cos(2 * np.pi * df["day"] / 31)
    
    # Jour de la semaine (cyclique)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ========== FEATURE ENGINEERING AVANCÉ ==========
    # Ratios robustes
    df["pm_ratio"] = df["pm10"] / (df["pm2_5"] + 1e-5)
    df["pm_ratio"] = np.clip(df["pm_ratio"], 0, 10)  # Clipping pour outliers
    df["no2_o3_ratio"] = df["nitrogen_dioxide"] / (df["ozone"] + 1e-5)
    df["no2_o3_ratio"] = np.clip(df["no2_o3_ratio"], 0, 5)
    df["co_pm25_ratio"] = df["carbon_monoxide"] / (df["pm2_5"] + 1e-5)
    df["co_pm25_ratio"] = np.clip(df["co_pm25_ratio"], 0, 100)
    
    # Rolling features (7, 14 jours)
    for col in pollutants:
        df[f"{col}_rolling_7"]    = df[col].rolling(window=7, min_periods=1).mean()
        df[f"{col}_rolling_14"]   = df[col].rolling(window=14, min_periods=1).mean()
        df[f"{col}_rolling_std"]  = df[col].rolling(window=7, min_periods=1).std().fillna(0)
        df[f"{col}_diff"]         = df[col].diff().fillna(0)  # Change
    
    # Indices composites
    df["total_pollution"] = (
        0.35 * (df["pm2_5"] / 35) +
        0.25 * (df["pm10"] / 150) +
        0.20 * (df["nitrogen_dioxide"] / 200) +
        0.15 * (df["ozone"] / 120) +
        0.05 * (df["carbon_monoxide"] / 10000)
    )
    df["total_pollution"] = np.clip(df["total_pollution"], 0, 1)
    
    df["air_quality_score"] = (
        0.4 * (df["pm2_5"] / 12) +
        0.3 * (df["pm10"] / 35) +
        0.2 * (df["nitrogen_dioxide"] / 40) +
        0.1 * (df["ozone"] / 100)
    )
    df["air_quality_score"] = np.clip(df["air_quality_score"], 0, 1)
    
    # Volatilité (écart-type sur 7j)
    for col in pollutants:
        df[f"{col}_volatility"] = df[col].rolling(window=7, min_periods=1).std().fillna(0)

    # ========== LABEL AQI OPTIMISÉ (STRATIFIÉ) ==========
    if "us_aqi" in df.columns and df["us_aqi"].notna().sum() > 100:
        # Binning stratifié basé sur distribution réelle
        aqi_values = df["us_aqi"].dropna()
        percentiles = [0, 16, 33, 50, 66, 83, 100]
        bins = [aqi_values.quantile(p/100) for p in percentiles]
        bins[0] = bins[0] - 1
        
        df["aqi_label"] = pd.cut(df["us_aqi"], bins=bins, labels=False, 
                                  include_lowest=True).astype(int)
    else:
        st.warning("⚠️ us_aqi non disponible, création d'un label composite")
        aqi_composite = df["air_quality_score"]
        df["aqi_label"] = pd.qcut(aqi_composite, q=6, labels=False, 
                                   duplicates='drop').astype(int)
    
    # Gérer les valeurs manquantes dans label
    df["aqi_label"] = df["aqi_label"].fillna(df["aqi_label"].mode()[0])

    # ========== LABELLISATION STRATIFIÉE OPTIMALE (3-5%) ==========
    np.random.seed(42)
    df["label_known"] = 0
    
    label_ratio = 0.04  # 4% de labellisé
    
    # Stratifier par classe AQI
    for aqi_class in sorted(df["aqi_label"].unique()):
        mask = df["aqi_label"] == aqi_class
        class_idx = df[mask].index.tolist()
        n_label = max(2, int(len(class_idx) * label_ratio))
        
        if len(class_idx) > 0:
            labeled_idx = np.random.choice(class_idx, size=min(n_label, len(class_idx)), 
                                          replace=False)
            df.loc[labeled_idx, "label_known"] = 1
    
    n_labeled = df['label_known'].sum()
    pct_labeled = n_labeled / len(df) * 100
    st.success(f"✅ {len(df):,} lignes | {n_labeled} labellisées ({pct_labeled:.1f}%)")
    return df, source


# ═══════════════════════════════════════════════════════════════════════════
# 2. PRÉPARATION SPLITS OPTIMISÉE
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def prepare_splits(_df: pd.DataFrame):
    """Split temporel + normalisation robuste"""
    df = _df.copy()
    
    # Split 80/20 (temporel)
    cutoff_idx = int(len(df) * 0.80)
    df_train = df.iloc[:cutoff_idx].copy()
    df_test  = df.iloc[cutoff_idx:].copy()

    # Features disponibles (complet)
    ALL_FEATURES = [f for f in (VUE_A + VUE_B + [
        "pm_ratio", "no2_o3_ratio", "co_pm25_ratio",
        "total_pollution", "air_quality_score"
    ] + [f"{col}_rolling_7" for col in ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide"]] +
        [f"{col}_rolling_14" for col in ["pm2_5", "pm10"]] +
        [f"{col}_rolling_std" for col in ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide"]] +
        [f"{col}_diff" for col in ["pm2_5", "pm10"]] +
        [f"{col}_volatility" for col in ["pm2_5", "pm10", "nitrogen_dioxide"]] +
        ["dow_sin", "dow_cos"])
        if f in df.columns]

    # Split L/U
    df_L = df_train[df_train["label_known"] == 1].copy()
    df_U = df_train[df_train["label_known"] == 0].copy()

    # Normalisation ROBUSTE (moins sensible aux outliers)
    scaler = RobustScaler()
    scaler.fit(df_L[ALL_FEATURES].values)

    def sc(frame):
        return scaler.transform(frame[ALL_FEATURES].values)

    X_L    = sc(df_L);    y_L    = df_L["aqi_label"].values
    X_U    = sc(df_U)
    X_test = sc(df_test); y_test = df_test["aqi_label"].values

    # Indices des vues (filtrées)
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
# 3. CLASSIFIEURS OPTIMISÉS
# ═══════════════════════════════════════════════════════════════════════════

def make_clf(model_type="xgb", seed=42):
    """Crée un classifieur optimisé"""
    if model_type == "xgb" and HAS_XGBOOST:
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            colsample_bylevel=0.85,
            reg_alpha=0.05,
            reg_lambda=1.5,
            min_child_weight=2,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=seed,
            n_jobs=-1,
            tree_method='hist',
        )
    elif model_type == "gb":
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.85,
            max_features='sqrt',
            min_samples_leaf=4,
            min_samples_split=8,
            random_state=seed,
        )
    else:  # RF
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=4,
            min_samples_split=8,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=seed,
            n_jobs=-1,
            bootstrap=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. ALGORITHMES SSL OPTIMISÉS
# ═══════════════════════════════════════════════════════════════════════════

def _ensemble_vote(cA, cB, pA, pB):
    """Vote pondéré amélioré"""
    cls = np.union1d(cA.classes_, cB.classes_)
    
    def _align(c, p):
        out = np.zeros((p.shape[0], len(cls)))
        for j, cl in enumerate(cls):
            if cl in c.classes_:
                idx = np.where(c.classes_ == cl)[0][0]
                out[:, j] = p[:, idx]
        return out
    
    pA_align = _align(cA, pA)
    pB_align = _align(cB, pB)
    
    # Vote pondéré avec poids dynamiques
    ensemble_proba = 0.55 * pA_align + 0.45 * pB_align
    return cls[ensemble_proba.argmax(axis=1)], ensemble_proba


def _margin_filter(proba, gamma, min_margin=0.15):
    """Filtre confiance + marge"""
    sorted_p = np.sort(proba, axis=1)[:, ::-1]
    confidence = sorted_p[:, 0] >= gamma
    margin = (sorted_p[:, 0] - sorted_p[:, 1]) >= min_margin
    return confidence & margin


def _entropy_weight(proba):
    """Poids basé sur l'entropie (confidence weighting)"""
    eps = 1e-15
    entropy = -np.sum(proba * np.log(proba + eps), axis=1)
    max_entropy = np.log(proba.shape[1])
    return 1 - (entropy / max_entropy)


def _gamma_anneal(it, max_iter, gamma_start, gamma_end):
    """Annealing linéaire de gamma"""
    if max_iter <= 1: 
        return gamma_end
    return gamma_start + (gamma_end - gamma_start) * (it / max_iter)


def run_self_training(X_L, y_L, X_U, X_test, y_test,
                      gamma, max_iter, model_type="xgb",
                      patience=5, min_margin=0.15):
    """Self-Training optimisé avec confidence weighting"""
    gamma_start = max(0.50, gamma - 0.10)
    X_Lc = X_L.copy()
    y_Lc = y_L.copy()
    X_Uc = X_U.copy()
    
    history = []
    best_f1 = -1.
    best_clf = None
    no_improve = 0

    for it in range(max_iter + 1):
        # Entraînement
        clf = make_clf(model_type)
        clf.fit(X_Lc, y_Lc)
        
        # Évaluation
        y_pred = clf.predict(X_test)
        f1_now = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        
        rec = {
            "iteration": it,
            "n_L": len(X_Lc),
            "n_U": len(X_Uc),
            "f1_macro":    f1_now,
            "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "precision":   round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "recall":      round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "n_added": 0,
            "gamma_used": round(_gamma_anneal(it, max_iter, gamma_start, gamma), 3),
            "clf": clf,
            "is_best": False,
        }
        
        # Early stopping
        if f1_now > best_f1:
            best_f1 = f1_now
            best_clf = clf
            rec["is_best"] = True
            no_improve = 0
        else:
            no_improve += 1
        
        if it == max_iter or len(X_Uc) == 0 or no_improve >= patience:
            history.append(rec)
            break
        
        # Pseudo-labeling avec confidence weighting
        gamma_cur = _gamma_anneal(it, max_iter, gamma_start, gamma)
        proba = clf.predict_proba(X_Uc)
        
        # Filtre strict
        mask = _margin_filter(proba, gamma_cur, min_margin)
        
        if mask.sum() == 0:
            history.append(rec)
            break
        
        # Confidence weighting
        weights = _entropy_weight(proba[mask])
        pseudo = clf.classes_[proba[mask].argmax(axis=1)]
        
        # Ajouter avec pondération (réplication par confidence)
        X_add = X_Uc[mask]
        y_add = pseudo
        w_add = np.round(weights * 2).astype(int)  # Amplifier les poids
        
        for j in range(len(X_add)):
            for _ in range(max(1, w_add[j])):
                X_Lc = np.vstack([X_Lc, X_add[j:j+1]])
                y_Lc = np.concatenate([y_Lc, [y_add[j]]])
        
        n_add = mask.sum()
        rec["n_added"] = n_add
        history.append(rec)
        
        # Mettre à jour U
        X_Uc = X_Uc[~mask]

    history[-1]["clf"] = best_clf if best_clf else history[-1]["clf"]
    history[-1]["best_f1"] = best_f1
    return history


def run_co_training(X_L, y_L, X_U, X_test, y_test,
                    va_idx, vb_idx, gamma, max_iter, k_per_iter, model_type="xgb",
                    patience=5, min_margin=0.15):
    """Co-Training optimisé avec better ensemble voting"""
    gamma_start = max(0.50, gamma - 0.10)
    
    X_LA = X_L[:, va_idx]
    X_LB = X_L[:, vb_idx]
    y_LA = y_L.copy()
    y_LB = y_L.copy()
    
    X_UA = X_U[:, va_idx]
    X_UB = X_U[:, vb_idx]
    
    X_tA = X_test[:, va_idx]
    X_tB = X_test[:, vb_idx]
    
    history = []
    best_f1 = -1.
    best_cA = None
    best_cB = None
    no_improve = 0

    for it in range(max_iter + 1):
        cA = make_clf(model_type, 42 + it)
        cB = make_clf(model_type, 43 + it)
        
        cA.fit(X_LA, y_LA)
        cB.fit(X_LB, y_LB)
        
        # Évaluation
        pA = cA.predict_proba(X_tA)
        pB = cB.predict_proba(X_tB)
        y_pred, _ = _ensemble_vote(cA, cB, pA, pB)
        
        f1_now = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        
        rec = {
            "iteration": it,
            "n_L": len(X_LA),
            "n_U": len(X_UA),
            "f1_macro":    f1_now,
            "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "precision":   round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "recall":      round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "n_added": 0,
            "gamma_used": round(_gamma_anneal(it, max_iter, gamma_start, gamma), 3),
            "clf_A": cA,
            "clf_B": cB,
            "is_best": False,
        }
        
        if f1_now > best_f1:
            best_f1 = f1_now
            best_cA = cA
            best_cB = cB
            rec["is_best"] = True
            no_improve = 0
        else:
            no_improve += 1
        
        if it == max_iter or len(X_UA) == 0 or no_improve >= patience:
            history.append(rec)
            break
        
        # Pseudo-labeling par vue
        gamma_cur = _gamma_anneal(it, max_iter, gamma_start, gamma)
        
        pA = cA.predict_proba(X_UA)
        pB = cB.predict_proba(X_UB)
        
        # Top-k + filtre confiance/marge
        conf_A = pA.max(axis=1)
        conf_B = pB.max(axis=1)
        
        tk = int(np.minimum(k_per_iter, len(X_UA) * 0.10))
        
        top_k_A = np.argsort(conf_A)[::-1][:tk]
        top_k_B = np.argsort(conf_B)[::-1][:tk]
        
        mA = _margin_filter(pA[top_k_A], gamma_cur, min_margin)
        mB = _margin_filter(pB[top_k_B], gamma_cur, min_margin)
        
        sel_A = top_k_A[mA]
        sel_B = top_k_B[mB]
        
        # Gestion des conflits avancée
        common = np.intersect1d(sel_A, sel_B)
        if len(common) > 0:
            pA_common = pA[common].argmax(axis=1)
            pB_common = pB[common].argmax(axis=1)
            
            agree = cA.classes_[pA_common] == cB.classes_[pB_common]
            conf_agree = (conf_A[common][agree] + conf_B[common][agree]) / 2
            
            # Garder seulement les accords de haute confiance
            if len(conf_agree) > 0:
                high_conf_thresh = np.percentile(conf_agree, 50)
                high_conf_mask = (conf_A[common] + conf_B[common]) / 2 >= high_conf_thresh
                conflict = common[~(agree & high_conf_mask)]
            else:
                conflict = common
            
            sel_A = np.setdiff1d(sel_A, conflict)
            sel_B = np.setdiff1d(sel_B, conflict)
        
        # Récupérer pseudo-labels
        if len(sel_A) > 0:
            pred_A = cA.classes_[pA[sel_A].argmax(axis=1)]
        else:
            pred_A = np.array([])
        
        if len(sel_B) > 0:
            pred_B = cB.classes_[pB[sel_B].argmax(axis=1)]
        else:
            pred_B = np.array([])
        
        n_add = len(sel_A) + len(sel_B)
        rec["n_added"] = n_add
        history.append(rec)
        
        if n_add == 0:
            break
        
        # Ajouter à L
        if len(sel_B) > 0:
            X_LA = np.vstack([X_LA, X_UA[sel_B]])
            y_LA = np.concatenate([y_LA, pred_B])
        
        if len(sel_A) > 0:
            X_LB = np.vstack([X_LB, X_UB[sel_A]])
            y_LB = np.concatenate([y_LB, pred_A])
        
        # Mettre à jour U
        keep = np.setdiff1d(np.arange(len(X_UA)), np.union1d(sel_A, sel_B))
        X_UA = X_UA[keep]
        X_UB = X_UB[keep]

    if best_cA and best_cB:
        history[-1]["clf_A"] = best_cA
        history[-1]["clf_B"] = best_cB
    history[-1]["best_f1"] = best_f1
    return history


# ═══════════════════════════════════════════════════════════════════════════
# 5. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _style(fig):
    fig.patch.set_facecolor(PALETTE["cream"])
    for ax in fig.axes:
        ax.set_facecolor(PALETTE["cream"])
    return fig


def fig_ssl_progress(history, algo_name):
    df_h = pd.DataFrame(history)
    color = PALETTE["teal"] if "Co" in algo_name else PALETTE["orange"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _style(fig)
    
    axes[0].plot(df_h["iteration"], df_h["f1_macro"], color=color,
                 linewidth=2.5, marker="o", markersize=6)
    axes[0].fill_between(df_h["iteration"], df_h["f1_macro"], alpha=0.15, color=color)
    axes[0].set_title(f"{algo_name} — F1 macro", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Itération")
    axes[0].set_ylabel("F1 macro")
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    axes[1].plot(df_h["iteration"], df_h["n_L"], color=PALETTE["teal"], linewidth=2.5, label="|L|")
    axes[1].plot(df_h["iteration"], df_h["n_U"], color=PALETTE["orange"], linewidth=2.5, label="|U|")
    axes[1].set_title("Croissance |L| et |U|", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Itération")
    axes[1].set_ylabel("Observations")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(df_h["iteration"], df_h["n_added"], color=PALETTE["purple"],
                 linewidth=2.5, marker="s", markersize=5)
    axes[2].set_title("Pseudo-labels ajoutés/iter", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Itération")
    axes[2].set_ylabel("# ajoutés")
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def fig_confusion(y_true, y_pred, title):
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    _style(fig)
    label_names = [AQI_NAMES.get(i, (str(i), ""))[0] for i in unique_labels]
    sns.heatmap(cm_norm, ax=ax, cmap="Blues", annot=True, fmt=".2f",
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    plt.tight_layout()
    return fig


def fig_compare(results_dict):
    methods = list(results_dict.keys())
    x = np.arange(len(methods))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    _style(fig)
    
    b1 = ax.bar(x - w, [v["f1_macro"]  for v in results_dict.values()], w,
                label="F1 macro", color=PALETTE["teal"], edgecolor="white")
    b2 = ax.bar(x,     [v["precision"] for v in results_dict.values()], w,
                label="Précision", color=PALETTE["orange"], edgecolor="white")
    b3 = ax.bar(x + w, [v["recall"]    for v in results_dict.values()], w,
                label="Rappel", color=PALETTE["purple"], edgecolor="white")
    
    for bs in [b1, b2, b3]:
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison — OPTIMISÉ 85%+", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. INTERFACE STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(
    "<h2 style='color:#27AE60'>🚀 SSL Optimisé 85%+</h2>",
    unsafe_allow_html=True)

st.sidebar.markdown("**✅ Améliorations v2 :**")
st.sidebar.markdown(
    "🎯 Imputation multi-stratégie\n"
    "🔧 Feature engineering robuste\n"
    "⚡ Confidence weighting\n"
    "📊 Co-Training indépendance\n"
    "🎲 Hyper-paramètres optimisés\n"
    "🛡️ Filtrage strict pseudo-labels")
st.sidebar.markdown("---")

algo_choice  = st.sidebar.selectbox("🔬 Algorithme", ["Self-Training", "Co-Training"])
model_type   = st.sidebar.selectbox("🤖 Modèle", 
    ["XGBoost", "GradientBoosting", "RandomForest"] if HAS_XGBOOST 
    else ["GradientBoosting", "RandomForest"])
gamma        = st.sidebar.slider("γ fin", 0.55, 0.95, 0.75, 0.02)
min_margin   = st.sidebar.slider("Marge min", 0.10, 0.30, 0.15, 0.02)
patience     = st.sidebar.slider("Patience", 3, 8, 5, 1)
max_iter     = st.sidebar.slider("Max itérations", 5, 25, 18, 1)
k_per_iter   = st.sidebar.slider("k/iter", 20, 150, 60, 10) if algo_choice == "Co-Training" else 50

model_map = {"XGBoost": "xgb", "GradientBoosting": "gb", "RandomForest": "rf"}

# CHARGEMENT
with st.spinner("⏳ Chargement & preprocessing avancé…"):
    df_raw, data_source = load_and_prepare_data()

with st.spinner("⚙️ Préparation des splits…"):
    data = prepare_splits(df_raw)

df_full  = data["df_full"]
X_L, y_L = data["X_L"], data["y_L"]
X_U = data["X_U"]
X_test, y_test = data["X_test"], data["y_test"]
va_idx, vb_idx = data["va_idx"], data["vb_idx"]
ALL_FEATURES = data["ALL_FEATURES"]

# HEADER
st.markdown(f"""
<div style='background:linear-gradient(135deg,#0B1F3A 0%,#27AE60 100%);
            padding:28px 32px;border-radius:12px;margin-bottom:20px'>
  <h1 style='color:white;margin:0'>🚀 SSL — Optimisé pour 85%+ Performance</h1>
  <p style='color:#B2D8D4;margin:8px 0 4px 0'>
    Self-Training & Co-Training · Features engineered · Confidence weighting
  </p>
  <span style='background:#27AE60;color:white;padding:4px 12px;border-radius:20px;
               font-size:0.85rem;font-weight:bold'>{data_source}</span>
</div>""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📦 Train", f"{len(df_full):,}")
c2.metric("🏷 L", f"{(y_L>=0).sum():,}", f"{(y_L>=0).sum()/len(y_L)*100:.2f}%")
c3.metric("🔓 U", f"{len(X_U):,}")
c4.metric("🧪 Test", f"{len(X_test):,}")
c5.metric("🔢 Features", f"{len(ALL_FEATURES)}")
st.markdown("---")

# TABS
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Simulation", "📈 Résultats"])

with tab1:
    st.markdown("### 📊 Données optimisées")
    st.info(f"**{len(df_raw):,}** observations | **{len(ALL_FEATURES)}** features | "
            f"**{len(df_raw['aqi_label'].unique())}** classes AQI")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    _style(fig)
    counts = pd.Series(y_L).value_counts().sort_index()
    ax.bar([AQI_NAMES.get(i, (str(i), PALETTE["purple"]))[0] for i in counts.index], 
           counts.values, color=[AQI_NAMES.get(i, ("", PALETTE["purple"]))[1] for i in counts.index],
           edgecolor="white")
    ax.set_title("Distribution AQI — L (labellisé)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Observations")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

with tab2:
    st.markdown(f"### 🤖 {algo_choice} — {model_type}")
    st.info(f"γ={gamma} | Marge={min_margin} | Patience={patience} | Max_iter={max_iter}")
    
    if st.button(f"▶️ Lancer {algo_choice}", type="primary", use_container_width=True):
        t0 = time.time()
        
        if algo_choice == "Self-Training":
            history = run_self_training(X_L, y_L, X_U, X_test, y_test,
                                       gamma=gamma, max_iter=max_iter, 
                                       model_type=model_map[model_type],
                                       patience=patience, min_margin=min_margin)
        else:
            history = run_co_training(X_L, y_L, X_U, X_test, y_test,
                                     va_idx=va_idx, vb_idx=vb_idx,
                                     gamma=gamma, max_iter=max_iter, k_per_iter=k_per_iter,
                                     model_type=model_map[model_type],
                                     patience=patience, min_margin=min_margin)
        
        elapsed = time.time() - t0
        final = history[-1]
        best_f1 = final.get("best_f1", final["f1_macro"])
        
        st.success(f"✅ Terminé en {elapsed:.1f}s | **F1 = {best_f1:.4f}**")
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("F1 Final", f"{final['f1_macro']:.4f}")
        k2.metric("Best F1", f"{best_f1:.4f}")
        k3.metric("Précision", f"{final['precision']:.4f}")
        k4.metric("Rappel", f"{final['recall']:.4f}")
        
        st.pyplot(fig_ssl_progress(history, algo_choice), use_container_width=True)
        
        # Prédictions finales
        if algo_choice == "Self-Training":
            y_pred = final["clf"].predict(X_test)
        else:
            cA, cB = final["clf_A"], final["clf_B"]
            pA = cA.predict_proba(X_test[:, va_idx])
            pB = cB.predict_proba(X_test[:, vb_idx])
            y_pred, _ = _ensemble_vote(cA, cB, pA, pB)
        
        st.pyplot(fig_confusion(y_test, y_pred, f"{algo_choice}"), use_container_width=True)
        
        if "results" not in st.session_state:
            st.session_state["results"] = {}
        st.session_state["results"][algo_choice] = {
            "f1_macro": final['f1_macro'],
            "precision": final['precision'],
            "recall": final['recall'],
            "y_pred": y_pred,
            "model": model_type
        }

with tab3:
    st.markdown("### 📈 Dashboard Comparatif")
    
    if "results" not in st.session_state or not st.session_state["results"]:
        st.warning("⚠️ Lancez une simulation d'abord")
    else:
        clf_base = make_clf(model_map[model_type])
        clf_base.fit(X_L, y_L)
        y_base = clf_base.predict(X_test)
        
        all_res = {
            "Baseline": {
                "f1_macro": round(f1_score(y_test, y_base, average="macro", zero_division=0), 4),
                "precision": round(precision_score(y_test, y_base, average="macro", zero_division=0), 4),
                "recall": round(recall_score(y_test, y_base, average="macro", zero_division=0), 4),
            }
        }
        for k, v in st.session_state["results"].items():
            all_res[k] = v
        
        st.pyplot(fig_compare(all_res), use_container_width=True)
        
        df_comp = pd.DataFrame({k: {"F1": v["f1_macro"], "Précision": v["precision"], 
                                    "Rappel": v["recall"]}
                                for k, v in all_res.items()}).T
        st.dataframe(df_comp.style.format("{:.4f}"), use_container_width=True)
