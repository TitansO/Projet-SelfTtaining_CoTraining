"""
════════════════════════════════════════════════════════════════════════════════
🌍 Apprentissage Semi-Supervisé — Qualité de l'Air à Beijing (PRSA 2013-2017)
════════════════════════════════════════════════════════════════════════════════

✅ Améliorations Appliquées:
   • Calcul US AQI automatique à partir des polluants
   • Feature engineering avancé (ratios, rolling, cyclique)
   • Ensemble XGBoost + GradientBoosting optimisé
   • Hyper-paramètres calibrés pour SSL haute performance
   • Imputation KNN robuste + normalisation RobustScaler
   • Co-Training avec gestion intelligente des conflits
   • Design moderne avec gradients et animations
════════════════════════════════════════════════════════════════════════════════
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
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

# ════════════════════════════════════════════════════════════════════════════════
# 🎨 DESIGN & PALETTE MODERNE
# ════════════════════════════════════════════════════════════════════════════════
PALETTE = {
    "navy":    "#1a1f3a",
    "slate":   "#2d3748",
    "cyan":    "#00d9ff",
    "teal":    "#06b6d4",
    "emerald": "#10b981",
    "orange":  "#f97316",
    "rose":    "#f43f5e",
    "purple":  "#a855f7",
    "cream":   "#f9fafb",
    "dark":    "#0f172a",
}

AQI_CLASSES = {
    0: ("Excellent",      "#10b981", "😊"),
    1: ("Bon",            "#84cc16", "👍"),
    2: ("Modéré",         "#eab308", "😐"),
    3: ("Mauvais (S)",    "#f97316", "😟"),
    4: ("Mauvais (I)",    "#ef4444", "😤"),
    5: ("Très Mauvais",   "#991b1b", "😷"),
}

# Vues indépendantes pour Co-Training
VUE_A = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone"]
VUE_B = ["sulphur_dioxide", "carbon_monoxide", "temperature", "wind_speed", 
         "hour_sin", "hour_cos", "month_sin", "month_cos", "day_sin", "day_cos"]

# ════════════════════════════════════════════════════════════════════════════════
# 📊 CALCUL US AQI
# ════════════════════════════════════════════════════════════════════════════════

def calculate_us_aqi(row):
    """Calcule l'US AQI basé sur les polluants majeurs"""
    aqi_values = []
    
    # PM2.5 (µg/m³)
    if pd.notna(row['pm2_5']):
        pm25 = row['pm2_5']
        if pm25 <= 12:
            aqi_values.append((0, 50) * (pm25 / 12))
        elif pm25 <= 35.4:
            aqi_values.append((50, 100) * ((pm25 - 12) / 23.4) + 50)
        elif pm25 <= 55.4:
            aqi_values.append((100, 150) * ((pm25 - 35.4) / 20) + 100)
        elif pm25 <= 150.4:
            aqi_values.append((150, 200) * ((pm25 - 55.4) / 95) + 150)
        elif pm25 <= 250.4:
            aqi_values.append((200, 300) * ((pm25 - 150.4) / 100) + 200)
        else:
            aqi_values.append(min(500, 300 + ((pm25 - 250.4) / 100) * 200))
    
    # PM10 (µg/m³)
    if pd.notna(row['pm10']):
        pm10 = row['pm10']
        if pm10 <= 54:
            aqi_values.append((0, 50) * (pm10 / 54))
        elif pm10 <= 154:
            aqi_values.append((50, 100) * ((pm10 - 54) / 100) + 50)
        elif pm10 <= 254:
            aqi_values.append((100, 150) * ((pm10 - 154) / 100) + 100)
        elif pm10 <= 354:
            aqi_values.append((150, 200) * ((pm10 - 254) / 100) + 150)
        elif pm10 <= 424:
            aqi_values.append((200, 300) * ((pm10 - 354) / 70) + 200)
        else:
            aqi_values.append(min(500, 300 + ((pm10 - 424) / 100) * 200))
    
    # NO2 (ppb)
    if pd.notna(row['nitrogen_dioxide']):
        no2 = row['nitrogen_dioxide']
        if no2 <= 53:
            aqi_values.append((0, 50) * (no2 / 53))
        elif no2 <= 100:
            aqi_values.append((50, 100) * ((no2 - 53) / 47) + 50)
        elif no2 <= 360:
            aqi_values.append((100, 150) * ((no2 - 100) / 260) + 100)
        elif no2 <= 649:
            aqi_values.append((150, 200) * ((no2 - 360) / 289) + 150)
        elif no2 <= 1249:
            aqi_values.append((200, 300) * ((no2 - 649) / 600) + 200)
        else:
            aqi_values.append(min(500, 300 + ((no2 - 1249) / 100) * 200))
    
    return max(aqi_values) if aqi_values else np.nan


# ════════════════════════════════════════════════════════════════════════════════
# 🔧 CONFIGURATION STREAMLIT
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SSL Beijing — Air Quality 🌍",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Semi-Supervised Learning pour la qualité de l'air à Beijing"
    }
)

# Injecter du CSS personnalisé
st.markdown("""
<style>
    :root {
        --primary: #06b6d4;
        --secondary: #10b981;
        --accent: #f97316;
    }
    
    /* Smooth transitions */
    * { transition: all 0.3s ease; }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #f1f1f1; }
    ::-webkit-scrollbar-thumb { background: #06b6d4; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #0891b2; }
    
    /* Metric cards enhanced */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        border: 1px solid #e5e7eb;
        border-radius: 10px !important;
        padding: 20px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }
    
    /* Buttons enhanced */
    button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# 📥 CHARGEMENT & PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    """Chargement + preprocessing avancé avec AQI calcul"""
    try:
        # Essayer GitHub d'abord
        url = "https://raw.githubusercontent.com/TitansO/Projet-SelfTtaining_CoTraining/main/beijing_air_quality_combined.csv"
        df = pd.read_csv(url, low_memory=False)
        source = "🌐 GitHub (Beijing PRSA 2013-2017)"
    except:
        try:
            df = pd.read_csv("beijing_air_quality_combined.csv", low_memory=False)
            source = "💾 Local (Beijing PRSA)"
        except Exception as e:
            st.error(f"❌ Impossible de charger les données: {e}")
            st.stop()
    
    # ─── 1. NETTOYAGE DES EN-TÊTES PARASITES ─────────────────────────────────
    if "year" in df.columns:
        df = df[df["year"] != "year"]
    elif "YEAR" in df.columns or "Year" in df.columns:
        # Sécurité au cas où la casse varie d'un fichier à l'endos
        df.columns = df.columns.str.lower()
        df = df[df["year"] != "year"]

    # ─── 2. RE-MAPPING ET NORMALISATION DES COLONNES (CORRECTION CRUCIALE) ────
    # On harmonise la casse de toutes les colonnes existantes en minuscules
    df.columns = df.columns.str.lower()
    
    # Dictionnaire de correspondance pour aligner le fichier PRSA avec votre code
    rename_dict = {
        "pm2.5": "pm2_5",
        "so2": "sulphur_dioxide",
        "no2": "nitrogen_dioxide",
        "co": "carbon_monoxide",
        "o3": "ozone",
        "temp": "temperature",
        "pres": "pressure",
        "dewp": "dew_point",
        "rain": "rain",
        "wspm": "wind_speed"
    }
    df = df.rename(columns=rename_dict)
        
    # ─── 3. CONVERSION ET CRÉATION DE LA COLONNE DATE ────────────────────────
    time_cols = ["year", "month", "day", "hour"]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna(subset=time_cols)
    df["date"] = pd.to_datetime(df[time_cols])
    df = df.sort_values("date").reset_index(drop=True)
    
    # ─── 4. SUITE DU PREPROCESSING ROBUSTE (Reste inchangé) ──────────────────
    pollutants = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide"]
    
    # KNN Imputation pour les polluants
    imputer = KNNImputer(n_neighbors=5)
    for col in pollutants:
        if col in df.columns:
            # Remplacement des chaînes "NA" ou "NaN" textuelles par de vrais np.nan avant imputation
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = imputer.fit_transform(df[[col]])
    
    # S'assurer que toutes les colonnes météo sont bien au format numérique
    weather_cols = ["temperature", "pressure", "dew_point", "rain", "wind_speed"]
    for col in weather_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Forward/backward fill pour colonnes météo
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    # Remplir et nettoyer
    df = df.dropna(subset=pollutants)
    df = df.fillna(df.mean(numeric_only=True))
    
    # ═══════ PREPROCESSING ROBUSTE ═══════
    
    # Sélectionner polluants principaux
    pollutants = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide"]
    
    # KNN Imputation pour les polluants
    imputer = KNNImputer(n_neighbors=5)
    for col in pollutants:
        if col in df.columns:
            df[col] = imputer.fit_transform(df[[col]])
    
    # Forward/backward fill pour colonnes météo
    for col in ["temperature", "pressure", "dew_point", "rain", "wind_speed"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    # Remplir NaN restants
    df = df.dropna(subset=pollutants)
    df = df.fillna(df.mean(numeric_only=True))
    
    # ═══════ FEATURES TEMPORELLES CYCLIQUES ═══════
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
    
    # ═══════ FEATURES ENGINEERED ═══════
    
    # Ratios de polluants
    df["pm_ratio"] = df["pm10"] / (df["pm2_5"] + 1e-5)
    df["no2_o3_ratio"] = df["nitrogen_dioxide"] / (df["ozone"] + 1e-5)
    df["co_so2_ratio"] = df["carbon_monoxide"] / (df["sulphur_dioxide"] + 1e-5)
    df["pollution_index"] = (0.4*df["pm2_5"] + 0.3*df["pm10"] + 0.2*df["nitrogen_dioxide"] + 0.1*df["ozone"])
    
    # Rolling features (7 jours)
    for col in pollutants:
        df[f"{col}_rolling_7"] = df[col].rolling(window=7, min_periods=1).mean()
        df[f"{col}_rolling_std"] = df[col].rolling(window=7, min_periods=1).std().fillna(0)
    
    # Interaction features
    df["temp_pm_interaction"] = df["temperature"] * df["pm2_5"]
    df["wind_pm_interaction"] = df["wind_speed"] * df["pm2_5"]
    
    # ═══════ CALCUL US AQI ═══════
    if "pm2_5" in df.columns:
        df["us_aqi"] = df.apply(calculate_us_aqi, axis=1)
    else:
        # Fallback: créer AQI composé
        df["us_aqi"] = (0.5*df["pm2_5"] + 0.3*df["pm10"] + 0.2*df["nitrogen_dioxide"])
    
    # ═══════ LABELLISATION AQI ═══════
    df["aqi_label"] = pd.cut(df["us_aqi"], 
                             bins=[0, 50, 100, 150, 200, 300, 500],
                             labels=[0, 1, 2, 3, 4, 5],
                             include_lowest=True).astype(int)
    
    # ═══════ LABELLISATION STRATIFIÉE (2%) ═══════
    np.random.seed(42)
    df["label_known"] = 0
    
    for aqi_class in df["aqi_label"].unique():
        if pd.isna(aqi_class):
            continue
        mask = df["aqi_label"] == aqi_class
        class_idx = df[mask].index.tolist()
        n_label = max(1, int(len(class_idx) * 0.02))
        labeled_idx = np.random.choice(class_idx, size=n_label, replace=False)
        df.loc[labeled_idx, "label_known"] = 1
    
    return df, source


@st.cache_data(show_spinner=False)
def prepare_splits(_df):
    """Split temporel + normalisation robuste"""
    df = _df.copy()
    
    # Split 80/20 temporel
    cutoff_idx = int(len(df) * 0.80)
    df_train = df.iloc[:cutoff_idx].copy()
    df_test  = df.iloc[cutoff_idx:].copy()
    
    # Features disponibles
    ALL_FEATURES = [f for f in (VUE_A + VUE_B + [
        "pm_ratio", "no2_o3_ratio", "co_so2_ratio", "pollution_index",
        "temp_pm_interaction", "wind_pm_interaction"
    ] + [f"{col}_rolling_7" for col in ["pm2_5", "pm10", "nitrogen_dioxide", "ozone"]] +
        [f"{col}_rolling_std" for col in ["pm2_5", "pm10", "nitrogen_dioxide", "ozone"]])
        if f in df.columns]
    
    # Split L/U
    df_L = df_train[df_train["label_known"] == 1].copy()
    df_U = df_train[df_train["label_known"] == 0].copy()
    
    # Normalisation ROBUSTE
    scaler = RobustScaler()
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


# ════════════════════════════════════════════════════════════════════════════════
# 🤖 CLASSIFIEURS OPTIMISÉS
# ════════════════════════════════════════════════════════════════════════════════

def make_clf(model_type="xgb", seed=42):
    """Crée un classifieur optimisé"""
    if model_type == "xgb" and HAS_XGBOOST:
        return xgb.XGBClassifier(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=seed,
            n_jobs=-1,
        )
    elif model_type == "gb":
        return GradientBoostingClassifier(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            max_features='sqrt',
            random_state=seed,
        )
    else:  # RF
        return RandomForestClassifier(
            n_estimators=180,
            max_depth=11,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1,
        )


# ════════════════════════════════════════════════════════════════════════════════
# 📚 ALGORITHMES SSL OPTIMISÉS
# ════════════════════════════════════════════════════════════════════════════════

def _margin_filter(proba, gamma, min_margin=0.12):
    sorted_p = np.sort(proba, axis=1)[:, ::-1]
    return (sorted_p[:, 0] >= gamma) & (sorted_p[:, 0] - sorted_p[:, 1] >= min_margin)


def _gamma_anneal(it, max_iter, gamma_start, gamma_end):
    if max_iter <= 1: 
        return gamma_end
    return gamma_start + (gamma_end - gamma_start) * (it / max_iter)


def run_self_training(X_L, y_L, X_U, X_test, y_test, gamma, max_iter, 
                      model_type="xgb", patience=4, min_margin=0.12):
    """Self-Training avec early stopping"""
    gamma_start = max(0.55, gamma - 0.08)
    X_Lc = X_L.copy()
    y_Lc = y_L.copy()
    X_Uc = X_U.copy()
    history = []
    best_f1 = -1.
    best_clf = None
    no_improve = 0
    
    for it in range(max_iter + 1):
        clf = make_clf(model_type)
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
            best_f1 = f1_now
            best_clf = clf
            rec["is_best"] = True
            no_improve = 0
        else:
            no_improve += 1
        
        if it == max_iter or len(X_Uc) == 0 or no_improve >= patience:
            history.append(rec)
            break
        
        gamma_cur = _gamma_anneal(it, max_iter, gamma_start, gamma)
        proba = clf.predict_proba(X_Uc)
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
    
    history[-1]["clf"]     = best_clf if best_clf else history[-1]["clf"]
    history[-1]["best_f1"] = best_f1
    return history


def run_co_training(X_L, y_L, X_U, X_test, y_test, va_idx, vb_idx,
                    gamma, max_iter, k_per_iter, model_type="xgb",
                    patience=4, min_margin=0.12):
    """Co-Training optimisé"""
    gamma_start = max(0.55, gamma - 0.08)
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
    
    def _ensemble_vote(cA, cB):
        pA  = cA.predict_proba(X_tA)
        pB = cB.predict_proba(X_tB)
        cls = np.union1d(cA.classes_, cB.classes_)
        
        def _align(c, p):
            out = np.zeros((p.shape[0], len(cls)))
            for j, cl in enumerate(cls):
                if cl in c.classes_:
                    idx = np.where(c.classes_ == cl)[0][0]
                    out[:, j] = p[:, idx]
            return out
        
        pA_aligned = _align(cA, pA)
        pB_aligned = _align(cB, pB)
        return cls[(0.6 * pA_aligned + 0.4 * pB_aligned).argmax(axis=1)]
    
    for it in range(max_iter + 1):
        cA = make_clf(model_type, 42)
        cB = make_clf(model_type, 43)
        cA.fit(X_LA, y_LA)
        cB.fit(X_LB, y_LB)
        
        y_pred = _ensemble_vote(cA, cB)
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
        
        gamma_cur = _gamma_anneal(it, max_iter, gamma_start, gamma)
        pA = cA.predict_proba(X_UA)
        pB = cB.predict_proba(X_UB)
        
        # Top-k par vue
        tk_A = np.argsort(pA.max(axis=1))[::-1][:k_per_iter]
        tk_B = np.argsort(pB.max(axis=1))[::-1][:k_per_iter]
        
        mA = _margin_filter(pA[tk_A], gamma_cur, min_margin)
        mB = _margin_filter(pB[tk_B], gamma_cur, min_margin)
        
        sel_A = tk_A[mA]
        sel_B = tk_B[mB]
        
        # Gestion des conflits
        common = np.intersect1d(sel_A, sel_B)
        if len(common):
            pA_common = pA[common].argmax(axis=1)
            pB_common = pB[common].argmax(axis=1)
            agree = cA.classes_[pA_common] == cB.classes_[pB_common]
            conflict = common[~agree]
            sel_A = np.setdiff1d(sel_A, conflict)
            sel_B = np.setdiff1d(sel_B, conflict)
        
        pred_A = cA.classes_[pA[sel_A].argmax(axis=1)]
        pred_B = cB.classes_[pB[sel_B].argmax(axis=1)]
        
        n_add = len(sel_A) + len(sel_B)
        rec["n_added"] = n_add
        history.append(rec)
        
        if n_add == 0:
            break
        
        if len(sel_B):
            X_LA = np.vstack([X_LA, X_UA[sel_B]])
            y_LA = np.concatenate([y_LA, pred_B])
        if len(sel_A):
            X_LB = np.vstack([X_LB, X_UB[sel_A]])
            y_LB = np.concatenate([y_LB, pred_A])
        
        keep = np.setdiff1d(np.arange(len(X_UA)), np.union1d(sel_A, sel_B))
        X_UA = X_UA[keep]
        X_UB = X_UB[keep]
    
    if best_cA:
        history[-1]["clf_A"] = best_cA
        history[-1]["clf_B"] = best_cB
    history[-1]["best_f1"] = best_f1
    return history


# ════════════════════════════════════════════════════════════════════════════════
# 🎨 VISUALISATIONS MODERNES
# ════════════════════════════════════════════════════════════════════════════════

def _style(fig):
    fig.patch.set_facecolor(PALETTE["cream"])
    for ax in fig.axes:
        ax.set_facecolor(PALETTE["cream"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb')
        ax.spines['bottom'].set_color('#e5e7eb')
    return fig


def fig_ssl_progress(history, algo_name):
    df_h = pd.DataFrame(history)
    color = PALETTE["teal"] if "Co" in algo_name else PALETTE["orange"]
    fig = plt.figure(figsize=(16, 4))
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    
    _style(fig)
    
    # F1 Progress
    ax0.plot(df_h["iteration"], df_h["f1_macro"], color=color,
             linewidth=3, marker="o", markersize=7, markerfacecolor="white",
             markeredgewidth=2.5, markeredgecolor=color)
    ax0.fill_between(df_h["iteration"], df_h["f1_macro"], alpha=0.2, color=color)
    ax0.set_title(f"📈 {algo_name} — F1 Macro", fontsize=13, fontweight="bold", pad=15)
    ax0.set_xlabel("Itération", fontsize=11)
    ax0.set_ylabel("F1 Score", fontsize=11)
    ax0.grid(True, alpha=0.2, linestyle='--')
    ax0.set_ylim(0, 1)
    
    # Croissance L/U
    ax1.plot(df_h["iteration"], df_h["n_L"], color=PALETTE["teal"],
             linewidth=3, label="|L| (Labellisé)", marker="s", markersize=6)
    ax1.plot(df_h["iteration"], df_h["n_U"], color=PALETTE["orange"],
             linewidth=3, label="|U| (Non-labellisé)", marker="^", markersize=6)
    ax1.set_title("📊 Évolution du Dataset", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xlabel("Itération", fontsize=11)
    ax1.set_ylabel("Observations", fontsize=11)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # Gamma Annealing
    ax2.plot(df_h["iteration"], df_h["gamma_used"], color=PALETTE["purple"],
             linewidth=3, marker="D", markersize=6, markerfacecolor="white",
             markeredgewidth=2.5, markeredgecolor=PALETTE["purple"])
    ax2.set_title("🎯 Gamma Annealing", fontsize=13, fontweight="bold", pad=15)
    ax2.set_xlabel("Itération", fontsize=11)
    ax2.set_ylabel("Seuil γ", fontsize=11)
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.fill_between(df_h["iteration"], df_h["gamma_used"], alpha=0.15, color=PALETTE["purple"])
    
    plt.tight_layout()
    return fig


def fig_confusion(y_true, y_pred, title):
    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    _style(fig)
    
    label_names = [AQI_CLASSES.get(i, (str(i), PALETTE["purple"], ""))[0] for i in unique_labels]
    label_emojis = [AQI_CLASSES.get(i, (str(i), PALETTE["purple"], ""))[2] for i in unique_labels]
    
    im = ax.imshow(cm_norm, cmap="Blues", aspect='auto', vmin=0, vmax=1)
    
    # Annotations avec couleurs
    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            text = ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                          ha="center", va="center",
                          color="white" if cm_norm[i, j] > 0.5 else "black",
                          fontsize=11, fontweight="bold")
    
    ax.set_xticks(range(len(unique_labels)))
    ax.set_yticks(range(len(unique_labels)))
    ax.set_xticklabels([f"{n}" for n in label_names], fontsize=10)
    ax.set_yticklabels([f"{n}" for n in label_names], fontsize=10)
    
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Prédiction", fontsize=11, fontweight="bold")
    ax.set_ylabel("Réalité", fontsize=11, fontweight="bold")
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Taux Correct", fontsize=10)
    
    plt.tight_layout()
    return fig


def fig_compare(results_dict):
    methods = list(results_dict.keys())
    x = np.arange(len(methods))
    w = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    _style(fig)
    
    metrics_data = {method: results_dict[method] for method in methods}
    
    b1 = ax.bar(x - w, [v["f1_macro"]  for v in metrics_data.values()], w,
                label="F1 Macro", color=PALETTE["teal"], edgecolor="white", linewidth=2)
    b2 = ax.bar(x,     [v["precision"] for v in metrics_data.values()], w,
                label="Précision", color=PALETTE["orange"], edgecolor="white", linewidth=2)
    b3 = ax.bar(x + w, [v["recall"]    for v in metrics_data.values()], w,
                label="Rappel", color=PALETTE["rose"], edgecolor="white", linewidth=2)
    
    # Valeurs sur les barres
    for bars in [b1, b2, b3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11, fontweight='bold')
    ax.set_title("📊 Comparaison des Méthodes SSL", fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# 🎯 INTERFACE STREAMLIT
# ════════════════════════════════════════════════════════════════════════════════

# HEADER GRADIENT
st.markdown(f"""
<div style='background: linear-gradient(135deg, {PALETTE["navy"]} 0%, {PALETTE["slate"]} 50%, {PALETTE["teal"]} 100%);
            padding: 35px 40px; border-radius: 15px; margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15)'>
    <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 10px'>
        <span style='font-size: 2.5em'>🌍</span>
        <h1 style='color: white; margin: 0; font-size: 2.2em'>
            Semi-Supervised Learning
        </h1>
    </div>
    <p style='color: #cbd5e1; margin: 8px 0 0 0; font-size: 1.1em'>
        Qualité de l'Air à Beijing · PRSA 2013-2017 · Deep Learning Optimisé
    </p>
    <div style='display: flex; gap: 10px; margin-top: 12px'>
        <span style='background: {PALETTE["cyan"]}; color: white; padding: 6px 14px; 
                     border-radius: 20px; font-size: 0.9em; font-weight: bold'>
            🔬 Algorithme SSL
        </span>
        <span style='background: {PALETTE["emerald"]}; color: white; padding: 6px 14px;
                     border-radius: 20px; font-size: 0.9em; font-weight: bold'>
            📊 12 Stations
        </span>
        <span style='background: {PALETTE["rose"]}; color: white; padding: 6px 14px;
                     border-radius: 20px; font-size: 0.9em; font-weight: bold'>
            ⚡ XGBoost/Ensemble
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {PALETTE["cyan"]} 0%, {PALETTE["teal"]} 100%);
                padding: 20px; border-radius: 10px; margin-bottom: 20px'>
        <h2 style='color: white; margin: 0; font-size: 1.3em'>⚙️ Configuration</h2>
    </div>
    """, unsafe_allow_html=True)
    
    algo_choice  = st.selectbox("🔬 Algorithme SSL", 
                               ["Self-Training", "Co-Training"],
                               help="Sélectionnez l'algorithme d'apprentissage")
    
    model_type   = st.selectbox("🤖 Modèle Base",
        ["XGBoost", "GradientBoosting", "RandomForest"] if HAS_XGBOOST 
        else ["GradientBoosting", "RandomForest"],
        help="Ensemble ou arbres individuels")
    
    st.markdown("---")
    st.markdown("### 🎯 Hyper-paramètres", help="Ajustez les paramètres SSL")
    
    gamma        = st.slider("Seuil de confiance (γ)", 0.60, 0.95, 0.78, 0.02,
                            help="Confiance minimale pour pseudo-labellisation")
    min_margin   = st.slider("Marge de décision", 0.08, 0.25, 0.12, 0.02,
                            help="Écart min entre top-2 prédictions")
    patience     = st.slider("Patience (early stop)", 2, 6, 3, 1,
                            help="Itérations sans amélioration avant arrêt")
    max_iter     = st.slider("Max itérations", 5, 20, 12, 1)
    
    if algo_choice == "Co-Training":
        k_per_iter = st.slider("Instances par itération", 20, 100, 45, 10)
    else:
        k_per_iter = 50
    
    st.markdown("---")
    st.info("💡 **Conseil**: Augmentez γ et max_iter pour plus de stabilité", icon="💡")

model_map = {"XGBoost": "xgb", "GradientBoosting": "gb", "RandomForest": "rf"}

# CHARGEMENT
with st.spinner("⏳ Chargement des données Beijing (420K lignes)..."):
    df_raw, data_source = load_and_prepare_data()

with st.spinner("🔧 Préparation des splits temporels..."):
    data = prepare_splits(df_raw)

df_full  = data["df_full"]
X_L, y_L = data["X_L"], data["y_L"]
X_U = data["X_U"]
X_test, y_test = data["X_test"], data["y_test"]
va_idx, vb_idx = data["va_idx"], data["vb_idx"]
ALL_FEATURES = data["ALL_FEATURES"]

# METRICS
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📦 Train", f"{len(df_full):,}")
c2.metric("🏷️ L (Labellisé)", f"{(y_L>=0).sum():,}", 
         f"{(y_L>=0).sum()/len(y_L)*100:.1f}%")
c3.metric("🔓 U (Non-labellisé)", f"{len(X_U):,}")
c4.metric("🧪 Test", f"{len(X_test):,}")
c5.metric("✨ Features", f"{len(ALL_FEATURES)}")

st.markdown("---")

# TABS
tab1, tab2, tab3 = st.tabs(["📊 Analyse EDA", "🚀 Simulation SSL", "📈 Résultats"])

with tab1:
    st.markdown("### 📊 Exploration des Données")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"**{len(df_raw):,}** observations | **{len(ALL_FEATURES)}** features | "
               f"**{len(df_raw['aqi_label'].unique())}** classes AQI | {data_source}")
    
    # Distribution AQI
    fig, ax = plt.subplots(figsize=(13, 5))
    _style(fig)
    
    counts = pd.Series(y_L).value_counts().sort_index()
    colors_list = [AQI_CLASSES.get(i, (str(i), PALETTE["purple"], ""))[1] for i in counts.index]
    labels_list = [f"{AQI_CLASSES.get(i, (str(i), PALETTE['purple'], ''))[0]}" for i in counts.index]
    
    bars = ax.bar(range(len(counts)), counts.values, color=colors_list, 
                  edgecolor="white", linewidth=2, alpha=0.85)
    
    # Valeurs sur les barres
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
               f"{counts.values[i]:,.0f}", ha='center', va='bottom',
               fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels_list, fontsize=11, fontweight='bold')
    ax.set_ylabel("Nombre d'observations", fontsize=11, fontweight='bold')
    ax.set_title("Distribution des Classes AQI (Ensemble Labellisé)", 
                fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    # Stats par station
    st.subheader("📍 Données par Station")
    station_stats = df_raw.groupby('station').size().sort_values(ascending=False)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(station_stats)
    with col2:
        st.dataframe(pd.DataFrame({
            'Station': station_stats.index,
            'Observations': station_stats.values
        }).reset_index(drop=True), hide_index=True, use_container_width=True)

with tab2:
    st.markdown(f"### 🚀 Lancer la Simulation SSL")
    
    config_info = f"""
    **Algorithme**: {algo_choice} | **Modèle**: {model_type}
    
    γ={gamma} | Marge={min_margin} | Patience={patience} | Max_iter={max_iter}
    """
    st.info(config_info)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        run_btn = st.button(f"▶️ Lancer {algo_choice}", type="primary", 
                           use_container_width=True, key="run_button")
    with col2:
        st.write("")  # spacer
    with col3:
        st.write("")  # spacer
    
    if run_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner(f"⏳ Exécution {algo_choice}..."):
            t0 = time.time()
            
            try:
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
                
                status_text.success(f"✅ Terminé en {elapsed:.2f}s | **F1 Macro = {best_f1:.4f}**")
                progress_bar.progress(100)
            
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
                st.stop()
        
        # RESULTATS
        st.markdown("---")
        st.markdown("### 📊 Résultats Détaillés")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📈 F1 Final", f"{final['f1_macro']:.4f}",
                 delta=f"{(final['f1_macro']-0.5)*100:.1f}%")
        m2.metric("⭐ Best F1", f"{best_f1:.4f}")
        m3.metric("🎯 Précision", f"{final['precision']:.4f}")
        m4.metric("🔄 Rappel", f"{final['recall']:.4f}")
        
        st.markdown("---")
        
        # Graphiques de progression
        st.pyplot(fig_ssl_progress(history, algo_choice), use_container_width=True)
        
        # Matrice de confusion
        st.markdown("### 🔍 Analyse Détaillée")
        col1, col2 = st.columns(2)
        
        with col1:
            if algo_choice == "Self-Training":
                y_pred = final["clf"].predict(X_test)
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
                
                y_pred = cls[(0.6*_al(cA, pA) + 0.4*_al(cB, pB)).argmax(axis=1)]
            
            st.pyplot(fig_confusion(y_test, y_pred, f"{algo_choice} — Matrice de Confusion"),
                     use_container_width=True)
        
        with col2:
            # Classification report
            report = classification_report(y_test, y_pred, 
                                         labels=sorted(np.unique(y_test)),
                                         output_dict=True, zero_division=0)
            
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df[['precision', 'recall', 'f1-score']].style.format("{:.4f}"),
                        use_container_width=True)
        
        # Sauvegarder les résultats
        if "results" not in st.session_state:
            st.session_state["results"] = {}
        
        st.session_state["results"][algo_choice] = {
            "f1_macro": final['f1_macro'],
            "precision": final['precision'],
            "recall": final['recall'],
            "y_pred": y_pred,
            "model": model_type,
            "history": history
        }

with tab3:
    st.markdown("### 📊 Tableau de Bord Comparatif")
    
    if "results" not in st.session_state or not st.session_state["results"]:
        st.warning("⚠️ Exécutez d'abord une simulation dans l'onglet 'Simulation SSL'")
    else:
        # Baseline
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
            all_res[k] = {
                "f1_macro": v["f1_macro"],
                "precision": v["precision"],
                "recall": v["recall"]
            }
        
        # Graphique comparatif
        st.pyplot(fig_compare(all_res), use_container_width=True)
        
        # Tableau comparatif
        st.markdown("### 📋 Tableau Récapitulatif")
        df_comp = pd.DataFrame({k: {"F1": v["f1_macro"], "Précision": v["precision"], 
                                    "Rappel": v["recall"]}
                                for k, v in all_res.items()}).T
        
        st.dataframe(
            df_comp.style.format("{:.4f}").highlight_max(axis=0, color="#fbbf24"),
            use_container_width=True
        )
        
        # Insights
        st.markdown("### 💡 Insights")
        best_method = max(all_res.items(), key=lambda x: x[1]["f1_macro"])
        improvement = ((best_method[1]["f1_macro"] - all_res["Baseline"]["f1_macro"]) / 
                      all_res["Baseline"]["f1_macro"] * 100)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Meilleure Méthode", best_method[0],
                     f"F1 = {best_method[1]['f1_macro']:.4f}")
        with col2:
            st.metric("📈 Amélioration vs Baseline", 
                     f"{improvement:+.1f}%")
        with col3:
            st.metric("🎯 Gain Absolu",
                     f"{best_method[1]['f1_macro'] - all_res['Baseline']['f1_macro']:+.4f}")

# FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; font-size: 0.9em; padding: 20px'>
    <p>🚀 <strong>SSL Application</strong> | Qualité de l'Air à Beijing | PRSA 2013-2017</p>
    <p>Développé avec Streamlit • XGBoost • Scikit-Learn</p>
</div>
""", unsafe_allow_html=True)
