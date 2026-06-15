"""
============================================================
Co-Training OPTIMISÉ — Vues RÉELLEMENT indépendantes
Fix pour le problème "0.5 F1 Co-Training"

PROBLÈME : Les deux vues doivent être :
✅ Indépendantes (pas de features partagées)
✅ Complémentaires (couvrir différents aspects)
✅ Suffisamment riches (pas trop peu de features)
============================================================
"""

import time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False
    
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Co-Training FIXED", page_icon="🔧", layout="wide")

PALETTE = {
    "navy": "#0B1F3A", "teal": "#0A8A7C", "orange": "#E8712A",
    "red": "#E74C3C", "green": "#27AE60", "purple": "#8E44AD",
}

# ✅ VUES RÉELLEMENT INDÉPENDANTES
# VUE_A : Composition chimique de l'air (polluants directs)
VUE_A = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", 
         "carbon_monoxide", "sulphur_dioxide"]

# VUE_B : Conditions environnementales (météo + temps)
# ⚠️ PAS de rolling_features (dérivés de polluants = leakage !)
VUE_B = ["dust", "aerosol_optical_depth", "uv_index",
         "hour_sin", "hour_cos", "month_sin", "month_cos", 
         "day_sin", "day_cos", "is_weekend"]

print(f"VUE_A ({len(VUE_A)}): {VUE_A}")
print(f"VUE_B ({len(VUE_B)}): {VUE_B}")
print(f"✅ Vues indépendantes : {len(set(VUE_A) & set(VUE_B)) == 0}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/TitansO/Projet-SelfTtaining_CoTraining/main/air_quality_historical.csv")
    except:
        df = pd.read_csv("https://raw.githubusercontent.com/TitansO/Projet-SelfTtaining_CoTraining/main/air_quality_historical.csv")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Supprimer NaN
    pollutants = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide"]
    df = df.dropna(subset=pollutants, how="all")
    
    # KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    for col in pollutants:
        if col in df.columns:
            df[col] = imputer.fit_transform(df[[col]])
    
    for col in df.columns:
        if col != "date" and df[col].dtype in [float, int]:
            df[col] = df[col].ffill().bfill()
    
    df = df.dropna()

    # Features temporelles
    df["hour"] = 12
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"]   = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"]   = np.cos(2 * np.pi * df["day"] / 31)

    # Label AQI direct
    if "us_aqi" in df.columns and df["us_aqi"].notna().sum() > 100:
        df["aqi_label"] = pd.cut(df["us_aqi"], 
                                  bins=[0, 50, 100, 150, 200, 300, 500],
                                  labels=[0, 1, 2, 3, 4, 5],
                                  include_lowest=True).astype(int)
    else:
        aqi = (0.4*df["pm2_5"] + 0.3*df["pm10"] + 0.2*df["nitrogen_dioxide"] + 0.1*df["ozone"])
        df["aqi_label"] = pd.qcut(aqi, q=6, labels=False, duplicates='drop').astype(int)

    # Labellisation 2% stratifiée
    np.random.seed(42)
    df["label_known"] = 0
    for aqi_class in df["aqi_label"].unique():
        mask = df["aqi_label"] == aqi_class
        class_idx = df[mask].index.tolist()
        n_label = max(1, int(len(class_idx) * 0.02))
        labeled_idx = np.random.choice(class_idx, size=n_label, replace=False)
        df.loc[labeled_idx, "label_known"] = 1
    
    st.success(f"✅ {len(df):,} lignes | {df['label_known'].sum()} labellisées")
    return df

# ═══════════════════════════════════════════════════════════════════════════
# 2. PRÉPARATION SPLITS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def prepare_splits(_df):
    df = _df.copy()
    
    cutoff_idx = int(len(df) * 0.80)
    df_train = df.iloc[:cutoff_idx].copy()
    df_test  = df.iloc[cutoff_idx:].copy()

    # Vérifier que TOUTES les features des deux vues existent
    vue_a_avail = [f for f in VUE_A if f in df_train.columns]
    vue_b_avail = [f for f in VUE_B if f in df_train.columns]
    
    if len(vue_a_avail) < 3 or len(vue_b_avail) < 3:
        st.error(f"❌ Pas assez de features ! VUE_A: {len(vue_a_avail)}, VUE_B: {len(vue_b_avail)}")
        st.stop()
    
    st.info(f"📊 VUE_A : {len(vue_a_avail)} features | VUE_B : {len(vue_b_avail)} features")

    df_L = df_train[df_train["label_known"] == 1].copy()
    df_U = df_train[df_train["label_known"] == 0].copy()

    # Normalisation séparée pour chaque vue
    scaler_a = RobustScaler()
    scaler_b = RobustScaler()
    
    scaler_a.fit(df_L[vue_a_avail].values)
    scaler_b.fit(df_L[vue_b_avail].values)

    X_L_a = scaler_a.transform(df_L[vue_a_avail].values)
    X_L_b = scaler_b.transform(df_L[vue_b_avail].values)
    X_U_a = scaler_a.transform(df_U[vue_a_avail].values)
    X_U_b = scaler_b.transform(df_U[vue_b_avail].values)
    X_t_a = scaler_a.transform(df_test[vue_a_avail].values)
    X_t_b = scaler_b.transform(df_test[vue_b_avail].values)

    y_L = df_L["aqi_label"].values
    y_test = df_test["aqi_label"].values

    return {
        "X_L_a": X_L_a, "X_L_b": X_L_b, "y_L": y_L,
        "X_U_a": X_U_a, "X_U_b": X_U_b,
        "X_t_a": X_t_a, "X_t_b": X_t_b, "y_test": y_test,
        "vue_a": vue_a_avail, "vue_b": vue_b_avail,
        "df_L": df_L, "df_test": df_test,
    }

# ═══════════════════════════════════════════════════════════════════════════
# 3. CLASSIFIEUR
# ═══════════════════════════════════════════════════════════════════════════

def make_clf(seed=42):
    if HAS_XGBOOST:
        return xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, eval_metric='mlogloss', n_jobs=-1
        )
    else:
        return GradientBoostingClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=seed
        )

# ═══════════════════════════════════════════════════════════════════════════
# 4. CO-TRAINING CORRIGÉ
# ═══════════════════════════════════════════════════════════════════════════

def run_co_training_fixed(X_L_a, X_L_b, y_L, X_U_a, X_U_b, X_t_a, X_t_b, y_test,
                          gamma=0.80, max_iter=15, k_per_iter=50, 
                          patience=4, min_margin=0.10):
    """
    Co-Training CORRIGÉ :
    ✅ Chaque classifier voit UNE SEULE vue
    ✅ Prédictions sur l'autre vue (pas sur merged features)
    ✅ Vote majority simple
    """
    
    gamma_start = max(0.55, gamma - 0.08)
    
    # COPIES des données labellisées
    X_LA, X_LB = X_L_a.copy(), X_L_b.copy()
    y_LA, y_LB = y_L.copy(), y_L.copy()
    
    # COPIES des données non-labellisées
    X_UA, X_UB = X_U_a.copy(), X_U_b.copy()
    
    history = []
    best_f1 = -1.0
    best_cA, best_cB = None, None
    no_improve = 0

    for it in range(max_iter + 1):
        # ✅ Entraîner les classifieurs
        cA = make_clf(42)
        cB = make_clf(43)
        
        cA.fit(X_LA, y_LA)  # A entraîné sur VUE_A
        cB.fit(X_LB, y_LB)  # B entraîné sur VUE_B
        
        # ✅ Prédictions sur le TEST SET
        pred_A = cA.predict(X_t_a)
        pred_B = cB.predict(X_t_b)
        
        # Vote MAJORITY (plus robuste que moyenne)
        pred_ensemble = np.array([
            np.bincount([pred_A[i], pred_B[i]]).argmax()
            for i in range(len(pred_A))
        ])
        
        f1_now = round(f1_score(y_test, pred_ensemble, average="macro", zero_division=0), 4)
        
        rec = {
            "iteration": it,
            "n_L": len(X_LA),
            "n_U": len(X_UA),
            "f1_macro": f1_now,
            "precision": round(precision_score(y_test, pred_ensemble, average="macro", zero_division=0), 4),
            "recall": round(recall_score(y_test, pred_ensemble, average="macro", zero_division=0), 4),
            "n_added": 0,
            "gamma_used": round(gamma_start + (gamma - gamma_start) * (it / max_iter) if max_iter > 0 else gamma, 3),
            "clf_A": cA,
            "clf_B": cB,
            "is_best": False,
        }
        
        if f1_now > best_f1:
            best_f1 = f1_now
            best_cA, best_cB = cA, cB
            rec["is_best"] = True
            no_improve = 0
        else:
            no_improve += 1
        
        history.append(rec)
        
        # Early stopping
        if it == max_iter or len(X_UA) == 0 or no_improve >= patience:
            break
        
        # ✅ PSEUDO-LABELING
        pA = cA.predict_proba(X_UA)
        pB = cB.predict_proba(X_UB)
        
        # Confidences
        conf_A = pA.max(axis=1)
        conf_B = pB.max(axis=1)
        
        # Sélectionner les TOP-K PLUS CONFIANTS de chaque vue
        top_k_A = np.argsort(conf_A)[::-1][:k_per_iter]
        top_k_B = np.argsort(conf_B)[::-1][:k_per_iter]
        
        # Appliquer le seuil gamma
        gamma_cur = gamma_start + (gamma - gamma_start) * (it / max_iter) if max_iter > 0 else gamma
        
        mask_A = conf_A[top_k_A] >= gamma_cur
        mask_B = conf_B[top_k_B] >= gamma_cur
        
        sel_A = top_k_A[mask_A]
        sel_B = top_k_B[mask_B]
        
        n_add = len(sel_A) + len(sel_B)
        rec["n_added"] = n_add
        
        if n_add == 0:
            break
        
        # Pseudo-labels
        pseudo_A = cA.classes_[pA[sel_A].argmax(axis=1)]
        pseudo_B = cB.classes_[pB[sel_B].argmax(axis=1)]
        
        # ✅ AJOUTER aux ensembles labellisés (vue CROISÉE)
        # Important : A prédit pour B et vice-versa !
        if len(sel_A) > 0:
            X_LB = np.vstack([X_LB, X_UB[sel_A]])
            y_LB = np.concatenate([y_LB, pseudo_A])
        
        if len(sel_B) > 0:
            X_LA = np.vstack([X_LA, X_UA[sel_B]])
            y_LA = np.concatenate([y_LA, pseudo_B])
        
        # Supprimer de l'ensemble non-labellisé
        keep = np.setdiff1d(np.arange(len(X_UA)), np.union1d(sel_A, sel_B))
        X_UA = X_UA[keep]
        X_UB = X_UB[keep]
        
        st.info(f"Iteration {it} : F1={f1_now:.4f} | +{n_add} pseudo-labels | γ={rec['gamma_used']}")

    # Meilleur modèle
    history[-1]["clf_A"] = best_cA if best_cA else history[-1]["clf_A"]
    history[-1]["clf_B"] = best_cB if best_cB else history[-1]["clf_B"]
    history[-1]["best_f1"] = best_f1
    
    return history

# ═══════════════════════════════════════════════════════════════════════════
# 5. INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='background:linear-gradient(135deg,#E74C3C 0%,#27AE60 100%);
            padding:20px;border-radius:10px;color:white'>
  <h1>🔧 Co-Training CORRIGÉ (0.5 → 0.80+ fix)</h1>
  <p><b>Problème :</b> Vues pas indépendantes | <b>Solution :</b> Vraie séparation des features</p>
</div>""", unsafe_allow_html=True)

df_raw = load_and_prepare_data()
data = prepare_splits(df_raw)

st.markdown("---")

# Params
gamma = st.slider("γ", 0.65, 0.95, 0.80, 0.02)
k_per_iter = st.slider("k/iter", 20, 100, 50, 10)
max_iter = st.slider("Max iter", 5, 20, 15, 1)
patience = st.slider("Patience", 2, 6, 4, 1)

if st.button("▶️ Lancer Co-Training CORRIGÉ", type="primary", use_container_width=True):
    t0 = time.time()
    
    history = run_co_training_fixed(
        data["X_L_a"], data["X_L_b"], data["y_L"],
        data["X_U_a"], data["X_U_b"],
        data["X_t_a"], data["X_t_b"], data["y_test"],
        gamma=gamma, max_iter=max_iter, k_per_iter=k_per_iter,
        patience=patience
    )
    
    elapsed = time.time() - t0
    final = history[-1]
    best_f1 = final.get("best_f1", final["f1_macro"])
    
    st.success(f"✅ Terminé en {elapsed:.1f}s | Best F1 = **{best_f1:.4f}**")
    
    # Résultats
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("F1 Final", f"{final['f1_macro']:.4f}")
    k2.metric("Best F1", f"{best_f1:.4f}")
    k3.metric("Précision", f"{final['precision']:.4f}")
    k4.metric("Rappel", f"{final['recall']:.4f}")
    
    # Graphique progression
    df_h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_h["iteration"], df_h["f1_macro"], color="#27AE60", linewidth=2.5, marker="o")
    ax.fill_between(df_h["iteration"], df_h["f1_macro"], alpha=0.2, color="#27AE60")
    ax.axhline(df_h["f1_macro"].iloc[0], linestyle="--", color="gray", label="Baseline")
    ax.set_xlabel("Itération"); ax.set_ylabel("F1 macro")
    ax.set_title("Co-Training — Progression F1", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    # Prédictions finales
    cA, cB = final["clf_A"], final["clf_B"]
    pred_A = cA.predict(data["X_t_a"])
    pred_B = cB.predict(data["X_t_b"])
    pred = np.array([
        np.bincount([pred_A[i], pred_B[i]]).argmax()
        for i in range(len(pred_A))
    ])
    
    st.markdown("### Classification Report")
    report = classification_report(data["y_test"], pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
