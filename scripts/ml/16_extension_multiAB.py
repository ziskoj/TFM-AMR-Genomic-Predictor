#!/usr/bin/env python3
"""
16_extension_multiAB.py
Extensión del estudio de carbapenémicos a fluoroquinolonas y cefalosporinas 3G.
Sigue EXACTAMENTE la misma metodología que los scripts 07-15 del TFM.

Input:  ml_matrix_binary.csv.gz (la misma del estudio original, 188 features)
Output: (en --output_dir)
        amr_fenotipos_<ab>.tsv         → datos BV-BRC crudos
        amr_fenotipos_<ab>_clean.tsv   → deduplicados R/S
        resultados_<ab>.tsv            → métricas CV 5-fold
        metricas_clinicas_<ab>.tsv     → TP/TN/FP/FN/VPP/VPN
        feature_importance_<ab>_GBT.tsv

Uso:
  python 16_extension_multiAB.py \
      --matrix   /mnt/f/TFM_Linux/ml_matrices/ml_matrix_binary.csv.gz \
      --output_dir /mnt/f/TFM_Linux/ml_matrices/resultados_multiAB \
      --antibiotics fluoroquinolones cephalosporins3g

Requisitos: mismos que el entorno rgi_env (pandas, sklearn, xgboost, lightgbm)
"""

import os, argparse, logging, time, warnings
import numpy as np
import pandas as pd
import requests
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate as sk_cv
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                              precision_score, recall_score, accuracy_score)
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier; HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier; HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ── Configuración BV-BRC ───────────────────────────────────────────────────
API_URL  = "https://www.bv-brc.org/api/genome_amr/"
CAMPOS   = ["genome_id","genome_name","antibiotic",
            "resistant_phenotype","evidence",
            "laboratory_typing_method","testing_standard"]

ANTIBIOTICS = {
    "fluoroquinolones": {
        "display":  "Fluoroquinolonas",
        "terms":    ["ciprofloxacin", "levofloxacin", "norfloxacin"],
        "primary":  "ciprofloxacin",
    },
    "cephalosporins3g": {
        "display": "Cefalosporinas 3G",
        "terms":   ["ceftriaxone", "cefotaxime", "ceftazidime"],
        "primary": "ceftriaxone",
    },
}

# ── Hiperparámetros optimizados (idénticos a scripts 11 y 13) ─────────────
SEED = 42
CV   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def get_models():
    models = {
        "GBT": GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            min_samples_split=5, min_samples_leaf=2,
            subsample=0.8, max_features=None, random_state=SEED),
        "RF": RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=2,
            min_samples_leaf=1, max_features='sqrt',
            class_weight='balanced', n_jobs=-1, random_state=SEED),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=1, gamma=0.1,
            reg_alpha=0.01, reg_lambda=1.5,
            eval_metric='auc', verbosity=0,
            random_state=SEED, n_jobs=-1)
    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=-1, learning_rate=0.05,
            num_leaves=31, min_child_samples=10,
            subsample=1.0, colsample_bytree=1.0,
            reg_alpha=0, reg_lambda=0.1,
            class_weight=None,
            random_state=SEED, n_jobs=-1, verbose=-1)
    return models

# ── Descarga BV-BRC (mismo patrón que 02_descarga_amr_fenotipos.py) ───────
def download_phenotypes(ab_terms, out_raw, out_clean):
    dfs = []
    for term in ab_terms:
        log.info(f"  Descargando BV-BRC: {term}...")
        rql = (f"eq(genome_name,Klebsiella%20pneumoniae*)&"
               f"eq(evidence,Laboratory%20Method)&"
               f"eq(antibiotic,{term})&"
               f"in(resistant_phenotype,(Resistant,Susceptible))&"
               f"select({','.join(CAMPOS)})&limit(50000)")
        try:
            r = requests.get(API_URL, params=rql,
                             headers={"accept": "text/tsv"}, timeout=120)
            r.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(r.text), sep='\t')
            df['genome_id'] = df['genome_id'].astype(str).str.replace('"','').str.strip()
            log.info(f"    → {len(df):,} registros  "
                     f"{df['resistant_phenotype'].value_counts().to_dict()}")
            dfs.append(df)
        except Exception as e:
            log.error(f"    Error: {e}")
        time.sleep(1)

    if not dfs:
        log.error("Sin datos descargados. Verifica conexión a internet.")
        return None

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(out_raw, sep='\t', index=False)

    # Deduplicar: mismo criterio que script 02 (Resistant > Susceptible)
    df_all['resistant_phenotype'] = pd.Categorical(
        df_all['resistant_phenotype'],
        categories=['Susceptible', 'Resistant'], ordered=True)
    df_clean = (df_all.sort_values('resistant_phenotype', ascending=False)
                      .drop_duplicates(subset='genome_id', keep='first')
                      .reset_index(drop=True))
    df_clean.to_csv(out_clean, sep='\t', index=False)

    log.info(f"  Total genomas únicos: {len(df_clean):,}")
    log.info(f"  Distribución: {df_clean['resistant_phenotype'].value_counts().to_dict()}")
    return df_clean

# ── Evaluación completa (mismo esquema que 07 + 15) ───────────────────────
def evaluar_modelo(model, X, y, nombre):
    scoring = {'roc_auc':'roc_auc','f1':'f1',
               'precision':'precision','recall':'recall','accuracy':'accuracy'}
    scores = sk_cv(model, X, y, cv=CV, scoring=scoring,
                   n_jobs=-1, return_train_score=False)
    res = {'Modelo': nombre}
    for m in scoring:
        v = scores[f'test_{m}']
        res[f'{m}_mean'] = round(v.mean(), 4)
        res[f'{m}_std']  = round(v.std(),  4)
    log.info(f"    AUC={res['roc_auc_mean']:.4f}±{res['roc_auc_std']:.4f}  "
             f"F1={res['f1_mean']:.4f}  "
             f"Recall={res['recall_mean']:.4f}  "
             f"Precision={res['precision_mean']:.4f}")
    return res

def metricas_clinicas(model, X, y, nombre):
    """Entrena sobre todos los datos y calcula métricas clínicas completas."""
    model.fit(X, y)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.decision_function(X)
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    auc  = roc_auc_score(y, y_prob)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    vpp  = tp / (tp + fp) if (tp + fp) > 0 else 0
    vpn  = tn / (tn + fn) if (tn + fn) > 0 else 0
    lr_pos = sens / (1 - spec) if spec < 1 else float('inf')
    lr_neg = (1 - sens) / spec if spec > 0 else float('inf')

    return {'modelo': nombre, 'AUC': round(auc,4),
            'Sensibilidad': round(sens,4), 'Especificidad': round(spec,4),
            'VPP': round(vpp,4), 'VPN': round(vpn,4),
            'LR+': round(lr_pos,2), 'LR-': round(lr_neg,3),
            'F1': round(f1_score(y,y_pred),4),
            'Accuracy': round(accuracy_score(y,y_pred),4),
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)}

def feature_importance_gbt(model, feature_names, out_path):
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        imp = imp.sort_values(ascending=False)
        imp.to_csv(out_path, sep='\t', header=['importance'])
        log.info(f"  Top-5: {imp.head(5).to_dict()}")

# ── Pipeline principal ─────────────────────────────────────────────────────
def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"Cargando feature matrix: {args.matrix}")
    df_mat = pd.read_csv(args.matrix, index_col=0)
    feature_names = [c for c in df_mat.columns if c != 'target']
    # No usamos la columna target de la matriz — cada antibiótico tiene la suya
    X_full = df_mat[feature_names]
    X_full.index = X_full.index.astype(str).str.strip()
    log.info(f"  {X_full.shape[0]:,} genomas × {X_full.shape[1]} features")

    all_summary = []

    for ab_key in args.antibiotics:
        if ab_key not in ANTIBIOTICS:
            log.error(f"Antibiótico desconocido: {ab_key}")
            continue

        cfg = ANTIBIOTICS[ab_key]
        log.info(f"\n{'='*60}")
        log.info(f"ANTIBIÓTICO: {cfg['display']} ({cfg['primary']})")
        log.info(f"{'='*60}")

        # 1. Descargar fenotipos
        raw_path   = out / f"amr_fenotipos_{ab_key}.tsv"
        clean_path = out / f"amr_fenotipos_{ab_key}_clean.tsv"

        if clean_path.exists() and not args.force:
            log.info(f"  Usando fenotipos ya descargados: {clean_path}")
            df_pheno = pd.read_csv(clean_path, sep='\t')
        else:
            log.info("  Paso 1: Descargando fenotipos BV-BRC...")
            df_pheno = download_phenotypes(cfg['terms'], raw_path, clean_path)
            if df_pheno is None:
                continue

        # 2. Cruzar con feature matrix
        log.info("  Paso 2: Cruzando con feature matrix...")
        df_pheno['genome_id'] = df_pheno['genome_id'].astype(str).str.strip()
        df_pheno['target'] = (df_pheno['resistant_phenotype'] == 'Resistant').astype(int)
        df_pheno = df_pheno.drop_duplicates('genome_id').set_index('genome_id')

        merged = X_full.join(df_pheno[['target']], how='inner')
        if len(merged) < 50:
            log.error(f"  Solo {len(merged)} genomas en común. Revisa genome_id. Omitiendo.")
            continue

        X = merged[feature_names]
        y = merged['target']
        log.info(f"  Dataset final: {len(X):,} genomas | "
                 f"R={y.sum():,} ({y.mean()*100:.1f}%) | "
                 f"S={(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")

        # 3. Evaluar modelos con CV 5-fold
        log.info("  Paso 3: Validación cruzada 5-fold (mismos hiperparámetros que estudio original)...")
        models = get_models()
        resultados_cv = []
        metricas_clin = []

        for nombre, model in models.items():
            log.info(f"  → {nombre}...")
            res = evaluar_modelo(model, X, y, nombre)
            resultados_cv.append(res)
            mc  = metricas_clinicas(model, X.copy(), y.copy(), nombre)
            metricas_clin.append(mc)

            if nombre == 'GBT':
                feature_importance_gbt(
                    model, feature_names,
                    out / f"feature_importance_{ab_key}_GBT.tsv")
                joblib.dump(model, out / f"modelo_{ab_key}_GBT.joblib")

        # 4. Guardar resultados
        df_cv = pd.DataFrame(resultados_cv)
        df_cv.to_csv(out / f"resultados_{ab_key}.tsv", sep='\t', index=False)

        df_mc = pd.DataFrame(metricas_clin)
        df_mc.to_csv(out / f"metricas_clinicas_{ab_key}.tsv", sep='\t', index=False)

        log.info(f"\n  Resultados guardados en {out}/")

        # Resumen pantalla
        print(f"\n{'─'*65}")
        print(f"RESUMEN: {cfg['display']}")
        print(f"{'─'*65}")
        print(df_cv[['Modelo','roc_auc_mean','roc_auc_std','f1_mean',
                     'recall_mean','precision_mean']].to_string(index=False))

        for r in resultados_cv:
            all_summary.append({
                'Antibiótico': cfg['display'],
                'Modelo': r['Modelo'],
                'N_genomas': len(X),
                'Pct_R': round(y.mean()*100, 1),
                'AUC': r['roc_auc_mean'], 'AUC_SD': r['roc_auc_std'],
                'F1': r['f1_mean'],
                'Sensibilidad': r['recall_mean'],
                'Especificidad': next(
                    m['Especificidad'] for m in metricas_clin
                    if m['modelo'] == r['Modelo']),
            })

    # 5. Tabla comparativa global (incluye referencia carbapenémicos)
    carba_ref = [
        {'Antibiótico':'Carbapenémicos','Modelo':'GBT',      'N_genomas':4044,'Pct_R':39.5,'AUC':0.8461,'AUC_SD':0.0092,'F1':0.708,'Sensibilidad':0.629,'Especificidad':0.903},
        {'Antibiótico':'Carbapenémicos','Modelo':'XGBoost',   'N_genomas':4044,'Pct_R':39.5,'AUC':0.8449,'AUC_SD':0.0087,'F1':0.720,'Sensibilidad':0.704,'Especificidad':0.835},
        {'Antibiótico':'Carbapenémicos','Modelo':'LightGBM',  'N_genomas':4044,'Pct_R':39.5,'AUC':0.8437,'AUC_SD':0.0117,'F1':0.711,'Sensibilidad':0.639,'Especificidad':0.896},
        {'Antibiótico':'Carbapenémicos','Modelo':'RF',        'N_genomas':4044,'Pct_R':39.5,'AUC':0.8386,'AUC_SD':0.0114,'F1':0.683,'Sensibilidad':0.586,'Especificidad':0.914},
        {'Antibiótico':'Carbapenémicos','Modelo':'MLP',       'N_genomas':4044,'Pct_R':39.5,'AUC':0.8186,'AUC_SD':0.0152,'F1':0.687,'Sensibilidad':0.680,'Especificidad':0.780},
    ]
    df_global = pd.DataFrame(carba_ref + all_summary)
    df_global.to_csv(out / "comparison_all_antibiotics.csv", index=False)

    print(f"\n{'='*65}")
    print("TABLA COMPARATIVA GLOBAL (todos antibióticos)")
    print(f"{'='*65}")
    print(df_global.to_string(index=False))
    print(f"\n✅ Todo guardado en: {out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extensión multi-AB siguiendo metodología TFM")
    parser.add_argument('--matrix', required=True,
                        help='Ruta a ml_matrix_binary.csv.gz')
    parser.add_argument('--output_dir', required=True,
                        help='Carpeta de salida')
    parser.add_argument('--antibiotics', nargs='+',
                        default=['fluoroquinolones','cephalosporins3g'],
                        choices=list(ANTIBIOTICS.keys()))
    parser.add_argument('--force', action='store_true',
                        help='Forzar re-descarga aunque ya existan los TSV')
    args = parser.parse_args()
    run(args)
