#!/usr/bin/env python3
"""
11_optimizar_modelo.py
Optimización de hiperparámetros para Gradient Boosting y Random Forest
mediante GridSearchCV y RandomizedSearchCV con validación cruzada 5-fold
Input:  ml_matrix_binary.csv.gz (matriz combinada RGI+ResFinder)
Output: mejor_modelo_optimizado.joblib + resultados_optimizacion.tsv
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib

from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, GridSearchCV, cross_validate
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ──────────────────────────────────────────────────────────────────
ML_DIR  = Path('/mnt/f/TFM_Linux/ml_matrices')
OUT_DIR = Path('/mnt/f/TFM_Linux/ml_matrices/resultados')
LOG_DIR = Path('/mnt/f/MIS_DATOS_TFM/logs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'optimizacion.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

N_JOBS = 6
SEED   = 42
CV     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def load_data():
    log.info('Cargando matriz combinada...')
    df = pd.read_csv(ML_DIR / 'ml_matrix_binary.csv.gz', index_col=0)
    X = df.drop(columns=['target'])
    y = df['target']
    log.info(f'  Shape: {X.shape} · Balance: {y.value_counts().to_dict()}')
    return X, y

def optimizar_gbt(X, y):
    """Optimización de Gradient Boosting con RandomizedSearchCV."""
    log.info('\n=== Optimizando Gradient Boosting ===')

    param_dist = {
        'n_estimators':      [100, 200, 300, 500],
        'max_depth':         [2, 3, 4, 5],
        'learning_rate':     [0.01, 0.05, 0.1, 0.2],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
        'subsample':         [0.7, 0.8, 0.9, 1.0],
        'max_features':      ['sqrt', 'log2', None],
    }

    base = GradientBoostingClassifier(random_state=SEED)
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=50,
        scoring='roc_auc',
        cv=CV,
        n_jobs=N_JOBS,
        random_state=SEED,
        verbose=1
    )
    search.fit(X, y)

    log.info(f'  Mejor AUC CV: {search.best_score_:.4f}')
    log.info(f'  Mejores parámetros: {search.best_params_}')
    return search

def optimizar_rf(X, y):
    """Optimización de Random Forest con RandomizedSearchCV."""
    log.info('\n=== Optimizando Random Forest ===')

    param_dist = {
        'n_estimators':      [100, 200, 300, 500],
        'max_depth':         [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
        'max_features':      ['sqrt', 'log2', 0.3],
        'class_weight':      ['balanced', 'balanced_subsample', None],
    }

    base = RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS)
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=50,
        scoring='roc_auc',
        cv=CV,
        n_jobs=1,
        random_state=SEED,
        verbose=1
    )
    search.fit(X, y)

    log.info(f'  Mejor AUC CV: {search.best_score_:.4f}')
    log.info(f'  Mejores parámetros: {search.best_params_}')
    return search

def evaluar_final(X, y, modelo, nombre):
    """Evaluación completa del modelo optimizado."""
    scoring = {
        'roc_auc':   'roc_auc',
        'f1':        'f1',
        'precision': 'precision',
        'recall':    'recall',
        'accuracy':  'accuracy',
    }
    scores = cross_validate(modelo, X, y, cv=CV, scoring=scoring, n_jobs=N_JOBS)
    log.info(f'\n  Evaluación final {nombre}:')
    for m in scoring:
        v = scores[f'test_{m}']
        log.info(f'    {m:12s}: {v.mean():.4f} ± {v.std():.4f}')
    return scores

def feature_importance(modelo, X, nombre):
    """Análisis de importancia de features."""
    if hasattr(modelo, 'feature_importances_'):
        imp = pd.Series(modelo.feature_importances_, index=X.columns)
        imp = imp.sort_values(ascending=False)
        out = OUT_DIR / f'feature_importance_{nombre}.tsv'
        imp.to_csv(out, sep='\t', header=['importance'])
        log.info(f'\n  Top 20 features más importantes ({nombre}):')
        log.info(imp.head(20).to_string())

def main():
    log.info('=== OPTIMIZACIÓN DE HIPERPARÁMETROS ===')
    X, y = load_data()

    resultados = []

    # Optimizar GBT
    search_gbt = optimizar_gbt(X, y)
    scores_gbt = evaluar_final(X, y, search_gbt.best_estimator_, 'GBT_optimizado')
    feature_importance(search_gbt.best_estimator_, X, 'GBT')
    resultados.append({
        'modelo': 'GBT_optimizado',
        'auc_mean': scores_gbt['test_roc_auc'].mean(),
        'auc_std':  scores_gbt['test_roc_auc'].std(),
        'f1_mean':  scores_gbt['test_f1'].mean(),
        'params':   str(search_gbt.best_params_)
    })

    # Optimizar RF
    search_rf = optimizar_rf(X, y)
    scores_rf = evaluar_final(X, y, search_rf.best_estimator_, 'RF_optimizado')
    feature_importance(search_rf.best_estimator_, X, 'RF')
    resultados.append({
        'modelo': 'RF_optimizado',
        'auc_mean': scores_rf['test_roc_auc'].mean(),
        'auc_std':  scores_rf['test_roc_auc'].std(),
        'f1_mean':  scores_rf['test_f1'].mean(),
        'params':   str(search_rf.best_params_)
    })

    # Guardar resultados
    df_res = pd.DataFrame(resultados)
    df_res.to_csv(OUT_DIR / 'resultados_optimizacion.tsv', sep='\t', index=False)

    # Guardar mejor modelo global
    mejor_auc_gbt = scores_gbt['test_roc_auc'].mean()
    mejor_auc_rf  = scores_rf['test_roc_auc'].mean()

    if mejor_auc_gbt >= mejor_auc_rf:
        mejor = search_gbt.best_estimator_
        nombre_mejor = 'GBT'
        mejor_auc = mejor_auc_gbt
    else:
        mejor = search_rf.best_estimator_
        nombre_mejor = 'RF'
        mejor_auc = mejor_auc_rf

    mejor.fit(X, y)
    joblib.dump(mejor, OUT_DIR / 'mejor_modelo_optimizado.joblib')

    print('\n' + '='*60)
    print('COMPARATIVA BASELINE vs OPTIMIZADO')
    print('='*60)
    print(f'GBT  baseline:   AUC = 0.831')
    print(f'GBT  optimizado: AUC = {mejor_auc_gbt:.4f}  ({mejor_auc_gbt-0.831:+.4f})')
    print(f'RF   baseline:   AUC = 0.807')
    print(f'RF   optimizado: AUC = {mejor_auc_rf:.4f}  ({mejor_auc_rf-0.807:+.4f})')
    print(f'\nMejor modelo final: {nombre_mejor} · AUC = {mejor_auc:.4f}')
    print('='*60)

if __name__ == '__main__':
    main()
