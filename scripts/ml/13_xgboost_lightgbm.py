#!/usr/bin/env python3
"""
13_xgboost_lightgbm.py
Entrenamiento y optimización de XGBoost y LightGBM
Input:  ml_matrix_binary.csv.gz (matriz completa RGI+ResFinder+PointFinder)
Output: resultados_xgb_lgbm.tsv + modelos guardados
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ──────────────────────────────────────────────────────────────────
ML_DIR  = Path('/mnt/f/TFM_Linux/ml_matrices')
OUT_DIR = Path('/mnt/f/TFM_Linux/ml_matrices/resultados')
LOG_DIR = Path('/mnt/f/MIS_DATOS_TFM/logs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'xgb_lgbm.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

N_JOBS = 6
SEED   = 42
CV     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def load_data():
    log.info('Cargando matriz completa...')
    df = pd.read_csv(ML_DIR / 'ml_matrix_binary.csv.gz', index_col=0)
    X = df.drop(columns=['target'])
    y = df['target']
    log.info(f'  Shape: {X.shape} · Balance: {y.value_counts().to_dict()}')
    return X, y

def optimizar_xgboost(X, y):
    log.info('\n=== Optimizando XGBoost ===')
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    param_dist = {
        'n_estimators':     [100, 200, 300, 500],
        'max_depth':        [3, 4, 5, 6],
        'learning_rate':    [0.01, 0.05, 0.1, 0.2],
        'subsample':        [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma':            [0, 0.1, 0.2, 0.5],
        'reg_alpha':        [0, 0.01, 0.1],
        'reg_lambda':       [1, 1.5, 2],
    }

    base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        random_state=SEED,
        n_jobs=N_JOBS,
        verbosity=0
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=60,
        scoring='roc_auc', cv=CV,
        n_jobs=1, random_state=SEED, verbose=1
    )
    search.fit(X, y)
    log.info(f'  Mejor AUC CV: {search.best_score_:.4f}')
    log.info(f'  Mejores parámetros: {search.best_params_}')
    return search

def optimizar_lightgbm(X, y):
    log.info('\n=== Optimizando LightGBM ===')

    param_dist = {
        'n_estimators':     [100, 200, 300, 500],
        'max_depth':        [-1, 5, 7, 10],
        'learning_rate':    [0.01, 0.05, 0.1, 0.2],
        'num_leaves':       [15, 31, 63, 127],
        'min_child_samples':[10, 20, 30, 50],
        'subsample':        [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'reg_alpha':        [0, 0.01, 0.1],
        'reg_lambda':       [0, 0.01, 0.1],
        'class_weight':     ['balanced', None],
    }

    base = LGBMClassifier(
        random_state=SEED,
        n_jobs=N_JOBS,
        verbose=-1
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=60,
        scoring='roc_auc', cv=CV,
        n_jobs=1, random_state=SEED, verbose=1
    )
    search.fit(X, y)
    log.info(f'  Mejor AUC CV: {search.best_score_:.4f}')
    log.info(f'  Mejores parámetros: {search.best_params_}')
    return search

def evaluar_final(X, y, modelo, nombre):
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
    if hasattr(modelo, 'feature_importances_'):
        imp = pd.Series(modelo.feature_importances_, index=X.columns)
        imp = imp.sort_values(ascending=False)
        out = OUT_DIR / f'feature_importance_{nombre}.tsv'
        imp.to_csv(out, sep='\t', header=['importance'])
        log.info(f'\n  Top 15 features ({nombre}):')
        log.info(imp.head(15).to_string())

def main():
    log.info('=== XGBoost + LightGBM ===')
    X, y = load_data()

    resultados = []

    # XGBoost
    search_xgb = optimizar_xgboost(X, y)
    scores_xgb = evaluar_final(X, y, search_xgb.best_estimator_, 'XGBoost')
    feature_importance(search_xgb.best_estimator_, X, 'XGB')
    joblib.dump(search_xgb.best_estimator_, OUT_DIR / 'modelo_xgboost.joblib')
    resultados.append({
        'modelo': 'XGBoost',
        'auc_mean': scores_xgb['test_roc_auc'].mean(),
        'auc_std':  scores_xgb['test_roc_auc'].std(),
        'f1_mean':  scores_xgb['test_f1'].mean(),
        'f1_std':   scores_xgb['test_f1'].std(),
        'params':   str(search_xgb.best_params_)
    })

    # LightGBM
    search_lgbm = optimizar_lightgbm(X, y)
    scores_lgbm = evaluar_final(X, y, search_lgbm.best_estimator_, 'LightGBM')
    feature_importance(search_lgbm.best_estimator_, X, 'LGBM')
    joblib.dump(search_lgbm.best_estimator_, OUT_DIR / 'modelo_lightgbm.joblib')
    resultados.append({
        'modelo': 'LightGBM',
        'auc_mean': scores_lgbm['test_roc_auc'].mean(),
        'auc_std':  scores_lgbm['test_roc_auc'].std(),
        'f1_mean':  scores_lgbm['test_f1'].mean(),
        'f1_std':   scores_lgbm['test_f1'].std(),
        'params':   str(search_lgbm.best_params_)
    })

    # Guardar resultados
    df_res = pd.DataFrame(resultados)
    df_res.to_csv(OUT_DIR / 'resultados_xgb_lgbm.tsv', sep='\t', index=False)

    # Comparativa final
    gbt_auc = 0.846
    print('\n' + '='*65)
    print('COMPARATIVA COMPLETA — TODOS LOS MODELOS')
    print('='*65)
    print(f'GBT optimizado (baseline):  AUC = {gbt_auc:.4f}')
    print(f'XGBoost optimizado:         AUC = {scores_xgb["test_roc_auc"].mean():.4f} ± {scores_xgb["test_roc_auc"].std():.4f}')
    print(f'LightGBM optimizado:        AUC = {scores_lgbm["test_roc_auc"].mean():.4f} ± {scores_lgbm["test_roc_auc"].std():.4f}')
    mejor_auc = max(scores_xgb['test_roc_auc'].mean(), scores_lgbm['test_roc_auc'].mean())
    print(f'\nMejor modelo hasta ahora:   AUC = {mejor_auc:.4f}')
    print('='*65)

if __name__ == '__main__':
    main()
