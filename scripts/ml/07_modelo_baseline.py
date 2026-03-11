#!/usr/bin/env python3
"""
07_modelo_baseline.py
Entrena y evalúa modelos baseline para predicción de resistencia AMR
Input:  ml_matrix_binary.csv.gz
Output: resultados de evaluación + modelo guardado
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, matthews_corrcoef,
    make_scorer
)
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ──────────────────────────────────────────────────────────────────
ML_DIR   = Path('/mnt/f/TFM_Linux/ml_matrices')
OUT_DIR  = Path('/mnt/f/TFM_Linux/ml_matrices/resultados')
LOG_DIR  = Path('/mnt/f/MIS_DATOS_TFM/logs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'modelo_baseline.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Parámetros ─────────────────────────────────────────────────────────────
N_SPLITS  = 5   # Validación cruzada 5-fold estratificada
N_JOBS    = 4   # Hilos paralelos
SEED      = 42  # Reproducibilidad

def load_matrix() -> tuple:
    """Carga la matriz ML y separa features y target."""
    log.info('Cargando matriz ML...')
    df = pd.read_csv(ML_DIR / 'ml_matrix_binary.csv.gz', index_col=0)
    X = df.drop(columns=['target'])
    y = df['target']
    log.info(f'  Shape: {X.shape}')
    log.info(f'  Balance: {y.value_counts().to_dict()}')
    return X, y

def evaluar_modelos(X, y) -> pd.DataFrame:
    """Evalúa múltiples modelos con validación cruzada estratificada."""
    modelos = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=None,
            class_weight='balanced', n_jobs=N_JOBS, random_state=SEED
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=SEED
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced',
            n_jobs=N_JOBS, random_state=SEED
        ),
        'SVM': SVC(
            kernel='linear', class_weight='balanced',
            probability=True, random_state=SEED
        ),
    }

    scoring = {
        'accuracy':  'accuracy',
        'f1':        'f1',
        'roc_auc':   'roc_auc',
        'precision': 'precision',
        'recall':    'recall',
    }

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    resultados = []

    for nombre, modelo in modelos.items():
        log.info(f'Evaluando: {nombre}...')
        try:
            scores = cross_validate(
                modelo, X, y,
                cv=cv, scoring=scoring,
                n_jobs=N_JOBS, return_train_score=False
            )
            fila = {'Modelo': nombre}
            for metrica in scoring:
                vals = scores[f'test_{metrica}']
                fila[f'{metrica}_mean'] = vals.mean()
                fila[f'{metrica}_std']  = vals.std()
            resultados.append(fila)
            log.info(f'  AUC: {fila["roc_auc_mean"]:.3f} ± {fila["roc_auc_std"]:.3f}')
            log.info(f'  F1:  {fila["f1_mean"]:.3f} ± {fila["f1_std"]:.3f}')
        except Exception as e:
            log.error(f'  Error en {nombre}: {e}')

    df_res = pd.DataFrame(resultados)
    return df_res

def entrenar_mejor_modelo(X, y, df_resultados: pd.DataFrame):
    """Entrena el mejor modelo sobre todos los datos y lo guarda."""
    mejor = df_resultados.loc[df_resultados['roc_auc_mean'].idxmax(), 'Modelo']
    log.info(f'\nMejor modelo: {mejor}')

    modelos = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, class_weight='balanced',
            n_jobs=N_JOBS, random_state=SEED
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=SEED
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced',
            n_jobs=N_JOBS, random_state=SEED
        ),
        'SVM': SVC(
            kernel='linear', class_weight='balanced',
            probability=True, random_state=SEED
        ),
    }

    modelo_final = modelos[mejor]
    modelo_final.fit(X, y)

    # Guardar modelo
    out_model = OUT_DIR / 'mejor_modelo.joblib'
    joblib.dump(modelo_final, out_model)
    log.info(f'Modelo guardado: {out_model}')

    # Importancia de features (si es Random Forest)
    if mejor == 'Random Forest':
        importancias = pd.Series(
            modelo_final.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        out_imp = OUT_DIR / 'feature_importances.tsv'
        importancias.to_csv(out_imp, sep='\t', header=['importance'])
        log.info(f'\nTop 10 genes más importantes:')
        log.info(importancias.head(10).to_string())

    return modelo_final

def main():
    X, y = load_matrix()

    log.info('\nEvaluando modelos baseline...')
    df_resultados = evaluar_modelos(X, y)

    # Guardar resultados
    out = OUT_DIR / 'resultados_baseline.tsv'
    df_resultados.to_csv(out, sep='\t', index=False)

    print('\n' + '='*65)
    print('RESULTADOS BASELINE — VALIDACIÓN CRUZADA 5-FOLD')
    print('='*65)
    cols_show = ['Modelo', 'roc_auc_mean', 'roc_auc_std', 'f1_mean', 'f1_std', 'accuracy_mean']
    print(df_resultados[cols_show].to_string(index=False))
    print('='*65)

    entrenar_mejor_modelo(X, y, df_resultados)
    log.info('Pipeline baseline completado.')

if __name__ == '__main__':
    main()
