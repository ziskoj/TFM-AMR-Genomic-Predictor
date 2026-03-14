#!/usr/bin/env python3
"""
15_metricas_clinicas.py
Métricas clínicas completas para todos los modelos
Sensibilidad, Especificidad, VPP, VPN, LR+, LR-
Curvas ROC superpuestas + Matrices de confusión
Input:  ml_matrix_binary.csv.gz + modelos guardados
Output: metricas_clinicas.tsv + curvas ROC + matrices confusión
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Rutas ──────────────────────────────────────────────────────────────────
ML_DIR  = Path('/mnt/f/TFM_Linux/ml_matrices')
OUT_DIR = Path('/mnt/f/TFM_Linux/ml_matrices/resultados')
EDA_DIR = Path('/mnt/f/TFM_Linux/ml_matrices/eda')
LOG_DIR = Path('/mnt/f/MIS_DATOS_TFM/logs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'metricas_clinicas.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

SEED     = 42
CV       = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
COLORES  = {
    'GBT':       '#E74C3C',
    'XGBoost':   '#E67E22',
    'LightGBM':  '#F1C40F',
    'RF':        '#2ECC71',
    'MLP':       '#9B59B6',
    'LR':        '#3498DB',
}

def load_data():
    log.info('Cargando matriz completa...')
    df = pd.read_csv(ML_DIR / 'ml_matrix_binary.csv.gz', index_col=0)
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def metricas_clinicas(y_true, y_pred, y_prob, nombre):
    """Calcula métricas clínicas completas."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sens  = tp / (tp + fn)           # Sensibilidad (Recall)
    spec  = tn / (tn + fp)           # Especificidad
    vpp   = tp / (tp + fp)           # Valor Predictivo Positivo
    vpn   = tn / (tn + fn)           # Valor Predictivo Negativo
    lr_pos = sens / (1 - spec)       # Likelihood Ratio positivo
    lr_neg = (1 - sens) / spec       # Likelihood Ratio negativo
    auc   = roc_auc_score(y_true, y_prob)
    f1    = f1_score(y_true, y_pred)
    acc   = accuracy_score(y_true, y_pred)

    return {
        'modelo':        nombre,
        'AUC':           round(auc, 4),
        'Sensibilidad':  round(sens, 4),
        'Especificidad': round(spec, 4),
        'VPP':           round(vpp, 4),
        'VPN':           round(vpn, 4),
        'LR+':           round(lr_pos, 2),
        'LR-':           round(lr_neg, 3),
        'F1':            round(f1, 4),
        'Accuracy':      round(acc, 4),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
    }

def evaluar_modelos(X, y):
    """Evalúa todos los modelos con cross_val_predict para métricas clínicas."""
    scale_pos = (y == 0).sum() / (y == 1).sum()

    modelos = {
        'GBT': joblib.load(OUT_DIR / 'mejor_modelo_optimizado.joblib'),
        'XGBoost': joblib.load(OUT_DIR / 'modelo_xgboost.joblib'),
        'LightGBM': joblib.load(OUT_DIR / 'modelo_lightgbm.joblib'),
        'RF': GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            random_state=SEED
        ),
    }

    resultados = []
    roc_data   = {}

    for nombre, modelo in modelos.items():
        log.info(f'  Evaluando {nombre}...')
        try:
            y_prob = cross_val_predict(
                modelo, X, y, cv=CV,
                method='predict_proba', n_jobs=6
            )[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            metrics = metricas_clinicas(y, y_pred, y_prob, nombre)
            resultados.append(metrics)
            roc_data[nombre] = (y, y_prob)

            log.info(f'    AUC={metrics["AUC"]:.4f} · Sens={metrics["Sensibilidad"]:.4f} · '
                     f'Spec={metrics["Especificidad"]:.4f} · VPP={metrics["VPP"]:.4f} · VPN={metrics["VPN"]:.4f}')
        except Exception as e:
            log.error(f'    Error en {nombre}: {e}')

    return pd.DataFrame(resultados), roc_data

def plot_roc_curves(roc_data):
    """Curvas ROC superpuestas para todos los modelos."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for nombre, (y_true, y_prob) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        color = COLORES.get(nombre, '#333333')
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{nombre} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    ax.set_xlabel('1 - Especificidad (Tasa de Falsos Positivos)', fontsize=12)
    ax.set_ylabel('Sensibilidad (Tasa de Verdaderos Positivos)', fontsize=12)
    ax.set_title('Curvas ROC — Predicción de Resistencia a Carbapenémicos\nKlebsiella pneumoniae · Validación Cruzada 5-fold',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = EDA_DIR / '06_curvas_roc.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'  Guardado: {out}')

def plot_confusion_matrices(roc_data):
    """Matrices de confusión para todos los modelos."""
    n = len(roc_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    for ax, (nombre, (y_true, y_prob)) in zip(axes, roc_data.items()):
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'{nombre}\nAUC={roc_auc_score(y_true, y_prob):.3f}',
                     fontsize=11, fontweight='bold')

        classes = ['Susceptible', 'Resistant']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=9)

        thresh = 0.5
        for i in range(2):
            for j in range(2):
                ax.text(j, i,
                        f'{cm[i,j]:,}\n({cm_norm[i,j]:.1%})',
                        ha='center', va='center', fontsize=10,
                        color='white' if cm_norm[i,j] > thresh else 'black')

        ax.set_ylabel('Real', fontsize=10)
        ax.set_xlabel('Predicho', fontsize=10)

    plt.suptitle('Matrices de Confusión — Validación Cruzada 5-fold',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = EDA_DIR / '07_matrices_confusion.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'  Guardado: {out}')

def print_tabla_clinica(df_metrics):
    """Imprime tabla clínica formateada."""
    print('\n' + '='*90)
    print('MÉTRICAS CLÍNICAS COMPLETAS — VALIDACIÓN CRUZADA 5-FOLD')
    print('='*90)
    cols = ['modelo', 'AUC', 'Sensibilidad', 'Especificidad', 'VPP', 'VPN', 'LR+', 'LR-', 'F1']
    print(df_metrics[cols].to_string(index=False))
    print('='*90)
    print('\nInterpretación clínica:')
    best = df_metrics.loc[df_metrics['AUC'].idxmax()]
    print(f"  Mejor modelo ({best['modelo']}): AUC={best['AUC']}")
    print(f"  - Sensibilidad {best['Sensibilidad']:.1%}: de 100 resistentes, detecta ~{best['Sensibilidad']*100:.0f}")
    print(f"  - Especificidad {best['Especificidad']:.1%}: de 100 susceptibles, identifica correctamente ~{best['Especificidad']*100:.0f}")
    print(f"  - VPP {best['VPP']:.1%}: si predice resistente, acierta el {best['VPP']*100:.0f}% de las veces")
    print(f"  - VPN {best['VPN']:.1%}: si predice susceptible, acierta el {best['VPN']*100:.0f}% de las veces")
    print(f"  - LR+ {best['LR+']:.1f}: un resultado positivo multiplica por {best['LR+']:.1f} la probabilidad de resistencia")

def main():
    log.info('=== MÉTRICAS CLÍNICAS ===')
    X, y = load_data()

    log.info('\nEvaluando modelos...')
    df_metrics, roc_data = evaluar_modelos(X, y)

    # Guardar métricas
    out = OUT_DIR / 'metricas_clinicas.tsv'
    df_metrics.to_csv(out, sep='\t', index=False)
    log.info(f'\nMétricas guardadas: {out}')

    # Gráficas
    log.info('\nGenerando gráficas...')
    plot_roc_curves(roc_data)
    plot_confusion_matrices(roc_data)

    # Tabla clínica
    print_tabla_clinica(df_metrics)

if __name__ == '__main__':
    main()
