#!/usr/bin/env python3
"""
18_figuras_modelos_carbapenems.py
─────────────────────────────────────────────────────────────────────────────
Genera las figuras de modelos ML para carbapenémicos de forma REPRODUCIBLE.

Input:
  --matrix       ml_matrix_binary.csv.gz
  --results_dir  resultados_reales/   (TSVs + modelos .joblib)

Output (--output_dir, por defecto ./figuras_tfm/):
  fig_roc_curves.png              – Curvas ROC superpuestas (4 modelos)
  fig_confusion_matrices.png      – Matrices de confusión (4 modelos)
  fig_feature_importance_top20.png – Top-20 genes más importantes (GBT)

Modos de ejecución:
  Modo COMPLETO (por defecto):
    Carga los modelos .joblib, ejecuta cross_val_predict y genera las figuras.
    Reproduce exactamente los valores del TFM.

  Modo RÁPIDO (--from-tsv):
    Lee los TSVs pre-computados (metricas_clinicas.tsv, feature_importance_GBT.tsv).
    Genera matrices de confusión y feature importance sin re-ejecutar modelos.
    Las curvas ROC se muestran como puntos de operación (Sens/Spec), no curvas continuas.

Uso:
  python 18_figuras_modelos_carbapenems.py
  python 18_figuras_modelos_carbapenems.py --from-tsv
  python 18_figuras_modelos_carbapenems.py \\
      --matrix data/ml_matrix_binary.csv.gz \\
      --results_dir resultados/ \\
      --output_dir figuras/
─────────────────────────────────────────────────────────────────────────────
"""
import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)

# Paleta de colores coherente con el resto del TFM
COLORES = {
    'GBT':      '#E74C3C',
    'XGBoost':  '#E67E22',
    'LightGBM': '#F1C40F',
    'RF':       '#2ECC71',
}

plt.rcParams.update({
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ══════════════════════════════════════════════════════════════════════════════
# MODO COMPLETO — carga modelos y re-ejecuta cross_val_predict
# ══════════════════════════════════════════════════════════════════════════════

def load_data(matrix_path: Path):
    log.info(f'Cargando matriz: {matrix_path}')
    df = pd.read_csv(matrix_path)
    df['genome_id'] = df['genome_id'].astype(str).str.strip()
    df = df.set_index('genome_id')
    X = df.drop(columns=['target'])
    y = df['target']
    log.info(f'  Shape: {df.shape}  |  R={y.sum()}  S={(y==0).sum()}')
    return X, y


def run_cross_val(matrix_path: Path, results_dir: Path):
    """Carga los modelos guardados y ejecuta cross_val_predict (5-fold)."""
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    X, y = load_data(matrix_path)
    SEED = 42
    CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    modelos = {}
    for nombre, fname in [('GBT',      'mejor_modelo_optimizado.joblib'),
                           ('XGBoost',  'modelo_xgboost.joblib'),
                           ('LightGBM', 'modelo_lightgbm.joblib')]:
        ruta = results_dir / fname
        if ruta.exists():
            modelos[nombre] = joblib.load(ruta)
            log.info(f'  Cargado: {fname}')
        else:
            log.warning(f'  No encontrado: {fname} — se omite {nombre}')

    # RF: el mejor_modelo.joblib es un RandomForest entrenado en 07_modelo_baseline.py
    rf_path = results_dir / 'mejor_modelo.joblib'
    if rf_path.exists():
        modelos['RF'] = joblib.load(rf_path)
        log.info(f'  Cargado: mejor_modelo.joblib (RF)')

    roc_data = {}
    for nombre, modelo in modelos.items():
        log.info(f'  cross_val_predict {nombre}...')
        try:
            y_prob = cross_val_predict(
                modelo, X, y, cv=CV,
                method='predict_proba', n_jobs=-1
            )[:, 1]
            auc = roc_auc_score(y, y_prob)
            log.info(f'    AUC = {auc:.4f}')
            roc_data[nombre] = (y.values, y_prob)
        except Exception as e:
            log.error(f'    Error: {e}')

    return roc_data


# ══════════════════════════════════════════════════════════════════════════════
# MODO RÁPIDO — desde TSVs pre-computados
# ══════════════════════════════════════════════════════════════════════════════

def load_metrics_from_tsv(results_dir: Path) -> pd.DataFrame:
    tsv = results_dir / 'metricas_clinicas.tsv'
    df = pd.read_csv(tsv, sep='\t')
    log.info(f'  metricas_clinicas.tsv cargado ({len(df)} modelos)')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FIGURAS
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves_full(roc_data: dict, out: Path):
    """Curvas ROC continuas desde cross_val_predict (modo completo)."""
    from sklearn.metrics import roc_curve, roc_auc_score

    fig, ax = plt.subplots(figsize=(8, 7))
    for nombre, (y_true, y_prob) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=COLORES.get(nombre, '#333333'),
                lw=2.5, label=f'{nombre} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    _format_roc_axes(ax)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


def plot_roc_curves_from_tsv(df_metrics: pd.DataFrame, out: Path):
    """Puntos de operación (Sens/Spec) cuando no se re-ejecutan los modelos."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for _, row in df_metrics.iterrows():
        nombre = row['modelo']
        fpr    = 1 - row['Especificidad']
        tpr    = row['Sensibilidad']
        auc    = row['AUC']
        color  = COLORES.get(nombre, '#333333')
        ax.scatter(fpr, tpr, color=color, s=120, zorder=5,
                   label=f'{nombre} (AUC = {auc:.3f})')
        ax.annotate(nombre, (fpr, tpr), fontsize=9,
                    xytext=(6, 4), textcoords='offset points')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    # Nota informativa
    ax.text(0.98, 0.05,
            'Puntos de operación (TSV)\nPara curvas continuas: omitir --from-tsv',
            ha='right', va='bottom', fontsize=8, color='gray',
            transform=ax.transAxes)
    _format_roc_axes(ax)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


def _format_roc_axes(ax):
    ax.set_xlabel('1 − Especificidad (Tasa de Falsos Positivos)', fontsize=12)
    ax.set_ylabel('Sensibilidad (Tasa de Verdaderos Positivos)', fontsize=12)
    ax.set_title(
        'Curvas ROC — Predicción de Resistencia a Carbapenémicos\n'
        'Klebsiella pneumoniae · Validación Cruzada 5-fold',
        fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)


def plot_confusion_matrices_full(roc_data: dict, out: Path):
    """Matrices de confusión desde cross_val_predict."""
    from sklearn.metrics import confusion_matrix, roc_auc_score

    n = len(roc_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (nombre, (y_true, y_prob)) in zip(axes, roc_data.items()):
        y_pred   = (y_prob >= 0.5).astype(int)
        cm       = confusion_matrix(y_true, y_pred)
        cm_norm  = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        auc      = roc_auc_score(y_true, y_prob)
        _draw_cm(ax, cm, cm_norm, nombre, auc)

    plt.suptitle('Matrices de Confusión — Validación Cruzada 5-fold',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


def plot_confusion_matrices_from_tsv(df_metrics: pd.DataFrame, out: Path):
    """Matrices de confusión desde TP/TN/FP/FN del TSV."""
    n = len(df_metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, df_metrics.iterrows()):
        nombre = row['modelo']
        tp, tn, fp, fn = int(row['TP']), int(row['TN']), int(row['FP']), int(row['FN'])
        cm = np.array([[tn, fp], [fn, tp]])
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        _draw_cm(ax, cm, cm_norm, nombre, row['AUC'])

    plt.suptitle('Matrices de Confusión — Validación Cruzada 5-fold',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


def _draw_cm(ax, cm: np.ndarray, cm_norm: np.ndarray, nombre: str, auc: float):
    ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'{nombre}\nAUC = {auc:.3f}', fontsize=11, fontweight='bold')
    classes     = ['Susceptible', 'Resistant']
    tick_marks  = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=9)
    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f'{cm[i, j]:,}\n({cm_norm[i, j]:.1%})',
                    ha='center', va='center', fontsize=10,
                    color='white' if cm_norm[i, j] > 0.5 else 'black')
    ax.set_ylabel('Real', fontsize=10)
    ax.set_xlabel('Predicho', fontsize=10)


def plot_feature_importance(results_dir: Path, out: Path, top_n: int = 20):
    """Top-N genes más importantes según GBT (desde TSV)."""
    tsv = results_dir / 'feature_importance_GBT.tsv'
    df  = pd.read_csv(tsv, sep='\t', index_col=0)
    df.columns = ['importance']
    df = df.sort_values('importance', ascending=False).head(top_n)

    # Colorear por prefijo (origen de la anotación)
    def _color(gene):
        if gene.startswith('RGI_'):  return '#E74C3C'
        if gene.startswith('RES_'):  return '#3498DB'
        if gene.startswith('PF_'):   return '#2ECC71'
        return '#95A5A6'

    colors = [_color(g) for g in df.index]

    # Etiquetas legibles (sin prefijo)
    labels = [g.replace('RGI_', '').replace('RES_', '').replace('PF_', '')
              for g in df.index]

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos   = np.arange(len(df))
    ax.barh(y_pos, df['importance'].values, color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Importancia (Gradient Boosting)', fontsize=11)
    ax.set_title(
        f'Top {top_n} genes predictores de resistencia a carbapenémicos\n'
        'Gradient Boosting Trees — Feature Importance',
        fontsize=13, fontweight='bold', pad=15)

    # Leyenda de prefijos
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='#E74C3C', label='RGI (CARD)'),
        mpatches.Patch(color='#3498DB', label='ResFinder'),
        mpatches.Patch(color='#2ECC71', label='PointFinder'),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    here      = Path(__file__).resolve().parent
    workspace = here.parents[3]

    p = argparse.ArgumentParser(
        description='Genera figuras de modelos ML para carbapenémicos')
    p.add_argument('--matrix',
                   default=str(workspace / 'ml_matrix_binary.csv.gz'),
                   help='Ruta a ml_matrix_binary.csv.gz')
    p.add_argument('--results_dir',
                   default=str(workspace / 'resultados_reales'),
                   help='Directorio con TSVs y modelos .joblib')
    p.add_argument('--output_dir',
                   default=str(workspace / 'figuras_tfm'),
                   help='Directorio de salida para las figuras')
    p.add_argument('--from-tsv', action='store_true',
                   help='Usar TSVs pre-computados (no re-ejecuta modelos). '
                        'Las curvas ROC son puntos de operación, no curvas continuas.')
    return p.parse_args()


def main():
    args        = parse_args()
    matrix_path = Path(args.matrix)
    results_dir = Path(args.results_dir)
    out_dir     = Path(args.output_dir)
    from_tsv    = args.from_tsv
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info('═══ 18_figuras_modelos_carbapenems.py ═══')
    log.info(f'  Matriz     : {matrix_path}')
    log.info(f'  Resultados : {results_dir}')
    log.info(f'  Salida     : {out_dir}')
    log.info(f'  Modo       : {"TSV (rápido)" if from_tsv else "completo (cross_val_predict)"}')

    # ── Fig ROC + Confusion ────────────────────────────────────────────────
    if from_tsv:
        log.info('\n[Modo TSV] Cargando metricas_clinicas.tsv...')
        df_metrics = load_metrics_from_tsv(results_dir)

        log.info('\n[Fig ROC] Puntos de operación...')
        plot_roc_curves_from_tsv(
            df_metrics, out_dir / 'fig_roc_curves.png')

        log.info('\n[Fig Confusion] Matrices desde TP/TN/FP/FN...')
        plot_confusion_matrices_from_tsv(
            df_metrics, out_dir / 'fig_confusion_matrices.png')
    else:
        log.info('\n[Modo completo] Ejecutando cross_val_predict...')
        log.info('  (puede tardar 5–15 minutos según el hardware)')
        roc_data = run_cross_val(matrix_path, results_dir)

        log.info('\n[Fig ROC] Curvas continuas...')
        plot_roc_curves_full(roc_data, out_dir / 'fig_roc_curves.png')

        log.info('\n[Fig Confusion] Matrices de confusión...')
        plot_confusion_matrices_full(
            roc_data, out_dir / 'fig_confusion_matrices.png')

    # ── Fig Feature Importance (siempre desde TSV) ─────────────────────────
    log.info('\n[Fig Feature Importance] Top-20 genes GBT...')
    plot_feature_importance(
        results_dir, out_dir / 'fig_feature_importance_top20.png')

    log.info('\n═══ Completado ═══')
    for f in sorted(out_dir.glob('fig_*.png')):
        log.info(f'  {f.name}  ({f.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    main()
