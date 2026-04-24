#!/usr/bin/env python3
"""
19_figuras_multiAB.py
─────────────────────────────────────────────────────────────────────────────
Genera las figuras comparativas multi-antibiótico del TFM (figA–figE)
de forma REPRODUCIBLE, a partir de los TSVs de resultados.

NO requiere re-ejecutar modelos ni la matriz de features.

Input (--results_dir, por defecto ./resultados_multiAB/):
  amr_fenotipos_fluoroquinolones_clean.tsv
  amr_fenotipos_cephalosporins3g_clean.tsv
  feature_importance_fluoroquinolones_GBT.tsv
  feature_importance_cephalosporins3g_GBT.tsv
  comparison_all_antibiotics.csv
  resultados_fluoroquinolones.tsv
  resultados_cephalosporins3g.tsv
  metricas_clinicas_fluoroquinolones.tsv
  metricas_clinicas_cephalosporins3g.tsv

Output (--output_dir, por defecto ./figuras_multiab/):
  figA_fenotipos_multiab.png      – Distribución R/S por antibiótico (3 grupos)
  figB_genes_marcadores_multiab.png – Top-10 genes por antibiótico (FQ + Ceph3G)
  figC_roc_multiab.png            – AUC comparativa: todos modelos × antibiótico
  figD_heatmap_metricas.png       – Heatmap Sens/Spec/F1/AUC × modelo × antibiótico
  figE_auc_comparativa.png        – Gráfico de barras agrupadas AUC por modelo

Uso:
  python 19_figuras_multiAB.py
  python 19_figuras_multiAB.py \\
      --results_dir resultados_multiAB/ \\
      --output_dir figuras_multiab/
─────────────────────────────────────────────────────────────────────────────
"""
import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)

# ── Paleta de colores coherente con el TFM ────────────────────────────────
COLOR_AB = {
    'Carbapenémicos':    '#9B59B6',   # morado
    'Fluoroquinolonas':  '#E74C3C',   # rojo
    'Cefalosporinas 3G': '#2E86C1',   # azul
}
COLOR_MOD = {
    'GBT':      '#E74C3C',
    'RF':       '#2ECC71',
    'XGBoost':  '#E67E22',
    'LightGBM': '#F1C40F',
}
COLOR_FEAT = {
    'RGI': '#E74C3C',
    'RES': '#3498DB',
    'PF':  '#2ECC71',
}

plt.rcParams.update({
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def shorten_gene_name(name: str) -> str:
    """Abrevia nombres de genes RGI/RES/PF para que quepan en los ejes."""
    import re
    name = str(name)
    name = name.replace('RGI_', '').replace('RES_', '').replace('PF_', 'PF:')
    name = name.replace('Escherichia coli', 'E. coli')
    name = name.replace('Klebsiella pneumoniae', 'K. pneumoniae')
    name = name.replace('Salmonella serovars', 'Salmonella spp.')
    name = name.replace('Salmonella isangi', 'S. isangi')
    name = name.replace('Acinetobacter baumannii', 'A. baumannii')
    name = name.replace('conferring resistance to fluoroquinolones', '(res. FQ)')
    name = name.replace('conferring resistance to ciprofloxacin and tetracycline', '(res. Cip+Tet)')
    name = name.replace('conferring resistance to', '(res.')
    name = name.replace('mutations conferring resistance', 'mut.')
    name = re.sub(r'\s+', ' ', name).strip()
    if len(name) > 45:
        name = name[:42] + '...'
    return name


# ══════════════════════════════════════════════════════════════════════════════
# Fig A · Distribución fenotípica (R vs S) en los 3 grupos de antibiótico
# ══════════════════════════════════════════════════════════════════════════════

def plot_figA(results_dir: Path, comparison_csv: Path, out: Path):
    """
    Barras agrupadas: proporción R y S para Carbapenémicos, FQ y Ceph3G.
    Datos de n_genomas y Pct_R vienen de comparison_all_antibiotics.csv.
    """
    df = pd.read_csv(comparison_csv)

    # Un solo punto por antibiótico (el mejor modelo es suficiente para contar genomas)
    ab_info = (df.drop_duplicates('Antibiótico')
                 .set_index('Antibiótico')[['N_genomas', 'Pct_R']])
    ab_info['Pct_S'] = 100 - ab_info['Pct_R']
    ab_info['N_R']   = (ab_info['N_genomas'] * ab_info['Pct_R'] / 100).round().astype(int)
    ab_info['N_S']   = ab_info['N_genomas'] - ab_info['N_R']

    antibioticos = list(ab_info.index)
    x   = np.arange(len(antibioticos))
    w   = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars_r = ax.bar(x - w / 2, ab_info['N_R'].values, w,
                    color=[COLOR_AB.get(a, '#95A5A6') for a in antibioticos],
                    alpha=0.85, label='Resistente', edgecolor='white')
    bars_s = ax.bar(x + w / 2, ab_info['N_S'].values, w,
                    color=[COLOR_AB.get(a, '#95A5A6') for a in antibioticos],
                    alpha=0.40, label='Susceptible', edgecolor='white', hatch='//')

    for bar_r, bar_s, ab in zip(bars_r, bars_s, antibioticos):
        n   = ab_info.loc[ab, 'N_genomas']
        pct = ab_info.loc[ab, 'Pct_R']
        ax.text(bar_r.get_x() + bar_r.get_width() / 2,
                bar_r.get_height() + 30,
                f'{int(ab_info.loc[ab,"N_R"]):,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10)
        ax.text(bar_s.get_x() + bar_s.get_width() / 2,
                bar_s.get_height() + 30,
                f'{int(ab_info.loc[ab,"N_S"]):,}\n({100-pct:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(antibioticos, fontsize=11)
    ax.set_ylabel('Número de genomas', fontsize=11)
    ax.set_title(
        'Distribución fenotípica por grupo de antibiótico\n'
        'Klebsiella pneumoniae — BV-BRC',
        fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.set_ylim(0, ab_info[['N_R', 'N_S']].max().max() * 1.25)

    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# Fig B · Top-10 genes marcadores por antibiótico (FQ + Ceph3G)
# ══════════════════════════════════════════════════════════════════════════════

def plot_figB(results_dir: Path, out: Path, top_n: int = 8):
    """Barras horizontales: top genes por antibiótico. Mejorado: figsize 18×8, top_n=8."""
    groups = {
        'Fluoroquinolonas':  results_dir / 'feature_importance_fluoroquinolones_GBT.tsv',
        'Cefalosporinas 3G': results_dir / 'feature_importance_cephalosporins3g_GBT.tsv',
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, (ab, tsv_path) in zip(axes, groups.items()):
        df = pd.read_csv(tsv_path, sep='\t', index_col=0)
        df.columns = ['importance']
        df = df.sort_values('importance', ascending=False).head(top_n)

        labels = [shorten_gene_name(g) for g in df.index]
        colors = [COLOR_FEAT.get(g.split('_')[0], '#95A5A6') for g in df.index]

        y_pos = np.arange(len(df))
        bars = ax.barh(y_pos, df['importance'].values, color=colors,
                       edgecolor='white', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('Importancia (GBT)', fontsize=11)
        ax.set_title(f'{ab}\nTop {top_n} genes marcadores (GBT)',
                     fontsize=13, fontweight='bold', pad=12)
        ax.grid(axis='x', alpha=0.3)
        for bar, val in zip(bars, df['importance'].values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9, color='#333333')

    # Leyenda compartida
    _feat_labels = {'RGI': 'CARD', 'RES': 'ResFinder', 'PF': 'PointFinder'}
    handles = [mpatches.Patch(color=v, label=f'{k} ({_feat_labels.get(k, k)})')
               for k, v in COLOR_FEAT.items()]
    fig.legend(handles=handles, fontsize=10, loc='lower center',
               ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Genes marcadores más importantes por grupo de antibiótico\n'
                 'Gradient Boosting Trees — Feature Importance',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# Fig C · AUC comparativa: todos los modelos × antibiótico
# ══════════════════════════════════════════════════════════════════════════════

def plot_figC(comparison_csv: Path, out: Path):
    """Gráfico de puntos/líneas: AUC por modelo coloreado por antibiótico."""
    df = pd.read_csv(comparison_csv)

    modelos     = df['Modelo'].unique().tolist()
    antibioticos = df['Antibiótico'].unique().tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    x_ticks = np.arange(len(modelos))

    for ab in antibioticos:
        sub = df[df['Antibiótico'] == ab].set_index('Modelo')
        aucs = [sub.loc[m, 'AUC'] if m in sub.index else np.nan for m in modelos]
        stds = [sub.loc[m, 'AUC_SD'] if m in sub.index and 'AUC_SD' in sub.columns
                else 0 for m in modelos]
        color = COLOR_AB.get(ab, '#95A5A6')
        ax.errorbar(x_ticks, aucs, yerr=stds,
                    fmt='o-', color=color, lw=2, markersize=8,
                    label=ab, capsize=4)

    ax.axhline(0.80, color='gray', linestyle='--', lw=1, alpha=0.7,
               label='Objetivo AUC ≥ 0.80')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(modelos, fontsize=11)
    ax.set_ylabel('AUC (ROC) · media ± SD (5-fold)', fontsize=11)
    ax.set_ylim(0.75, 0.98)
    ax.set_title(
        'Comparativa de AUC entre modelos y grupos de antibiótico\n'
        'Klebsiella pneumoniae · Validación Cruzada 5-fold',
        fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# Fig D · Heatmap de métricas (Sens, Spec, F1, AUC) × modelo × antibiótico
# ══════════════════════════════════════════════════════════════════════════════

def plot_figD(results_dir: Path, comparison_csv: Path, out: Path):
    """Heatmap multi-métrica, multi-antibiótico, multi-modelo."""
    df_comp = pd.read_csv(comparison_csv)

    # Cargar métricas clínicas de FQ y Ceph3G
    metricas = {}
    for key, fname in [('Fluoroquinolonas',  'metricas_clinicas_fluoroquinolones.tsv'),
                       ('Cefalosporinas 3G', 'metricas_clinicas_cephalosporins3g.tsv')]:
        path = results_dir / fname
        if path.exists():
            tmp = pd.read_csv(path, sep='\t').rename(columns={'modelo': 'Modelo'})
            tmp['Antibiótico'] = key
            metricas[key] = tmp

    # Construir tabla wide: índice = (Antibiótico, Modelo), columnas = métricas
    rows = []
    for ab, sub in df_comp.groupby('Antibiótico'):
        for _, row in sub.iterrows():
            m = row['Modelo']
            entry = {
                'Antibiótico': ab,
                'Modelo':      m,
                'AUC':         row['AUC'],
                'F1':          row['F1'],
                'Sensibilidad': row.get('Sensibilidad', np.nan),
                'Especificidad': row.get('Especificidad', np.nan),
            }
            # Enriquecer con métricas clínicas si están disponibles
            if ab in metricas:
                mc = metricas[ab]
                mc_row = mc[mc['Modelo'].str.upper() == m.upper()]
                if not mc_row.empty:
                    entry['Sensibilidad']  = mc_row.iloc[0].get('Sensibilidad', entry['Sensibilidad'])
                    entry['Especificidad'] = mc_row.iloc[0].get('Especificidad', entry['Especificidad'])
            rows.append(entry)

    df = pd.DataFrame(rows)
    df['Grupo'] = df['Antibiótico'] + '\n' + df['Modelo']
    pivot = df.set_index('Grupo')[['AUC', 'F1', 'Sensibilidad', 'Especificidad']]

    fig, ax = plt.subplots(figsize=(9, max(8, len(pivot) * 0.65)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0.6, vmax=1.0, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Valor', 'shrink': 0.6},
                annot_kws={'size': 11})
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=11)
    ax.set_title(
        'Métricas de rendimiento por modelo y grupo de antibiótico\n'
        'Klebsiella pneumoniae · Validación Cruzada 5-fold',
        fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Métrica', fontsize=11)
    ax.set_ylabel('Antibiótico · Modelo', fontsize=11)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# Fig E · Barras agrupadas AUC por modelo y antibiótico
# ══════════════════════════════════════════════════════════════════════════════

def plot_figE(comparison_csv: Path, out: Path):
    """Barras agrupadas: AUC para cada modelo, agrupado por antibiótico."""
    df = pd.read_csv(comparison_csv)

    antibioticos = df['Antibiótico'].unique().tolist()
    modelos      = ['GBT', 'XGBoost', 'LightGBM', 'RF']
    modelos      = [m for m in modelos if m in df['Modelo'].values]

    x = np.arange(len(antibioticos))
    w = 0.18
    offsets = np.linspace(-(len(modelos)-1)/2, (len(modelos)-1)/2, len(modelos)) * w

    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, modelo in zip(offsets, modelos):
        aucs = []
        stds = []
        for ab in antibioticos:
            sub = df[(df['Antibiótico'] == ab) & (df['Modelo'] == modelo)]
            aucs.append(sub['AUC'].values[0] if len(sub) else np.nan)
            stds.append(sub['AUC_SD'].values[0]
                        if len(sub) and 'AUC_SD' in sub.columns else 0)
        bars = ax.bar(x + offset, aucs, w, yerr=stds,
                      color=COLOR_MOD.get(modelo, '#95A5A6'),
                      label=modelo, edgecolor='white', alpha=0.88,
                      capsize=4)
        for bar, auc in zip(bars, aucs):
            if not np.isnan(auc):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f'{auc:.3f}', ha='center', va='bottom', fontsize=7)

    ax.axhline(0.80, color='gray', linestyle='--', lw=1, alpha=0.7,
               label='Objetivo AUC ≥ 0.80')
    ax.set_xticks(x)
    ax.set_xticklabels(antibioticos, fontsize=11)
    ax.set_ylabel('AUC (ROC)', fontsize=11)
    ax.set_ylim(0.75, 0.98)
    ax.set_title(
        'AUC por modelo y grupo de antibiótico\n'
        'Klebsiella pneumoniae · Validación Cruzada 5-fold',
        fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=10, ncol=len(modelos))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    here      = Path(__file__).resolve().parent
    workspace = here.parents[3]

    p = argparse.ArgumentParser(
        description='Genera figuras multi-AB del TFM (figA–figE)')
    p.add_argument('--results_dir',
                   default=str(workspace / 'resultados_multiAB'),
                   help='Directorio con TSVs de resultados multi-AB')
    p.add_argument('--output_dir',
                   default=str(workspace / 'figuras_multiab'),
                   help='Directorio de salida para las figuras')
    return p.parse_args()


def main():
    args        = parse_args()
    results_dir = Path(args.results_dir)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison  = results_dir / 'comparison_all_antibiotics.csv'

    log.info('═══ 19_figuras_multiAB.py — Figuras multi-antibiótico ═══')
    log.info(f'  Resultados : {results_dir}')
    log.info(f'  Salida     : {out_dir}')

    log.info('\n[Fig A] Distribución fenotípica por antibiótico...')
    plot_figA(results_dir, comparison, out_dir / 'figA_fenotipos_multiab.png')

    log.info('\n[Fig B] Top-10 genes marcadores por antibiótico...')
    plot_figB(results_dir, out_dir / 'figB_genes_marcadores_multiab.png')

    log.info('\n[Fig C] AUC comparativa todos modelos × antibiótico...')
    plot_figC(comparison, out_dir / 'figC_roc_multiab.png')

    log.info('\n[Fig D] Heatmap métricas completas...')
    plot_figD(results_dir, comparison, out_dir / 'figD_heatmap_metricas.png')

    log.info('\n[Fig E] Barras agrupadas AUC por modelo...')
    plot_figE(comparison, out_dir / 'figE_auc_comparativa.png')

    log.info('\n═══ Completado ═══')
    for f in sorted(out_dir.glob('fig*.png')):
        log.info(f'  {f.name}  ({f.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    main()
