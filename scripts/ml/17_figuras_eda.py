#!/usr/bin/env python3
"""
17_figuras_eda.py
─────────────────────────────────────────────────────────────────────────────
Genera las figuras EDA del TFM (fig4–fig8) de forma REPRODUCIBLE.
Refactorizado de 08_eda.py con rutas configurables por argumento.

Input  (por defecto los ficheros del workspace):
  --matrix   ml_matrix_binary.csv.gz
  --dataset  dataset_final.tsv

Output (--output_dir, por defecto ./figuras_tfm/):
  fig4_distribucion_fenotipos.png  – Balance de clases (R vs S)
  fig5_genes_amr_histograma.png    – Distribución de genes AMR por genoma
  fig6_top30_genes_fenotipo.png    – Top-30 genes más frecuentes por fenotipo
  fig7_volcano_final.png           – Volcano plot (log2OR vs -log10 FDR)
  fig8_heatmap_40genes.png         – Heatmap top-40 genes significativos
  estadisticas_genes_eda.tsv       – Tabla estadística (OR, p_adj, etc.)

Uso:
  python 17_figuras_eda.py
  python 17_figuras_eda.py --matrix data/ml_matrix_binary.csv.gz \\
                            --output_dir resultados/figuras/
─────────────────────────────────────────────────────────────────────────────
"""
import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ── Estilo global ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})
COLORS = {'Susceptible': '#2E86C1', 'Resistant': '#E74C3C'}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)


# ── Carga de datos ─────────────────────────────────────────────────────────
def load_data(matrix_path: Path):
    log.info(f'Cargando matriz ML: {matrix_path}')
    df = pd.read_csv(matrix_path)
    df['genome_id'] = df['genome_id'].astype(str).str.strip()
    df = df.set_index('genome_id')
    X = df.drop(columns=['target'])
    y = df['target'].map({0: 'Susceptible', 1: 'Resistant'})
    log.info(f'  Shape: {df.shape}  |  Balance: {y.value_counts().to_dict()}')
    return X, y


# ── Fig 4 · Balance de clases ──────────────────────────────────────────────
def plot_balance(y: pd.Series, out: Path):
    """fig4_distribucion_fenotipos.png"""
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = y.value_counts()
    bars = ax.bar(vc.index, vc.values,
                  color=[COLORS[c] for c in vc.index],
                  width=0.5, edgecolor='white')
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f'{val:,}\n({val / len(y):.1%})',
                ha='center', va='bottom', fontsize=11)
    ax.set_title(
        'Distribución de Fenotipos AMR\nCarbapenémicos (meropenem + imipenem)',
        fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Número de genomas', fontsize=11)
    ax.set_ylim(0, vc.max() * 1.22)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


# ── Fig 5 · Distribución de genes AMR por genoma ──────────────────────────
def plot_genes_per_genome(X: pd.DataFrame, y: pd.Series, out: Path):
    """fig5_genes_amr_histograma.png"""
    n_genes = X.sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for clase, color in COLORS.items():
        mask = y == clase
        ax.hist(n_genes[mask], bins=30, alpha=0.7,
                color=color, label=f'{clase} (n={mask.sum():,})',
                edgecolor='white')
    ax.set_xlabel('Número de genes AMR detectados por genoma', fontsize=11)
    ax.set_ylabel('Frecuencia (nº de genomas)', fontsize=11)
    ax.set_title(
        'Distribución de genes AMR por genoma\nsegún fenotipo de resistencia',
        fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


# ── Fig 6 · Top-30 genes más frecuentes por fenotipo ─────────────────────
def plot_gene_frequency(X: pd.DataFrame, y: pd.Series, out: Path, top_n: int = 30):
    """fig6_top30_genes_fenotipo.png"""
    freq_r = X[y == 'Resistant'].mean()
    freq_s = X[y == 'Susceptible'].mean()
    genes  = X.mean().sort_values(ascending=False).head(top_n).index

    fig, ax = plt.subplots(figsize=(12, 7))
    xv, w = np.arange(len(genes)), 0.35
    ax.bar(xv - w / 2, freq_r[genes], w,
           label='Resistant', color=COLORS['Resistant'], alpha=0.85)
    ax.bar(xv + w / 2, freq_s[genes], w,
           label='Susceptible', color=COLORS['Susceptible'], alpha=0.85)
    ax.set_xticks(xv)
    ax.set_xticklabels(genes, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Frecuencia (proporción de genomas)', fontsize=11)
    ax.set_title(
        f'Top {top_n} genes AMR más frecuentes\npor fenotipo de resistencia',
        fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.12)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


# ── Análisis estadístico (Chi² + FDR) ─────────────────────────────────────
def statistical_analysis(X: pd.DataFrame, y: pd.Series, out_tsv: Path) -> pd.DataFrame:
    log.info('  Calculando Chi² + FDR (puede tardar ~30 s)...')
    y_bin = (y == 'Resistant').astype(int)
    rows = []
    for gen in X.columns:
        tabla = pd.crosstab(X[gen], y_bin)
        if tabla.shape != (2, 2):
            continue
        _, p_fisher = fisher_exact(tabla)
        chi2, p_chi2, _, _ = chi2_contingency(tabla)
        a, b = tabla.iloc[1, 1], tabla.iloc[1, 0]
        c, d = tabla.iloc[0, 1], tabla.iloc[0, 0]
        or_val = (a * d) / (b * c) if (b * c) > 0 else np.nan
        rows.append({
            'gen': gen,
            'freq_resistant':   X[y == 'Resistant'][gen].mean(),
            'freq_susceptible': X[y == 'Susceptible'][gen].mean(),
            'odds_ratio': or_val,
            'p_fisher': p_fisher,
            'p_chi2':   p_chi2,
        })

    df = pd.DataFrame(rows)
    _, p_adj, _, _ = multipletests(df['p_fisher'], method='fdr_bh')
    df['p_adj_fdr']   = p_adj
    df['significativo'] = df['p_adj_fdr'] < 0.05
    df = df.sort_values('p_adj_fdr').reset_index(drop=True)
    df.to_csv(out_tsv, sep='\t', index=False)
    log.info(f'  Genes significativos (FDR<0.05): {df["significativo"].sum()}')
    log.info(f'  ✔ {out_tsv.name}')
    return df


# ── Fig 7 · Volcano plot ───────────────────────────────────────────────────
def plot_volcano(df_stats: pd.DataFrame, out: Path):
    """fig7_volcano_final.png"""
    df = df_stats.copy()
    df['log2_or']    = np.log2(df['odds_ratio'].clip(0.01, 100))
    df['-log10_p']   = -np.log10(df['p_adj_fdr'].clip(1e-300))
    colors = df.apply(lambda r:
        COLORS['Resistant']   if r['significativo'] and r['log2_or'] > 0
        else COLORS['Susceptible'] if r['significativo'] and r['log2_or'] < 0
        else '#AAAAAA', axis=1)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df['log2_or'], df['-log10_p'], c=colors, alpha=0.7, s=40)
    ax.axhline(-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    ax.axvline(0,               color='gray', linestyle='--', linewidth=1)

    top = df[df['significativo']].nlargest(10, '-log10_p')
    for _, row in top.iterrows():
        ax.annotate(row['gen'], (row['log2_or'], row['-log10_p']),
                    fontsize=7, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    handles = [
        mpatches.Patch(color=COLORS['Resistant'],   label='Asociado a Resistencia'),
        mpatches.Patch(color=COLORS['Susceptible'], label='Asociado a Susceptibilidad'),
        mpatches.Patch(color='#AAAAAA',             label='No significativo'),
    ]
    ax.legend(handles=handles, fontsize=10)
    ax.set_xlabel('log₂(Odds Ratio)', fontsize=12)
    ax.set_ylabel('−log₁₀(p-valor ajustado FDR)', fontsize=12)
    ax.set_title(
        'Volcano Plot — Asociación de genes AMR con resistencia a carbapenémicos\n'
        'Klebsiella pneumoniae · n=4 044 genomas',
        fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


# ── Fig 8 · Heatmap top-40 genes ──────────────────────────────────────────
def plot_heatmap(X: pd.DataFrame, y: pd.Series,
                 df_stats: pd.DataFrame, out: Path, top_n: int = 40):
    """fig8_heatmap_40genes.png"""
    top_genes = df_stats[df_stats['significativo']].head(top_n)['gen'].tolist()
    if len(top_genes) < 5:
        top_genes = df_stats.head(top_n)['gen'].tolist()

    df = X[top_genes].copy()
    df['clase'] = y.values
    freq = df.groupby('clase')[top_genes].mean().T   # genes × clases

    fig, ax = plt.subplots(figsize=(8, max(8, len(top_genes) * 0.3)))
    sns.heatmap(freq, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0.5, vmin=0, vmax=1,
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Frecuencia'})
    ax.set_title(
        f'Top {top_n} genes AMR significativos\nFrecuencia por fenotipo',
        fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Fenotipo', fontsize=11)
    ax.set_ylabel('Gen AMR', fontsize=11)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  ✔ {out.name}')


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    # Detectar ruta base del script para defaults relativos
    here = Path(__file__).resolve().parent
    # Subimos hasta encontrar la raíz del workspace (donde está ml_matrix_binary)
    # Estructura esperada: workspace/repo_scripts/scripts/ml/17_figuras_eda.py
    workspace = here.parents[3]  # workspace/

    p = argparse.ArgumentParser(
        description='Genera figuras EDA del TFM (fig4–fig8)')
    p.add_argument('--matrix',
                   default=str(workspace / 'ml_matrix_binary.csv.gz'),
                   help='Ruta a ml_matrix_binary.csv.gz')
    p.add_argument('--output_dir',
                   default=str(workspace / 'figuras_tfm'),
                   help='Directorio de salida para las figuras')
    return p.parse_args()


def main():
    args = parse_args()
    matrix_path = Path(args.matrix)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info('═══ 17_figuras_eda.py — Figuras EDA ═══')
    log.info(f'  Matriz  : {matrix_path}')
    log.info(f'  Salida  : {out_dir}')

    X, y = load_data(matrix_path)

    log.info('\n[Fig 4] Distribución de fenotipos...')
    plot_balance(y, out_dir / 'fig4_distribucion_fenotipos.png')

    log.info('\n[Fig 5] Histograma genes por genoma...')
    plot_genes_per_genome(X, y, out_dir / 'fig5_genes_amr_histograma.png')

    log.info('\n[Fig 6] Top-30 genes por fenotipo...')
    plot_gene_frequency(X, y, out_dir / 'fig6_top30_genes_fenotipo.png')

    log.info('\n[Análisis estadístico] Chi² + FDR...')
    df_stats = statistical_analysis(
        X, y, out_dir / 'estadisticas_genes_eda.tsv')

    log.info('\n[Fig 7] Volcano plot...')
    plot_volcano(df_stats, out_dir / 'fig7_volcano_final.png')

    log.info('\n[Fig 8] Heatmap top-40 genes...')
    plot_heatmap(X, y, df_stats, out_dir / 'fig8_heatmap_40genes.png')

    log.info('\n═══ Completado ═══')
    log.info(f'Figuras guardadas en: {out_dir}')
    for f in sorted(out_dir.glob('fig*.png')):
        log.info(f'  {f.name}  ({f.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    main()
