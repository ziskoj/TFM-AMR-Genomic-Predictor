#!/usr/bin/env python3
"""
08_eda.py
Análisis Exploratorio de Datos (EDA) de la matriz ML
Genera estadísticas, visualizaciones y análisis estadístico de genes AMR
Input:  ml_matrix_binary.csv.gz + dataset_final.tsv
Output: gráficas y tablas en /mnt/f/TFM_Linux/ml_matrices/eda/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
import logging
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ──────────────────────────────────────────────────────────────────
ML_DIR   = Path('/mnt/f/TFM_Linux/ml_matrices')
EDA_DIR  = ML_DIR / 'eda'
LOG_DIR  = Path('/mnt/f/MIS_DATOS_TFM/logs')
EDA_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'eda.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Estilo de gráficas ─────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})
COLORS = {'Susceptible': '#2E86C1', 'Resistant': '#E74C3C'}

def load_data():
    """Carga la matriz ML."""
    log.info('Cargando matriz ML...')
    df = pd.read_csv(ML_DIR / 'ml_matrix_binary.csv.gz', index_col=0)
    X = df.drop(columns=['target'])
    y = df['target'].map({0: 'Susceptible', 1: 'Resistant'})
    log.info(f'  Shape: {df.shape}')
    log.info(f'  Balance: {y.value_counts().to_dict()}')
    return X, y, df

# ── 1. Balance de clases ───────────────────────────────────────────────────
def plot_balance(y):
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = y.value_counts()
    bars = ax.bar(vc.index, vc.values,
                  color=[COLORS[c] for c in vc.index], width=0.5, edgecolor='white')
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:,}\n({val/len(y):.1%})', ha='center', va='bottom', fontsize=11)
    ax.set_title('Distribución de Fenotipos AMR\nCarbapenémicos (meropenem + imipenem)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Número de genomas', fontsize=11)
    ax.set_ylim(0, vc.max() * 1.2)
    plt.tight_layout()
    out = EDA_DIR / '01_balance_clases.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  Guardado: {out}')

# ── 2. Frecuencia de genes ─────────────────────────────────────────────────
def plot_gene_frequency(X, y, top_n=30):
    df = X.copy()
    df['clase'] = y.values

    freq_r = df[df['clase']=='Resistant'].drop(columns='clase').mean()
    freq_s = df[df['clase']=='Susceptible'].drop(columns='clase').mean()

    # Top genes por frecuencia global
    freq_global = X.mean().sort_values(ascending=False).head(top_n)
    genes = freq_global.index

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(genes))
    w = 0.35
    ax.bar(x - w/2, freq_r[genes], w, label='Resistant',
           color=COLORS['Resistant'], alpha=0.85)
    ax.bar(x + w/2, freq_s[genes], w, label='Susceptible',
           color=COLORS['Susceptible'], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Frecuencia (proporción de genomas)', fontsize=11)
    ax.set_title(f'Top {top_n} genes AMR más frecuentes\npor fenotipo de resistencia',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    out = EDA_DIR / '02_frecuencia_genes.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  Guardado: {out}')

# ── 3. Análisis estadístico Chi-cuadrado + corrección FDR ─────────────────
def statistical_analysis(X, y):
    log.info('Ejecutando análisis estadístico...')
    y_bin = (y == 'Resistant').astype(int)
    resultados = []

    for gen in X.columns:
        tabla = pd.crosstab(X[gen], y_bin)
        if tabla.shape == (2, 2):
            _, p_fisher = fisher_exact(tabla)
            chi2, p_chi2, _, _ = chi2_contingency(tabla)
            # Odds ratio
            a, b = tabla.iloc[1, 1], tabla.iloc[1, 0]
            c, d = tabla.iloc[0, 1], tabla.iloc[0, 0]
            or_val = (a * d) / (b * c) if (b * c) > 0 else np.nan
            resultados.append({
                'gen': gen,
                'freq_resistant': X[y=='Resistant'][gen].mean(),
                'freq_susceptible': X[y=='Susceptible'][gen].mean(),
                'odds_ratio': or_val,
                'p_fisher': p_fisher,
                'p_chi2': p_chi2,
            })

    df_stats = pd.DataFrame(resultados)

    # Corrección por múltiples comparaciones (FDR Benjamini-Hochberg)
    _, p_adj, _, _ = multipletests(df_stats['p_fisher'], method='fdr_bh')
    df_stats['p_adj_fdr'] = p_adj
    df_stats['significativo'] = df_stats['p_adj_fdr'] < 0.05
    df_stats = df_stats.sort_values('p_adj_fdr')

    out = EDA_DIR / 'estadisticas_genes.tsv'
    df_stats.to_csv(out, sep='\t', index=False)
    log.info(f'  Genes significativos (FDR<0.05): {df_stats["significativo"].sum()}')
    log.info(f'  Guardado: {out}')

    # Top genes más asociados a resistencia
    log.info('\nTop 10 genes más asociados a RESISTENCIA:')
    top_r = df_stats[df_stats['odds_ratio'] > 1].head(10)
    log.info(top_r[['gen','freq_resistant','freq_susceptible','odds_ratio','p_adj_fdr']].to_string())

    return df_stats

# ── 4. Volcano plot ────────────────────────────────────────────────────────
def plot_volcano(df_stats):
    fig, ax = plt.subplots(figsize=(10, 7))
    df = df_stats.copy()
    df['log2_or'] = np.log2(df['odds_ratio'].clip(0.01, 100))
    df['-log10_p'] = -np.log10(df['p_adj_fdr'].clip(1e-300))

    # Colorear por significancia y dirección
    colors = df.apply(lambda r:
        COLORS['Resistant'] if r['significativo'] and r['log2_or'] > 0
        else COLORS['Susceptible'] if r['significativo'] and r['log2_or'] < 0
        else '#AAAAAA', axis=1)

    ax.scatter(df['log2_or'], df['-log10_p'], c=colors, alpha=0.7, s=40)
    ax.axhline(-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)

    # Etiquetar top genes
    top = df[df['significativo']].nlargest(10, '-log10_p')
    for _, row in top.iterrows():
        ax.annotate(row['gen'], (row['log2_or'], row['-log10_p']),
                    fontsize=7, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    patch_r = mpatches.Patch(color=COLORS['Resistant'], label='Asociado a Resistencia')
    patch_s = mpatches.Patch(color=COLORS['Susceptible'], label='Asociado a Susceptibilidad')
    patch_ns = mpatches.Patch(color='#AAAAAA', label='No significativo')
    ax.legend(handles=[patch_r, patch_s, patch_ns], fontsize=10)
    ax.set_xlabel('log2(Odds Ratio)', fontsize=12)
    ax.set_ylabel('-log10(p-valor ajustado FDR)', fontsize=12)
    ax.set_title('Volcano Plot — Asociación de genes AMR con resistencia a carbapenémicos',
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    out = EDA_DIR / '03_volcano_plot.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  Guardado: {out}')

# ── 5. Heatmap top genes ───────────────────────────────────────────────────
def plot_heatmap(X, y, df_stats, top_n=40):
    log.info('Generando heatmap...')
    top_genes = df_stats[df_stats['significativo']].head(top_n)['gen'].tolist()
    if len(top_genes) < 5:
        top_genes = df_stats.head(top_n)['gen'].tolist()

    # Frecuencia por clase
    df = X[top_genes].copy()
    df['clase'] = y.values
    freq = df.groupby('clase')[top_genes].mean().T

    fig, ax = plt.subplots(figsize=(8, max(8, len(top_genes) * 0.3)))
    sns.heatmap(freq, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0.5, vmin=0, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Frecuencia'})
    ax.set_title(f'Top {top_n} genes AMR significativos\nFrecuencia por fenotipo',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Fenotipo', fontsize=11)
    ax.set_ylabel('Gen AMR', fontsize=11)
    plt.tight_layout()
    out = EDA_DIR / '04_heatmap_genes.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  Guardado: {out}')

# ── 6. Distribución de genes por genoma ───────────────────────────────────
def plot_genes_per_genome(X, y):
    n_genes = X.sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for clase, color in COLORS.items():
        mask = y == clase
        ax.hist(n_genes[mask], bins=30, alpha=0.7,
                color=color, label=f'{clase} (n={mask.sum()})', edgecolor='white')
    ax.set_xlabel('Número de genes AMR detectados por genoma', fontsize=11)
    ax.set_ylabel('Frecuencia', fontsize=11)
    ax.set_title('Distribución de genes AMR por genoma\nsegún fenotipo de resistencia',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    plt.tight_layout()
    out = EDA_DIR / '05_genes_por_genoma.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    log.info(f'  Guardado: {out}')

# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    log.info('=== EDA — Pipeline AMR ===')
    X, y, df = load_data()

    log.info('\n1. Balance de clases...')
    plot_balance(y)

    log.info('\n2. Frecuencia de genes...')
    plot_gene_frequency(X, y)

    log.info('\n3. Análisis estadístico...')
    df_stats = statistical_analysis(X, y)

    log.info('\n4. Volcano plot...')
    plot_volcano(df_stats)

    log.info('\n5. Heatmap...')
    plot_heatmap(X, y, df_stats)

    log.info('\n6. Genes por genoma...')
    plot_genes_per_genome(X, y)

    log.info('\n=== EDA completado ===')
    log.info(f'Gráficas guardadas en: {EDA_DIR}')
    print(f'\nArchivos generados en {EDA_DIR}:')
    for f in sorted(EDA_DIR.glob('*')):
        print(f'  {f.name}')

if __name__ == '__main__':
    main()
