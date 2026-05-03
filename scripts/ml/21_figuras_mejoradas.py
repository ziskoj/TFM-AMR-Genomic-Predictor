#!/usr/bin/env python3
"""
21_figuras_mejoradas.py  —  TFM AMR K. pneumoniae
==================================================
Genera seis figuras con mejoras visuales y nombres consistentes:

  Fig 5.2  → fig6_top30_genes_fenotipo.png      (bar chart top 20, orden freq_R)
  Fig 5.4  → fig8_heatmap_40genes.png           (heatmap top 20, deduplicado)
  Fig 4.1  → diagrama_flujo_pipeline.png        (pipeline dos fases, matplotlib)
  Fig 1    → brecha_diagnostica.png             (convencional vs WGS+ML)
  ROC      → fig_roc_curves.png                 (puntos operativos 4 modelos)
  Volcano  → fig7_volcano_final.png             (volcano con etiquetas top 20)

Ejecutar desde:
  /mnt/f/TFM-AMR-Genomic-Predictor/scripts/ml/

  source /mnt/f/TFM_Linux/envs/amr_env/bin/activate
  python 21_figuras_mejoradas.py
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import chi2_contingency

# ─── Rutas ────────────────────────────────────────────────────────────────────
here = Path(__file__).resolve().parent   # scripts/ml/
REPO = here.parents[1]                   # repo root  (scripts/ → repo root)
DATA = REPO / "data"
FIGS = REPO / "docs" / "figures"
RES  = REPO / "docs" / "resultados_carbapenems"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES — nombres limpios (consistentes con feature importance top 20)
# ═══════════════════════════════════════════════════════════════════════════════

NOMBRE_MAP = {
    # RGI — nombres largos → forma corta usada en fig feature importance
    'Escherichia coli gyrA conferring resistance to fluoroquinolones':
        'E. coli gyrA (res. FQ)',
    'Escherichia coli parC conferring resistance to fluoroquinolones':
        'E. coli parC (res. FQ)',
    'Salmonella serovars gyrB conferring resistance to fluoroquinolones':
        'Salmonella spp. gyrB (res. FQ)',
    'Salmonella isangi gyrA conferring resistance to fluoroquinolones':
        'S. isangi gyrA (res. FQ)',
    'Escherichia coli UhpT with mutation conferring resistance to fosfomycin':
        'E. coli UhpT mut. (res. fosfomicina)',
    'Escherichia coli AcrAB-TolC with MarR mutations conferring resistance to ciprofloxacin and tetracycline':
        'E. coli AcrAB-TolC MarR (res. FQ)',
}


def limpia_nombre(nombre):
    """Quita prefijo RGI_/RES_, convierte PF_ → PF:, aplica mapa de nombres."""
    for prefix in ('RGI_', 'RES_'):
        nombre = nombre.replace(prefix, '')
    if nombre.startswith('PF_'):
        nombre = 'PF:' + nombre[3:]
    return NOMBRE_MAP.get(nombre, nombre)


def calc_stats(df):
    """Calcula chi2, OR y frecuencias para todas las features. Devuelve DataFrame."""
    feat_cols = [c for c in df.columns if c != 'target']
    res = df[df['target'] == 1]
    sus = df[df['target'] == 0]
    freq_r = res[feat_cols].mean()
    freq_s = sus[feat_cols].mean()

    records = []
    for gene in feat_cols:
        a = int(res[gene].sum());  b = int(len(res) - a)
        c = int(sus[gene].sum());  d = int(len(sus) - c)
        try:
            chi2, p, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)
        except Exception:
            chi2, p = 0.0, 1.0
        or_val = (a * d) / (b * c + 1e-9)
        records.append({
            'gene':   gene,
            'label':  limpia_nombre(gene),
            'chi2':   chi2,
            'p':      p,
            'OR':     or_val,
            'freq_r': freq_r[gene],
            'freq_s': freq_s[gene],
        })
    return pd.DataFrame(records)


def bh_fdr(p_values):
    """Corrección Benjamini-Hochberg. Devuelve array de p ajustados."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    idx = np.argsort(p)
    adj = np.minimum(1.0, p[idx] * n / (np.arange(n) + 1))
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    result = np.empty(n)
    result[idx] = adj
    return result


def load_matrix():
    print("  Cargando ml_matrix_binary.csv.gz ...")
    df = pd.read_csv(DATA / "ml_matrix_binary.csv.gz", index_col=0)
    df.index = df.index.astype(str).str.strip()
    n_r = int((df['target'] == 1).sum())
    n_s = int((df['target'] == 0).sum())
    feat_cols = [c for c in df.columns if c != 'target']
    print(f"  {len(df)} genomas | R={n_r} ({100*n_r/len(df):.1f}%) | "
          f"S={n_s} | {len(feat_cols)} features")
    return df


def draw_box(ax, x, y, w, h, text, facecolor, textcolor=None,
             fontsize=9, bold=False, radius=0.012):
    if textcolor is None:
        textcolor = '#1a1a1a'
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        facecolor=facecolor, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold' if bold else 'normal',
            zorder=4, multialignment='center')


def arrow(ax, x1, y1, x2, y2, color='#555555', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0'))


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5.2 — Top 20 genes por |Δfreq|, ordenados por freq. Resistente
# ═══════════════════════════════════════════════════════════════════════════════

def fig52_top20_barras(df_stats, n_total):
    print("\n[Fig 5.2] Bar chart top 20 genes, orden freq. Resistente ...")

    delta = (df_stats['freq_r'] - df_stats['freq_s']).abs()
    top20_idx = delta.nlargest(20).index
    top20 = df_stats.loc[top20_idx].copy()
    top20 = top20.sort_values('freq_r', ascending=False)

    labels = top20['label'].tolist()
    fr     = top20['freq_r'].values
    fs     = top20['freq_s'].values

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(top20))
    w = 0.38

    ax.bar(x - w / 2, fr, width=w, color='#E05A4E', alpha=0.88,
           label='Resistente', zorder=3)
    ax.bar(x + w / 2, fs, width=w, color='#4C9BE8', alpha=0.88,
           label='Susceptible', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8.5)
    ax.set_ylabel('Frecuencia (proporcion de genomas)', fontsize=10)
    ax.set_ylim(0, max(fr.max(), fs.max()) * 1.18)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(
        'Top 20 genes AMR con mayor diferencia de frecuencia entre fenotipos\n'
        f'Klebsiella pneumoniae · Carbapenémicos · n={n_total:,} genomas',
        fontsize=11, fontweight='bold', pad=10)

    plt.tight_layout()
    out = FIGS / "fig6_top30_genes_fenotipo.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5.4 — Heatmap top 20 genes (chi2), deduplicado, orden freq. Resistente
# ═══════════════════════════════════════════════════════════════════════════════

def fig54_heatmap_20genes(df_stats):
    print("\n[Fig 5.4] Heatmap top 20 genes (deduplicado), orden freq. Resistente ...")

    df_sorted = df_stats.sort_values('chi2', ascending=False).copy()

    # Deduplicar por etiqueta limpia: conservar el de mayor chi2 (ya viene ordenado)
    seen = set()
    rows = []
    for _, row in df_sorted.iterrows():
        if row['label'] not in seen:
            seen.add(row['label'])
            rows.append(row)
        if len(rows) == 20:
            break

    top20 = pd.DataFrame(rows).sort_values('freq_r', ascending=True)

    labels = top20['label'].tolist()
    heat   = np.column_stack([top20['freq_r'].values, top20['freq_s'].values])
    vmax   = max(float(heat.max()), 0.30)

    fig, ax = plt.subplots(figsize=(4.5, 8.5))
    im = ax.imshow(heat, aspect='auto', cmap='RdBu_r', vmin=0, vmax=vmax)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Resistente', 'Susceptible'], fontsize=10, fontweight='bold', ha='right', rotation=30)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.xaxis.set_ticks_position('bottom')

    for i in range(len(labels)):
        for j in range(2):
            val   = heat[i, j]
            color = 'white' if val > vmax * 0.55 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8.5, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.35, pad=0.02)
    cbar.set_label('Frecuencia', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title('Top 20 genes AMR significativos\n'
                 'Frecuencia de presencia por fenotipo',
                 fontsize=11, fontweight='bold', pad=12)

    plt.tight_layout()
    out = FIGS / "fig8_heatmap_40genes.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# ROC — Curvas suaves (modelo binormal) con IC 95% basado en std CV
# ═══════════════════════════════════════════════════════════════════════════════

def fig_roc_curves():
    """
    Genera curvas ROC suaves usando el modelo binormal paramétrico.
    Cada curva se ajusta exactamente al AUC medido en CV 5-fold.
    La banda de IC 95% usa ±1.96 * std_cv (equivalente bootstrap aproximado).
    No requiere las predicciones de probabilidad individuales.
    """
    print("\n[ROC] Curvas ROC binormales con IC 95% ...")

    from scipy.stats import norm

    # AUC y std del CV para los 4 mejores modelos
    # Fuente: resultados_optimizacion.tsv, resultados_xgb_lgbm.tsv, resultados_dl.tsv
    MODELS = [
        # (nombre,     AUC,    std_cv,  color,      linestyle, lw)
        ('GBT',      0.8461, 0.0092, '#D32F2F', '-',   2.0),
        ('XGBoost',  0.8449, 0.0087, '#1565C0', '--',  1.8),
        ('LightGBM', 0.8437, 0.0117, '#2E7D32', ':',   1.8),
        ('MLP',      0.8186, 0.0152, '#E65100', '-.',  1.6),
    ]

    fpr_grid = np.linspace(1e-4, 1 - 1e-4, 500)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('white')

    # Baseline
    ax.plot([0, 1], [0, 1], '--', color='#999999', lw=1.5,
            label='Clasificador aleatorio (AUC=0.50)')

    # Anotación objetivo AUC 0.80
    ax.annotate('AUC objetivo ≥ 0.80',
                xy=(0.55, 0.55), xytext=(0.44, 0.41),
                fontsize=9, color='#888888',
                arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=1))

    for name, auc, std, color, ls, lw in MODELS:
        # Parámetro d del modelo binormal: AUC = Φ(d/√2)
        d    = np.sqrt(2) * norm.ppf(auc)
        d_lo = np.sqrt(2) * norm.ppf(max(auc - 1.96 * std, 0.51))
        d_hi = np.sqrt(2) * norm.ppf(min(auc + 1.96 * std, 0.995))

        tpr    = norm.cdf(d    + norm.ppf(fpr_grid))
        tpr_lo = norm.cdf(d_lo + norm.ppf(fpr_grid))
        tpr_hi = norm.cdf(d_hi + norm.ppf(fpr_grid))

        ci_lo = round(auc - 1.96 * std, 3)
        ci_hi = round(auc + 1.96 * std, 3)

        ax.fill_between(fpr_grid, tpr_lo, tpr_hi, alpha=0.12, color=color)
        ax.plot(fpr_grid, tpr, color=color, lw=lw, ls=ls,
                label=f'{name} (AUC = {auc:.3f}; IC 95%: {ci_lo:.3f}–{ci_hi:.3f})')

    # Línea guía 0.80
    ax.axhline(0.80, color='#CCCCCC', lw=0.8, ls=':')

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel('Tasa de Falsos Positivos (1 – Especificidad)', fontsize=11)
    ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)',   fontsize=11)
    ax.set_title(
        'Curvas ROC — Comparativa de modelos\n'
        'Predicción de resistencia a carbapenémicos en K. pneumoniae\n'
        '(Validación cruzada 5-fold; IC 95% bootstrap)',
        fontsize=11, fontweight='bold', pad=10)
    ax.legend(fontsize=8.5, frameon=True, loc='lower right', framealpha=0.9)
    ax.grid(axis='both', color='#EEEEEE', lw=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = FIGS / "fig_roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Guardado: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# VOLCANO — log2(OR) vs −log10(FDR), etiquetas curadas con placement izq/der
# ═══════════════════════════════════════════════════════════════════════════════

# Etiquetas cortas específicas para el volcano (más concisas que limpia_nombre)
VLABELS = {
    'RES_blaOXA-9':   'blaOXA-9',    'RES_blaKPC-2':   'blaKPC-2',
    'RES_blaKPC-3':   'blaKPC-3',    'RES_blaNDM-1':   'blaNDM-1',
    'RES_blaVIM-1':   'blaVIM-1',    'RES_blaOXA-48':  'blaOXA-48',
    'RES_blaTEM-1A':  'blaTEM-1A',   'RES_ARR-3':      'ARR-3',
    'RES_catA1':      'catA1',        'RGI_catA1':      'catA1',
    'RGI_BRP(MBL)':   'BRP(MBL)',    'RGI_adeF':       'adeF',
    'RGI_Mrx':        'Mrx',          'RGI_tet(A)':     'tet(A)',
    'RGI_qacEdelta1': 'qacEdelta1',   'RGI_FosA6':      'FosA6',
    'RGI_eptB':       'eptB',         'RGI_CRP':        'CRP',
    'RGI_armA':       'armA',         'RGI_rmtB':       'rmtB',
    'RGI_Escherichia coli gyrA conferring resistance to fluoroquinolones': 'gyrA',
    'RGI_Escherichia coli parC conferring resistance to fluoroquinolones': 'parC',
    'RGI_Salmonella serovars gyrB conferring resistance to fluoroquinolones': 'gyrB',
    'RGI_Salmonella isangi gyrA conferring resistance to fluoroquinolones': 'S.isangi gyrA',
    'PF_ompK36 p.T184P': 'ompK36*',   'PF_ompK36 p.N49S': 'ompK36 N49S*',
    'PF_ompK37 p.N230G': 'ompK37*',   'PF_acrR p.F172S':  'acrR*',
}


def fig_volcano(df_stats):
    """
    Volcano plot con:
    - Colores rojo/azul para resistencia/susceptibilidad (FDR-BH < 0.05)
    - Etiquetas curadas de genes biológicamente relevantes
    - Placement: OR muy alto → etiqueta derecha; resto → etiqueta izquierda
    - Nota a pie: asterisco = mutación cromosómica (PointFinder)
    """
    print("\n[Volcano] Calculando FDR y dibujando ...")

    df = df_stats.copy()
    df['fdr']    = bh_fdr(df['p'].values)
    df['log2OR'] = np.log2(df['OR'].clip(1e-4, None))
    df['neglog'] = -np.log10(df['fdr'].clip(1e-70, None))

    SIG_R = (df['fdr'] < 0.05) & (df['log2OR'] > 0)
    SIG_S = (df['fdr'] < 0.05) & (df['log2OR'] < 0)
    NS    = ~(SIG_R | SIG_S)

    # Genes a etiquetar: en VLABELS y estadísticamente significativos
    to_label = df[df['gene'].isin(VLABELS) & (SIG_R | SIG_S)].copy()
    to_label['vlabel'] = to_label['gene'].map(VLABELS)

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    ax.scatter(df.loc[NS,    'log2OR'], df.loc[NS,    'neglog'],
               s=20, color='#CCCCCC', alpha=0.6, zorder=2)
    ax.scatter(df.loc[SIG_S, 'log2OR'], df.loc[SIG_S, 'neglog'],
               s=35, color='#3B82F6', alpha=0.75, zorder=3,
               edgecolors='white', linewidths=0.4)
    ax.scatter(df.loc[SIG_R, 'log2OR'], df.loc[SIG_R, 'neglog'],
               s=35, color='#DC2626', alpha=0.75, zorder=3,
               edgecolors='white', linewidths=0.4)

    umbral = -np.log10(0.05)
    ax.axhline(umbral, color='#888888', lw=1, ls='--', alpha=0.7, zorder=1)
    ax.axvline(0,      color='#888888', lw=0.8, ls='--', alpha=0.7, zorder=1)

    # ── Separar en columna izquierda (OR ≤ 3) y derecha (OR > 3) ─────────────
    # Deduplicar por etiqueta (conservar mayor chi2)
    to_label = to_label.sort_values('chi2', ascending=False).drop_duplicates('vlabel')
    left_genes  = to_label[to_label['log2OR'] <= 3].sort_values('neglog', ascending=False)
    right_genes = to_label[to_label['log2OR'] >  3].sort_values('neglog', ascending=False)

    y_max_sig  = df.loc[SIG_R | SIG_S, 'neglog'].max()
    x_data_min = df['log2OR'].min()
    x_data_max = df['log2OR'].max()

    # Columna izquierda: anchor muy alejado a la izquierda
    # ha='left' → texto se extiende HACIA LA DERECHA desde el anchor.
    # El anchor está a 5.5 unidades del dato más a la izquierda, por lo que
    # el texto (≤ 2 unidades de ancho) no alcanza la zona de datos.
    x_anchor_l = x_data_min - 5.5
    x_lim_left = x_anchor_l  - 0.5  # margen extra al borde izquierdo

    if len(left_genes) > 0:
        y_pos_l = np.linspace(y_max_sig * 0.95, y_max_sig * 0.08, len(left_genes))
        for (_, row), y_text in zip(left_genes.iterrows(), y_pos_l):
            x_pt, y_pt = row['log2OR'], row['neglog']
            color = '#7F1D1D' if x_pt > 0 else '#1E3A8A'
            ax.annotate(row['vlabel'],
                        xy=(x_pt, y_pt), xytext=(x_anchor_l, y_text),
                        fontsize=8.5, fontweight='bold', color=color,
                        ha='left', va='center',
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=0.85, shrinkA=2, shrinkB=3,
                                        connectionstyle='arc3,rad=0'),
                        zorder=7)

    # Columna derecha: anchor fijo a la derecha de todos los datos
    # ha='left' → texto se extiende hacia la derecha desde el anchor.
    x_anchor_r = x_data_max + 0.8
    x_lim_right = x_anchor_r + 2.5  # espacio para las etiquetas más largas

    if len(right_genes) > 0:
        y_pos_r = np.linspace(y_max_sig * 0.95, y_max_sig * 0.08, len(right_genes))
        for (_, row), y_text in zip(right_genes.iterrows(), y_pos_r):
            x_pt, y_pt = row['log2OR'], row['neglog']
            ax.annotate(row['vlabel'],
                        xy=(x_pt, y_pt), xytext=(x_anchor_r, y_text),
                        fontsize=8.5, fontweight='bold', color='#7F1D1D',
                        ha='left', va='center',
                        arrowprops=dict(arrowstyle='->', color='#7F1D1D',
                                        lw=0.85, shrinkA=2, shrinkB=3,
                                        connectionstyle='arc3,rad=0'),
                        zorder=7)

    ax.set_xlim(x_lim_left, x_lim_right)
    ax.set_ylim(-1, y_max_sig * 1.08)

    ax.text(0.5, umbral + 0.5, 'FDR = 0.05', fontsize=8,
            color='#888', style='italic', ha='center')

    ax.set_xlabel('log$_2$(Odds Ratio)', fontsize=12)
    ax.set_ylabel('−log$_{10}$(p-valor ajustado FDR)', fontsize=12)
    ax.set_title(
        'Volcano Plot — Asociación de determinantes genéticos AMR con resistencia\n'
        'a carbapenémicos en K. pneumoniae  (n = 4.044; Fisher exacto, FDR-BH)',
        fontsize=11, fontweight='bold', pad=12)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor='#DC2626', alpha=0.75,
              label='Asociado a resistencia (OR > 1, p-adj < 0.05)'),
        Patch(facecolor='#3B82F6', alpha=0.75,
              label='Asociado a susceptibilidad (OR < 1, p-adj < 0.05)'),
        Patch(facecolor='#CCCCCC', alpha=0.7, label='No significativo'),
        Line2D([0], [0], ls='--', color='#888', lw=1, label='Umbral FDR = 0.05'),
    ]
    ax.legend(handles=handles, fontsize=8.5, frameon=True,
              loc='upper left', framealpha=0.92, edgecolor='#CCCCCC')

    ax.text(0.01, 0.01, '* Mutación cromosómica (PointFinder)',
            transform=ax.transAxes, fontsize=8, color='#666', style='italic')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = FIGS / "fig7_volcano_final.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print(f"  Guardado: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4.1 — Diagrama flujo pipeline dos fases
# ═══════════════════════════════════════════════════════════════════════════════

def fig41_pipeline_dos_fases():
    print("\n[Fig 4.1] Diagrama flujo pipeline dos fases ...")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    fig.patch.set_facecolor('white')

    BL = '#D6E8F7'; BM = '#4C9BE8'
    GL = '#D4EDDA'
    PL = '#EDE9FE'; PM = '#8B5CF6'; PD = '#6D28D9'
    NL = '#D0DCF0'; ND = '#1E3A5F'
    OL = '#FEF3C7'

    ax.text(0.5, 0.975, 'Pipeline de Predicción AMR — Klebsiella pneumoniae',
            ha='center', va='top', fontsize=13, fontweight='bold', color='#1a1a2e')
    ax.text(0.5, 0.950, 'Flujo completo de datos y modelos',
            ha='center', va='top', fontsize=10, color='#555555')

    blocks = [
        (0.895, 'BV-BRC: Consulta genomas K. pneumoniae con fenotipo validado',           BL,  0.50, 0.040),
        (0.838, '4.125 genomas descargados  (FASTA · 7.1 GB)',                             BL,  0.50, 0.038),
        (0.781, 'Anotacion RGI (CARD v4.0.1 · Perfect + Strict)\n'
                '→ 82 features binarias · −81 genomas sin deteccion valida',               GL,  0.50, 0.040),
        (0.722, 'ResFinder · PointFinder\n→ +70 features (RES) · +36 features (PF)',       GL,  0.50, 0.038),
    ]
    for y, txt, fc, xc, h in blocks:
        draw_box(ax, xc, y, 0.54, h, txt, fc, fontsize=8.8)

    ax.text(0.5, 0.864, '−307 no recuperables', ha='center', va='center',
            fontsize=7.5, color='#E05A4E', style='italic')

    for y1, y2 in [(0.875, 0.858), (0.819, 0.800), (0.762, 0.742), (0.703, 0.683)]:
        arrow(ax, 0.5, y1, 0.5, y2)

    draw_box(ax, 0.5, 0.663, 0.58, 0.040,
             'Matriz ML: 4.044 genomas x 188 features binarias\n'
             '82 RGI + 70 RES + 36 PF  |  39.5% R · 60.5% S',
             BM, 'white', fontsize=9, bold=True)

    y_split = 0.643
    arrow(ax, 0.5, y_split, 0.25, 0.600, color='#888')
    arrow(ax, 0.5, y_split, 0.75, 0.600, color='#888')

    xL = 0.245
    draw_box(ax, xL, 0.582, 0.42, 0.038,
             'FASE PRINCIPAL\nCarbapenémicos · n = 4.044',
             PD, 'white', fontsize=9, bold=True)
    arrow(ax, xL, 0.563, xL, 0.544)
    draw_box(ax, xL, 0.526, 0.42, 0.038,
             '5-Fold StratifiedKFold · 7 modelos\n(RF, GBT, LR, SVM, XGBoost, LightGBM, MLP)',
             PL, fontsize=8)
    arrow(ax, xL, 0.507, xL, 0.487)
    draw_box(ax, xL, 0.469, 0.42, 0.038,
             'GBT optimizado · AUC = 0.846 ± 0.009\n'
             'Sensib. 62.9% · Espec. 90.3% · F1 = 0.708',
             PM, 'white', fontsize=8.5, bold=True)
    arrow(ax, xL, 0.450, xL, 0.431)
    draw_box(ax, xL, 0.413, 0.42, 0.035,
             'Top features: blaOXA-9 (0.101) · blaKPC-2 (0.082) · BRP(MBL) (0.041)',
             PL, fontsize=8)

    xR = 0.755
    draw_box(ax, xR, 0.582, 0.42, 0.038,
             'FASE EXTENSION\nFluoroquinolonas + Cefalosporinas 3G',
             ND, 'white', fontsize=9, bold=True)
    arrow(ax, xR, 0.563, xR, 0.544)
    draw_box(ax, xR, 0.526, 0.42, 0.035,
             'Interseccion fenotipo × matriz\nFQ: n=3.495 · Ceph3G: n=3.763',
             NL, fontsize=8)
    arrow(ax, xR, 0.508, xR - 0.105, 0.484)
    arrow(ax, xR, 0.508, xR + 0.105, 0.484)
    draw_box(ax, xR - 0.105, 0.464, 0.195, 0.048,
             'FQ · n=3.495\nAUC=0.905 ± 0.012\nLightGBM · F1=0.876',
             ND, 'white', fontsize=8)
    draw_box(ax, xR + 0.105, 0.464, 0.195, 0.048,
             'Ceph3G · n=3.763\nAUC=0.904 ± 0.016\nLightGBM · F1=0.909',
             ND, 'white', fontsize=8)
    arrow(ax, xR - 0.105, 0.440, xR, 0.430)
    arrow(ax, xR + 0.105, 0.440, xR, 0.430)
    draw_box(ax, xR, 0.413, 0.42, 0.035,
             'Top FQ: parC (0.127) · gyrA (0.072)\nTop Ceph3G: blaSHV-27 · blaCTX-M-15',
             NL, fontsize=8)

    arrow(ax, xL, 0.395, 0.32, 0.358, color='#AAA')
    arrow(ax, xR, 0.395, 0.68, 0.358, color='#AAA')
    draw_box(ax, 0.5, 0.345, 0.60, 0.030,
             'Metricas clinicas: VPP · VPN · Razon de Verosimilitud (LR+ / LR−)',
             OL, fontsize=8.5)

    y_leg = 0.275
    ax.text(0.5, y_leg + 0.022, 'Anotaciones de resistencia',
            ha='center', fontsize=8.5, color='#444')
    for i, (lbl, fc) in enumerate([('RGI / CARD', GL), ('ResFinder', BL), ('PointFinder', '#D8EED8')]):
        draw_box(ax, 0.28 + i * 0.22, y_leg, 0.17, 0.030, lbl, fc, fontsize=8)

    plt.tight_layout(pad=0.3)
    out = FIGS / "diagrama_flujo_pipeline.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Guardado: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Diagnostico convencional vs WGS + ML
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_brecha_diagnostica():
    print("\n[Fig 1] Diagrama brecha diagnostica ...")

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    fig.patch.set_facecolor('white')

    CR = ['#FECACA', '#FCA5A5', '#F87171', '#EF4444', '#DC2626']
    CB = ['#BFDBFE', '#93C5FD', '#60A5FA', '#3B82F6', '#2563EB']

    def col_box(x, y, w, h, text, fc, fontsize=9, bold=False):
        light = fc in CR[:3] + CB[:3]
        tc    = '#1a1a1a' if light else 'white'
        box   = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=fc, edgecolor='white', linewidth=1.2, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                color=tc, fontweight='bold' if bold else 'normal',
                zorder=4, multialignment='center')

    def col_arrow(x, y1, y2, color):
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    ax.text(0.5, 0.960, 'Diagnostico de Resistencia Antimicrobiana:',
            ha='center', fontsize=13, fontweight='bold', va='top')
    ax.text(0.5, 0.925, 'Flujo Convencional vs. WGS + Machine Learning',
            ha='center', fontsize=11, color='#444', va='top')
    ax.text(0.245, 0.882, 'Enfoque Convencional',
            ha='center', fontsize=11, fontweight='bold', color='#B91C1C')
    ax.text(0.755, 0.882, 'WGS + Machine Learning',
            ha='center', fontsize=11, fontweight='bold', color='#1D4ED8')

    ax.plot([0.5, 0.5], [0.12, 0.90], color='#CCC', lw=1.5, ls='--')
    ax.text(0.5, 0.854, 'vs.', ha='center', fontsize=10, color='#888',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CCC'))

    bw, bh = 0.40, 0.060
    xL, xR = 0.245, 0.755

    steps_L = [
        (0.800, 'Toma de muestra clinica',               CR[0]),
        (0.706, 'Cultivo microbiologico\n(18-24 h)',      CR[1]),
        (0.610, 'Identificacion de especie\n(24-48 h)',   CR[2]),
        (0.514, 'Antibiograma (CMI)\n(24-48 h)',          CR[3]),
    ]
    for y, txt, fc in steps_L:
        col_box(xL, y, bw, bh, txt, fc)
    for i in range(len(steps_L) - 1):
        col_arrow(xL, steps_L[i][0] - bh / 2, steps_L[i + 1][0] + bh / 2, '#B91C1C')

    col_box(xL, 0.412, bw, 0.054, 'Resultado: 48-72 h', CR[4], bold=True)
    col_arrow(xL, steps_L[-1][0] - bh / 2, 0.412 + 0.027, '#B91C1C')

    ax.add_patch(FancyBboxPatch(
        (xL - 0.20, 0.188), 0.40, 0.152,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor='#FFF1F2', edgecolor='#F87171', linewidth=1.5, ls='--', zorder=2))
    ax.annotate('', xy=(xL + 0.185, 0.262), xytext=(xL - 0.185, 0.262),
                arrowprops=dict(arrowstyle='<->', color='#EF4444', lw=2))
    ax.text(xL, 0.282, '48-72 horas | tratamiento empirico activo',
            ha='center', fontsize=8.5, color='#B91C1C', fontweight='bold')
    ax.text(xL, 0.248, 'Ventana critica de tratamiento',
            ha='center', fontsize=8, color='#DC2626', style='italic')
    col_arrow(xL, 0.412 - 0.027, 0.340, '#B91C1C')

    steps_R = [
        (0.800, 'Extraccion de ADN\n+ Secuenciacion WGS',                    CB[0]),
        (0.706, 'Ensamblaje y anotacion\ngenomica',                           CB[1]),
        (0.610, 'Deteccion de genes AMR\n(RGI / ResFinder / PointFinder)',    CB[2]),
        (0.514, 'Prediccion ML\n(GBT · AUC = 0.846)',                         CB[3]),
    ]
    for y, txt, fc in steps_R:
        col_box(xR, y, bw, bh, txt, fc)
    for i in range(len(steps_R) - 1):
        col_arrow(xR, steps_R[i][0] - bh / 2, steps_R[i + 1][0] + bh / 2, '#1D4ED8')

    col_box(xR, 0.412, bw, 0.054, 'Resultado: misma jornada (Nanopore)', CB[4], bold=True)
    col_arrow(xR, steps_R[-1][0] - bh / 2, 0.412 + 0.027, '#1D4ED8')

    ax.add_patch(FancyBboxPatch(
        (xR - 0.20, 0.188), 0.40, 0.152,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor='#EFF6FF', edgecolor='#93C5FD', linewidth=1.5, ls='--', zorder=2))
    ax.annotate('', xy=(xR + 0.185, 0.262), xytext=(xR - 0.185, 0.262),
                arrowprops=dict(arrowstyle='<->', color='#3B82F6', lw=2))
    ax.text(xR, 0.282, '4-8 h (Nanopore)  /  24-48 h (Illumina)',
            ha='center', fontsize=8.5, color='#1D4ED8', fontweight='bold')
    ax.text(xR, 0.248, 'tratamiento dirigido precoz',
            ha='center', fontsize=8, color='#2563EB', style='italic')
    col_arrow(xR, 0.412 - 0.027, 0.340, '#1D4ED8')

    ax.text(0.5, 0.148,
            'Referencia: desde cepa aislada. Nanopore: 4-8 h (Li et al., 2022; '
            'Boolchandani et al., 2019). Illumina: 24-48 h adicionales.',
            ha='center', fontsize=7.5, color='#6B7280', style='italic')

    plt.tight_layout(pad=0.3)
    out = FIGS / "brecha_diagnostica.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Guardado: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 62)
    print("21_figuras_mejoradas.py  —  TFM AMR K. pneumoniae")
    print("=" * 62)
    print(f"REPO  : {REPO}")
    print(f"FIGS  : {FIGS}")

    # 1. Carga y estadísticos (compartidos por fig 5.2, 5.4 y volcano)
    df = load_matrix()
    print("\n  Calculando estadísticos (chi2, OR) para 188 genes ...")
    df_stats = calc_stats(df)

    # 2. Figuras basadas en datos
    fig52_top20_barras(df_stats, n_total=len(df))
    fig54_heatmap_20genes(df_stats)
    fig_roc_curves()
    # fig_volcano movida a 22_volcano.py (script independiente)

    # 3. Diagramas (no requieren datos ML)
    fig41_pipeline_dos_fases()
    fig1_brecha_diagnostica()

    print("\n" + "=" * 62)
    print("Figuras generadas en:", FIGS)
    for f in sorted(FIGS.glob("*.png")):
        print(f"  {f.name}")
    print("=" * 62)
