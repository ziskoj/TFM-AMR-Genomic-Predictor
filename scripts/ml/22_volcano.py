#!/usr/bin/env python3
"""
22_volcano.py  —  Volcano plot TFM AMR K. pneumoniae
=====================================================
Genera: docs/figures/fig7_volcano_final.png

Diseño de etiquetas:
  • Columna IZQUIERDA  (log2OR < 0, genes protectores/susceptibilidad):
    anchor a la derecha de xlim, ha='right' → texto se extiende HACIA LA IZQUIERDA.
    Máximo 6 genes (los de mayor −log10 FDR).

  • Columna DERECHA (log2OR ≥ 0, genes de resistencia):
    anchor a la derecha de todos los datos, ha='left' → texto se extiende
    HACIA LA DERECHA dentro del margen. Máximo 12 genes.

  La separación en signo de OR garantiza que las flechas nunca crucen el eje Y.

Ejecutar desde:
  /mnt/f/TFM-AMR-Genomic-Predictor/scripts/ml/

  source /mnt/f/TFM_Linux/envs/amr_env/bin/activate
  python 22_volcano.py
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency

# ─── Rutas ────────────────────────────────────────────────────────────────────
here = Path(__file__).resolve().parent   # scripts/ml/
REPO = here.parents[1]                   # repo root  (scripts/ → repo root)
DATA = REPO / "data"
FIGS = REPO / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid':       False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# Corrección BH-FDR
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Carga y cálculo de estadísticos
# ═══════════════════════════════════════════════════════════════════════════════

def load_stats():
    print("  Cargando ml_matrix_binary.csv.gz ...")
    df = pd.read_csv(DATA / "ml_matrix_binary.csv.gz", index_col=0)
    df.index = df.index.astype(str).str.strip()
    feat_cols = [c for c in df.columns if c != 'target']
    res = df[df['target'] == 1]
    sus = df[df['target'] == 0]
    print(f"  {len(df)} genomas | R={int((df['target']==1).sum())} | "
          f"S={int((df['target']==0).sum())} | {len(feat_cols)} features")

    records = []
    for g in feat_cols:
        a = int(res[g].sum());  b = int(len(res) - a)
        c = int(sus[g].sum());  d = int(len(sus) - c)
        try:
            chi2, p, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)
        except Exception:
            chi2, p = 0.0, 1.0
        or_val = (a * d) / (b * c + 1e-9)
        records.append({'gene': g, 'chi2': chi2, 'p': p, 'OR': or_val})

    dv = pd.DataFrame(records)
    dv['fdr']    = bh_fdr(dv['p'].values)
    dv['log2OR'] = np.log2(dv['OR'].clip(1e-4, None))
    dv['neglog'] = -np.log10(dv['fdr'].clip(1e-70, None))
    return dv


# ═══════════════════════════════════════════════════════════════════════════════
# Etiquetas curadas para el volcano (nombres cortos, biológicamente relevantes)
# ═══════════════════════════════════════════════════════════════════════════════

# Genes de resistencia (log2OR > 0 esperado)
VLABELS_R = {
    'RES_blaOXA-9':   'blaOXA-9',
    'RES_blaKPC-2':   'blaKPC-2',
    'RES_blaKPC-3':   'blaKPC-3',
    'RES_blaNDM-1':   'blaNDM-1',
    'RES_blaVIM-1':   'blaVIM-1',
    'RES_blaOXA-48':  'blaOXA-48',
    'RES_blaTEM-1A':  'blaTEM-1A',
    'RES_catA1':      'catA1',
    'RGI_catA1':      'catA1',
    'RGI_BRP(MBL)':   'BRP(MBL)',
    'RGI_armA':       'armA',
    'RGI_rmtB':       'rmtB',
    'RGI_Escherichia coli gyrA conferring resistance to fluoroquinolones': 'gyrA',
    'RGI_Escherichia coli parC conferring resistance to fluoroquinolones': 'parC',
    'RGI_Salmonella serovars gyrB conferring resistance to fluoroquinolones': 'gyrB',
    'RGI_qacEdelta1': 'qacEdelta1',
    'RGI_Mrx':        'Mrx',
    'RES_ARR-3':      'ARR-3',
}

# Genes protectores / susceptibilidad (log2OR < 0 esperado)
VLABELS_S = {
    'RGI_tet(A)':        'tet(A)',
    'RGI_adeF':          'adeF',
    'PF_ompK36 p.T184P': 'ompK36*',
    'PF_ompK36 p.N49S':  'ompK36 N49S*',
    'PF_ompK37 p.N230G': 'ompK37*',
    'PF_acrR p.F172S':   'acrR*',
    'RGI_eptB':          'eptB',
    'RGI_CRP':           'CRP',
}

ALL_VLABELS = {**VLABELS_R, **VLABELS_S}

# Número máximo de etiquetas por columna
N_LEFT  = 5    # susceptibilidad (izquierda)
N_RIGHT = 12   # resistencia (derecha)


# ═══════════════════════════════════════════════════════════════════════════════
# Volcano plot
# ═══════════════════════════════════════════════════════════════════════════════

def fig_volcano(dv):
    print("\n[Volcano] Dibujando ...")

    SIG_R = (dv['fdr'] < 0.05) & (dv['log2OR'] > 0)
    SIG_S = (dv['fdr'] < 0.05) & (dv['log2OR'] < 0)
    NS    = ~(SIG_R | SIG_S)

    # Seleccionar genes a etiquetar
    to_label = dv[dv['gene'].isin(ALL_VLABELS) & (SIG_R | SIG_S)].copy()
    to_label['vlabel'] = to_label['gene'].map(ALL_VLABELS)
    to_label = to_label.sort_values('chi2', ascending=False).drop_duplicates('vlabel')

    # Columna izquierda: genes genuinamente asociados a susceptibilidad (OR < 1)
    left_genes  = (to_label[to_label['log2OR'] < 0]
                   .sort_values('neglog', ascending=False)
                   .head(N_LEFT))

    # Columna derecha: genes genuinamente asociados a resistencia (OR > 1)
    right_genes = (to_label[to_label['log2OR'] > 0]
                   .sort_values('neglog', ascending=False)
                   .head(N_RIGHT))

    print(f"  Etiquetas izquierda ({len(left_genes)}): {left_genes['vlabel'].tolist()}")
    print(f"  Etiquetas derecha  ({len(right_genes)}): {right_genes['vlabel'].tolist()}")

    # ── Coordenadas clave ────────────────────────────────────────────────────
    x_data_min = dv['log2OR'].min()   # ~−3.28
    x_data_max = dv['log2OR'].max()   # ~+4.87
    y_max_sig  = dv.loc[SIG_R | SIG_S, 'neglog'].max()

    # Columna izquierda: texto alineado a la DERECHA del anchor
    # ha='right' → el borde derecho del texto está en x_anchor_l
    # El texto se extiende hacia la izquierda (fuera de los datos)
    x_anchor_l  = x_data_min - 1.2   # borde derecho de las etiquetas izq.
    x_lim_left  = x_anchor_l  - 2.0  # xlim: deja 2 unidades para el texto

    # Columna derecha: texto alineado a la IZQUIERDA del anchor
    # ha='left' → el borde izquierdo del texto está en x_anchor_r
    # El texto se extiende hacia la derecha (fuera de los datos)
    x_anchor_r  = x_data_max + 0.8   # borde izquierdo de las etiquetas der.
    x_lim_right = x_anchor_r  + 2.5  # xlim: deja 2.5 unidades para el texto

    # ── Figura ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    # Puntos
    ax.scatter(dv.loc[NS,    'log2OR'], dv.loc[NS,    'neglog'],
               s=18, color='#CCCCCC', alpha=0.55, zorder=2)
    ax.scatter(dv.loc[SIG_S, 'log2OR'], dv.loc[SIG_S, 'neglog'],
               s=32, color='#3B82F6', alpha=0.75, zorder=3,
               edgecolors='white', linewidths=0.4)
    ax.scatter(dv.loc[SIG_R, 'log2OR'], dv.loc[SIG_R, 'neglog'],
               s=32, color='#DC2626', alpha=0.75, zorder=3,
               edgecolors='white', linewidths=0.4)

    # Líneas de umbral
    umbral = -np.log10(0.05)
    ax.axhline(umbral, color='#888888', lw=1, ls='--', alpha=0.7, zorder=1)
    ax.axvline(0,      color='#888888', lw=0.8, ls='--', alpha=0.7, zorder=1)
    # Etiqueta de la línea FDR (coordenadas de datos)
    ax.text(0.2, umbral + 1.0, 'FDR = 0.05', fontsize=8,
            color='#888', style='italic', ha='center')

    # ── Etiquetas columna izquierda ─────────────────────────────────────────
    # ha='right': el borde derecho del texto queda en x_anchor_l
    # El texto NO toca los datos (que están a la derecha de x_data_min)
    if len(left_genes) > 0:
        y_pos_l = np.linspace(y_max_sig * 0.88, y_max_sig * 0.10, len(left_genes))
        for (_, row), y_text in zip(left_genes.iterrows(), y_pos_l):
            ax.annotate(
                row['vlabel'],
                xy=(row['log2OR'], row['neglog']),
                xytext=(x_anchor_l, y_text),
                fontsize=8.5, fontweight='bold', color='#1E3A8A',
                ha='right', va='center',
                arrowprops=dict(
                    arrowstyle='->', color='#1E40AF',
                    lw=0.9, shrinkA=2, shrinkB=3,
                    connectionstyle='arc3,rad=0.0'),
                zorder=7)

    # ── Etiquetas columna derecha ────────────────────────────────────────────
    # ha='left': el borde izquierdo del texto queda en x_anchor_r
    # El texto NO toca los datos (que están a la izquierda de x_data_max)
    if len(right_genes) > 0:
        y_pos_r = np.linspace(y_max_sig * 0.95, y_max_sig * 0.05, len(right_genes))
        for (_, row), y_text in zip(right_genes.iterrows(), y_pos_r):
            ax.annotate(
                row['vlabel'],
                xy=(row['log2OR'], row['neglog']),
                xytext=(x_anchor_r, y_text),
                fontsize=8.5, fontweight='bold', color='#7F1D1D',
                ha='left', va='center',
                arrowprops=dict(
                    arrowstyle='->', color='#991B1B',
                    lw=0.9, shrinkA=2, shrinkB=3,
                    connectionstyle='arc3,rad=0.0'),
                zorder=7)

    # ── Ejes y decoraciones ──────────────────────────────────────────────────
    ax.set_xlim(x_lim_left, x_lim_right)
    ax.set_ylim(-1, y_max_sig * 1.08)

    ax.set_xlabel('log$_2$(Odds Ratio)', fontsize=12)
    ax.set_ylabel('−log$_{10}$(p-valor ajustado FDR-BH)', fontsize=12)
    ax.set_title(
        'Volcano Plot — Asociación de determinantes genéticos AMR con resistencia\n'
        'a carbapenémicos en K. pneumoniae  (n = 4.044; Fisher exacto, FDR-BH)',
        fontsize=11, fontweight='bold', pad=12)

    handles = [
        Patch(facecolor='#DC2626', alpha=0.75,
              label='Asociado a resistencia (OR > 1, p-adj < 0.05)'),
        Patch(facecolor='#3B82F6', alpha=0.75,
              label='Asociado a susceptibilidad (OR < 1, p-adj < 0.05)'),
        Patch(facecolor='#CCCCCC', alpha=0.6, label='No significativo'),
        Line2D([0], [0], ls='--', color='#888', lw=1,
               label='Umbral FDR-BH = 0.05'),
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
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 62)
    print("22_volcano.py — Volcano plot AMR K. pneumoniae")
    print(f"REPO  : {REPO}")
    print(f"FIGS  : {FIGS}")
    print("=" * 62)

    dv = load_stats()
    fig_volcano(dv)

    print("\n" + "=" * 62)
    print("Figura generada:", FIGS / "fig7_volcano_final.png")
    print("=" * 62)
