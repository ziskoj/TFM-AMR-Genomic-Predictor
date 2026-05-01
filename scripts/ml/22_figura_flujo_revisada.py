#!/usr/bin/env python3
"""
22_figura_flujo_revisada.py  —  TFM AMR K. pneumoniae

Genera la figura comparativa del diagnóstico AMR convencional vs. WGS + ML.
Output: docs/figures/brecha_diagnostica.png

Ejecutar desde scripts/ml/ con el entorno activado:
  source /mnt/f/TFM_Linux/envs/amr_env/bin/activate
  python 22_figura_flujo_revisada.py
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

here = Path(__file__).resolve().parent   # scripts/ml/
REPO = here.parents[1]                   # repo root
FIGS = REPO / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})


def fig1_brecha_diagnostica():
    print("\n[Fig 1] Diagrama brecha diagnóstica ...")

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    CR = ['#FECACA', '#FCA5A5', '#F87171', '#EF4444', '#DC2626']
    CB = ['#BFDBFE', '#93C5FD', '#60A5FA', '#3B82F6', '#2563EB']

    def col_box(x, y, w, h, text, fc, fontsize=9, bold=False):
        light = fc in CR[:3] + CB[:3]
        tc = '#1a1a1a' if light else 'white'
        box = FancyBboxPatch(
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

    ax.text(0.5, 0.960, 'Diagnóstico de Resistencia Antimicrobiana:',
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

    # Lado izquierdo — convencional
    steps_L = [
        (0.800, 'Toma de muestra clínica',              CR[0]),
        (0.706, 'Cultivo microbiológico\n(18-24 h)',     CR[1]),
        (0.610, 'Identificación de especie\n(24-48 h)',  CR[2]),
        (0.514, 'Antibiograma (CMI)\n(24-48 h)',         CR[3]),
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
    ax.text(xL, 0.282, '48-72 horas | tratamiento empírico activo',
            ha='center', fontsize=8.5, color='#B91C1C', fontweight='bold')
    ax.text(xL, 0.248, 'Ventana crítica de tratamiento',
            ha='center', fontsize=8, color='#DC2626', style='italic')
    col_arrow(xL, 0.412 - 0.027, 0.340, '#B91C1C')

    # Lado derecho — WGS + ML
    steps_R = [
        (0.800, 'Extracción de ADN\n+ Secuenciación WGS',                 CB[0]),
        (0.706, 'Ensamblaje y anotación\ngenómica',                        CB[1]),
        (0.610, 'Detección de genes AMR\n(RGI / ResFinder / PointFinder)', CB[2]),
        (0.514, 'Predicción ML\n(GBT · AUC = 0.846)',                      CB[3]),
    ]
    for y, txt, fc in steps_R:
        col_box(xR, y, bw, bh, txt, fc)
    for i in range(len(steps_R) - 1):
        col_arrow(xR, steps_R[i][0] - bh / 2, steps_R[i + 1][0] + bh / 2, '#1D4ED8')

    col_box(xR, 0.412, bw, 0.054,
            'Resultado: misma jornada (Nanopore)$^{*}$',
            CB[4], bold=True)
    col_arrow(xR, steps_R[-1][0] - bh / 2, 0.412 + 0.027, '#1D4ED8')

    ax.add_patch(FancyBboxPatch(
        (xR - 0.20, 0.188), 0.40, 0.152,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor='#EFF6FF', edgecolor='#93C5FD', linewidth=1.5, ls='--', zorder=2))
    ax.annotate('', xy=(xR + 0.185, 0.262), xytext=(xR - 0.185, 0.262),
                arrowprops=dict(arrowstyle='<->', color='#3B82F6', lw=2))
    ax.text(xR, 0.282, 'aprox. 4-8 h (Nanopore)  /  24-48 h (Illumina)',
            ha='center', fontsize=8.5, color='#1D4ED8', fontweight='bold')
    ax.text(xR, 0.248, 'tratamiento dirigido precoz',
            ha='center', fontsize=8, color='#2563EB', style='italic')
    col_arrow(xR, 0.412 - 0.027, 0.340, '#1D4ED8')

    ax.text(
        0.5, 0.148,
        '$^{*}$En condiciones de laboratorio optimizadas, partiendo de colonia aislada. '
        'Nanopore: aprox. 4-8 h (Boolchandani et al., 2019). Illumina: 24-48 h adicionales. '
        'Los tiempos reales en entorno clínico pueden variar.',
        ha='center', fontsize=7.5, color='#6B7280', style='italic')

    plt.tight_layout(pad=0.3)
    out = FIGS / "brecha_diagnostica.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Guardado: {out}")


if __name__ == '__main__':
    print(f"REPO : {REPO}")
    print(f"FIGS : {FIGS}")
    fig1_brecha_diagnostica()
