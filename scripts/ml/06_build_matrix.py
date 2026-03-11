#!/usr/bin/env python3
"""
06_build_matrix.py
Genera la matriz binaria de presencia/ausencia de genes para ML
Input:  rgi_consolidado.tsv + dataset_final.tsv
Output: ml_matrix_binary.csv.gz
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# ── Rutas ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path('/mnt/f/MIS_DATOS_TFM/results')
META_DIR    = Path('/mnt/f/MIS_DATOS_TFM/metadata')
ML_DIR      = Path('/mnt/f/TFM_Linux/ml_matrices')
LOG_DIR     = Path('/mnt/f/MIS_DATOS_TFM/logs')
ML_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'build_matrix.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Parámetros ─────────────────────────────────────────────────────────────
MIN_FREQ = 0.01  # Eliminar genes presentes en menos del 1% de genomas
MAX_FREQ = 0.99  # Eliminar genes presentes en más del 99% de genomas

def load_data():
    """Carga resultados RGI y fenotipos."""
    log.info('Cargando resultados RGI consolidados...')
    df_rgi = pd.read_csv(RESULTS_DIR / 'rgi_consolidado.tsv', sep='\t')
    df_rgi['genome_id'] = df_rgi['genome_id'].astype(str).str.strip()
    log.info(f'  RGI: {len(df_rgi)} filas · {df_rgi["genome_id"].nunique()} genomas')

    log.info('Cargando fenotipos...')
    df_meta = pd.read_csv(META_DIR / 'dataset_final.tsv', sep='\t')
    df_meta['genome_id'] = df_meta['genome_id'].astype(str).str.replace('"','').str.strip()
    df_meta['target'] = (df_meta['resistant_phenotype'] == 'Resistant').astype(int)
    log.info(f'  Fenotipos: {len(df_meta)} genomas')
    log.info(f'  Balance: {df_meta["target"].value_counts().to_dict()}')
    return df_rgi, df_meta

def build_matrix(df_rgi: pd.DataFrame) -> pd.DataFrame:
    """Construye matriz binaria presencia/ausencia de genes."""
    log.info('Construyendo matriz binaria...')
    df_rgi['present'] = 1
    matrix = df_rgi.pivot_table(
        index='genome_id',
        columns='Best_Hit_ARO',
        values='present',
        aggfunc='max',
        fill_value=0
    ).astype(np.int8)
    log.info(f'  Matriz inicial: {matrix.shape[0]} genomas × {matrix.shape[1]} genes')
    return matrix

def filter_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Elimina genes de frecuencia muy baja o muy alta."""
    freqs = matrix.mean()
    mask = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)
    matrix_filtered = matrix.loc[:, mask]
    n_removed = (~mask).sum()
    log.info(f'  Genes eliminados por frecuencia: {n_removed}')
    log.info(f'  Matriz filtrada: {matrix_filtered.shape[0]} × {matrix_filtered.shape[1]}')
    return matrix_filtered

def add_target(matrix: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """Une la matriz con los fenotipos AMR."""
    phenotypes = df_meta.set_index('genome_id')['target']
    df = matrix.copy()
    df['target'] = df.index.map(phenotypes)
    n_missing = df['target'].isna().sum()
    if n_missing > 0:
        log.warning(f'  {n_missing} genomas sin fenotipo — se excluyen')
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)
    log.info(f'  Matriz final: {df.shape[0]} genomas × {df.shape[1]-1} genes + target')
    log.info(f'  Balance final: {df["target"].value_counts().to_dict()}')
    return df

def validate(df: pd.DataFrame):
    """Informe de calidad de la matriz."""
    gene_cols = [c for c in df.columns if c != 'target']
    print('\n' + '='*55)
    print('INFORME DE CALIDAD — MATRIZ ML')
    print('='*55)
    print(f'Genomas (filas):       {len(df):,}')
    print(f'Genes AMR (columnas):  {len(gene_cols):,}')
    print(f'Valores nulos:         {df.isna().sum().sum()}')
    print(f'Densidad matriz:       {df[gene_cols].values.mean():.2%}')
    vc = df['target'].value_counts()
    print(f'\nClase 0 (Susceptible): {vc.get(0,0):,} ({vc.get(0,0)/len(df):.1%})')
    print(f'Clase 1 (Resistant):   {vc.get(1,0):,} ({vc.get(1,0)/len(df):.1%})')
    ratio = vc.max() / vc.min()
    print(f'Ratio desbalance:      {ratio:.2f}:1', end='')
    print(' ✔' if ratio < 3 else ' ⚠️  Considerar balanceo')
    print('='*55)

def main():
    df_rgi, df_meta = load_data()
    matrix = build_matrix(df_rgi)
    matrix = filter_matrix(matrix)
    df_ml = add_target(matrix, df_meta)
    validate(df_ml)

    # Guardar matriz comprimida
    out = ML_DIR / 'ml_matrix_binary.csv.gz'
    df_ml.to_csv(out, index=True, compression='gzip')
    log.info(f'\nMatriz guardada: {out}')

    # Guardar índice de genes para trazabilidad
    gene_cols = [c for c in df_ml.columns if c != 'target']
    pd.Series(gene_cols).to_csv(ML_DIR / 'gene_index.txt', index=False, header=False)
    log.info(f'Índice de genes guardado: {ML_DIR}/gene_index.txt')
    return df_ml

if __name__ == '__main__':
    df_ml = main()
