#!/usr/bin/env python3
"""
12_consolidar_pointfinder.py
Consolida resultados PointFinder y amplía la matriz combinada
Input:  results/resfinder/*/PointFinder_results.txt
Output: pointfinder_consolidado.tsv + ml_matrix_full.csv.gz
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# ── Rutas ──────────────────────────────────────────────────────────────────
RES_DIR  = Path('/mnt/f/MIS_DATOS_TFM/results/resfinder')
ML_DIR   = Path('/mnt/f/TFM_Linux/ml_matrices')
META_DIR = Path('/mnt/f/MIS_DATOS_TFM/metadata')
LOG_DIR  = Path('/mnt/f/MIS_DATOS_TFM/logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'consolidacion_pointfinder.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

MIN_FREQ = 0.01
MAX_FREQ = 0.99

def consolidar_pointfinder():
    """Consolida todos los resultados PointFinder."""
    log.info('Consolidando PointFinder...')
    archivos = sorted(RES_DIR.glob('*/PointFinder_results.txt'))
    log.info(f'  Archivos encontrados: {len(archivos)}')

    dfs = []
    vacios = 0
    for f in tqdm(archivos, desc='Consolidando PointFinder'):
        genome_id = f.parent.name
        try:
            df = pd.read_csv(f, sep='\t')
            df = df[df['Mutation'].notna()]
            df = df[~df['Mutation'].str.startswith('No')]
            if df.empty:
                vacios += 1
                continue
            df['genome_id'] = genome_id
            dfs.append(df[['genome_id', 'Mutation', 'Amino acid change', 'Resistance']])
        except Exception as e:
            log.warning(f'Error en {genome_id}: {e}')

    if not dfs:
        log.error('No se encontraron resultados válidos')
        return None

    df_all = pd.concat(dfs, ignore_index=True)

    log.info(f'  Genomas con mutaciones: {df_all["genome_id"].nunique()}')
    log.info(f'  Genomas sin mutaciones: {vacios}')
    log.info(f'  Total mutaciones: {len(df_all)}')
    log.info(f'  Mutaciones únicas: {df_all["Mutation"].nunique()}')
    log.info(f'\nTop 15 mutaciones más frecuentes:')
    log.info(df_all['Mutation'].value_counts().head(15).to_string())
    log.info(f'\nResistencias asociadas:')
    log.info(df_all['Resistance'].value_counts().head(10).to_string())

    out = Path('/mnt/f/MIS_DATOS_TFM/results/pointfinder_consolidado.tsv')
    df_all.to_csv(out, sep='\t', index=False)
    log.info(f'\nGuardado: {out}')
    return df_all

def build_matrix_pointfinder(df_pf):
    """Construye matriz binaria de mutaciones PointFinder."""
    log.info('\nConstruyendo matriz PointFinder...')
    df_pf['present'] = 1
    matrix = df_pf.pivot_table(
        index='genome_id', columns='Mutation',
        values='present', aggfunc='max', fill_value=0
    ).astype(np.int8)
    matrix.index = matrix.index.astype(str)
    matrix.columns = ['PF_' + c for c in matrix.columns]
    log.info(f'  Matriz PointFinder inicial: {matrix.shape}')
    return matrix

def fusionar_con_combinada(mat_pf, df_meta):
    """Fusiona PointFinder con la matriz RGI+ResFinder existente."""
    log.info('\nCargando matriz combinada RGI+ResFinder...')
    df_combined = pd.read_csv(ML_DIR / 'ml_matrix_combined.csv.gz', index_col=0)
    df_combined.index = df_combined.index.astype(str)

    # Separar features y target
    target = df_combined['target']
    X_combined = df_combined.drop(columns=['target'])

    log.info(f'  Matriz RGI+ResFinder: {X_combined.shape}')
    log.info(f'  Matriz PointFinder:   {mat_pf.shape}')

    # Fusionar
    df = X_combined.join(mat_pf, how='left').fillna(0)
    df = df.astype(np.int8)

    log.info(f'  Matriz fusionada antes de filtro: {df.shape}')

    # Filtrar features PointFinder por frecuencia
    pf_cols = [c for c in df.columns if c.startswith('PF_')]
    freqs_pf = df[pf_cols].mean()
    mask_pf = (freqs_pf >= MIN_FREQ) & (freqs_pf <= MAX_FREQ)
    cols_eliminar = [c for c in pf_cols if not mask_pf.get(c, True)]
    df = df.drop(columns=cols_eliminar)
    log.info(f'  Features PointFinder eliminadas por frecuencia: {len(cols_eliminar)}')

    # Añadir target
    df['target'] = target
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    # Informe
    rgi_cols = [c for c in df.columns if c.startswith('RGI_')]
    res_cols = [c for c in df.columns if c.startswith('RES_')]
    pf_cols_final = [c for c in df.columns if c.startswith('PF_')]
    log.info(f'\nMatriz final:')
    log.info(f'  Genomas:            {len(df)}')
    log.info(f'  Features RGI:       {len(rgi_cols)}')
    log.info(f'  Features ResFinder: {len(res_cols)}')
    log.info(f'  Features PointFinder: {len(pf_cols_final)}')
    log.info(f'  Total features:     {len(rgi_cols)+len(res_cols)+len(pf_cols_final)}')
    vc = df['target'].value_counts()
    log.info(f'  Balance: Susceptible={vc.get(0,0)} · Resistant={vc.get(1,0)}')

    return df

def main():
    log.info('=== Consolidación PointFinder + Fusión Matriz Completa ===')

    # 1. Consolidar PointFinder
    df_pf = consolidar_pointfinder()
    if df_pf is None:
        return

    # 2. Construir matriz PointFinder
    mat_pf = build_matrix_pointfinder(df_pf)

    # 3. Cargar fenotipos
    df_meta = pd.read_csv(META_DIR / 'dataset_final.tsv', sep='\t')

    # 4. Fusionar con matriz combinada
    df_full = fusionar_con_combinada(mat_pf, df_meta)

    # 5. Guardar
    out = ML_DIR / 'ml_matrix_full.csv.gz'
    df_full.to_csv(out, index=True, compression='gzip')
    log.info(f'\nMatriz completa guardada: {out}')

    feat_cols = [c for c in df_full.columns if c != 'target']
    pd.Series(feat_cols).to_csv(ML_DIR / 'feature_index_full.txt', index=False, header=False)

    print('\n' + '='*55)
    print('RESUMEN MATRIZ COMPLETA RGI + ResFinder + PointFinder')
    print('='*55)
    rgi_cols = [c for c in df_full.columns if c.startswith('RGI_')]
    res_cols = [c for c in df_full.columns if c.startswith('RES_')]
    pf_cols  = [c for c in df_full.columns if c.startswith('PF_')]
    print(f'Genomas:              {len(df_full):,}')
    print(f'Features RGI:         {len(rgi_cols)}')
    print(f'Features ResFinder:   {len(res_cols)}')
    print(f'Features PointFinder: {len(pf_cols)}')
    print(f'Total features:       {len(rgi_cols)+len(res_cols)+len(pf_cols)}')
    vc = df_full['target'].value_counts()
    print(f'Susceptible:          {vc.get(0,0):,}')
    print(f'Resistant:            {vc.get(1,0):,}')
    print('='*55)

if __name__ == '__main__':
    main()
