#!/usr/bin/env python3
"""
10_consolidar_resfinder.py
Consolida resultados ResFinder y construye matriz ampliada RGI + ResFinder
Input:  results/resfinder/*/ResFinder_results_tab.txt + rgi_consolidado.tsv
Output: resfinder_consolidado.tsv + ml_matrix_combined.csv.gz
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# ── Rutas ──────────────────────────────────────────────────────────────────
RES_DIR  = Path('/mnt/f/MIS_DATOS_TFM/results/resfinder')
RGI_TSV  = Path('/mnt/f/MIS_DATOS_TFM/results/rgi_consolidado.tsv')
META_DIR = Path('/mnt/f/MIS_DATOS_TFM/metadata')
ML_DIR   = Path('/mnt/f/TFM_Linux/ml_matrices')
LOG_DIR  = Path('/mnt/f/MIS_DATOS_TFM/logs')

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'consolidacion_resfinder.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

MIN_FREQ = 0.01
MAX_FREQ = 0.99

def consolidar_resfinder():
    """Consolida todos los resultados ResFinder en una tabla única."""
    log.info('Consolidando resultados ResFinder...')
    archivos = sorted(RES_DIR.glob('*/ResFinder_results_tab.txt'))
    log.info(f'  Archivos encontrados: {len(archivos)}')

    dfs = []
    vacios = 0
    for f in tqdm(archivos, desc='Consolidando ResFinder'):
        genome_id = f.parent.name
        try:
            df = pd.read_csv(f, sep='\t')
            # Filtrar filas con gen detectado
            df = df[df['Resistance gene'].notna()]
            df = df[~df['Resistance gene'].str.startswith('No')]
            if df.empty:
                vacios += 1
                continue
            df['genome_id'] = genome_id
            df = df.rename(columns={'Resistance gene': 'gene'})
            dfs.append(df[['genome_id', 'gene', 'Identity', 'Coverage', 'Phenotype']])
        except Exception as e:
            log.warning(f'Error en {genome_id}: {e}')

    if not dfs:
        log.error('No se encontraron resultados válidos')
        return None

    df_all = pd.concat(dfs, ignore_index=True)
    log.info(f'  Genomas con genes detectados: {df_all["genome_id"].nunique()}')
    log.info(f'  Genomas sin hits: {vacios}')
    log.info(f'  Total filas: {len(df_all)}')
    log.info(f'  Genes únicos: {df_all["gene"].nunique()}')
    log.info(f'\nTop 10 genes más frecuentes:')
    log.info(df_all['gene'].value_counts().head(10).to_string())

    out = Path('/mnt/f/MIS_DATOS_TFM/results/resfinder_consolidado.tsv')
    df_all.to_csv(out, sep='\t', index=False)
    log.info(f'\nGuardado: {out}')
    return df_all

def build_matrix_resfinder(df_res):
    """Construye matriz binaria de ResFinder."""
    log.info('\nConstruyendo matriz ResFinder...')
    df_res['present'] = 1
    matrix = df_res.pivot_table(
        index='genome_id', columns='gene',
        values='present', aggfunc='max', fill_value=0
    ).astype(np.int8)
    # Añadir prefijo para distinguir de genes RGI
    matrix.columns = ['RES_' + c for c in matrix.columns]
    matrix.index = matrix.index.astype(str)
    log.info(f'  Matriz ResFinder: {matrix.shape}')
    return matrix

def build_matrix_rgi():
    """Reconstruye matriz binaria de RGI."""
    log.info('\nCargando matriz RGI...')
    df_rgi = pd.read_csv(RGI_TSV, sep='\t')
    df_rgi['present'] = 1
    matrix = df_rgi.pivot_table(
        index='genome_id', columns='Best_Hit_ARO',
        values='present', aggfunc='max', fill_value=0
    ).astype(np.int8)
    matrix.columns = ['RGI_' + c for c in matrix.columns]
    matrix.index = matrix.index.astype(str)
    log.info(f'  Matriz RGI: {matrix.shape}')
    return matrix

def fusionar_matrices(mat_rgi, mat_res, df_meta):
    """Fusiona ambas matrices y añade target."""
    log.info('\nFusionando matrices RGI + ResFinder...')

    # Unir por genome_id (outer join — conservar todos)
    mat_rgi.index = mat_rgi.index.astype(str)
    mat_res.index = mat_res.index.astype(str)
    df = mat_rgi.join(mat_res, how='outer').fillna(0).astype(np.int8)
    log.info(f'  Matriz combinada antes de filtro: {df.shape}')

    # Filtrar por frecuencia
    freqs = df.mean()
    mask = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)
    df = df.loc[:, mask]
    log.info(f'  Features eliminadas por frecuencia: {(~mask).sum()}')
    log.info(f'  Matriz combinada filtrada: {df.shape}')

    # Añadir target
    df_meta['genome_id'] = df_meta['genome_id'].astype(str).str.strip()
    phenotypes = df_meta.set_index('genome_id')['resistant_phenotype']
    df['target'] = df.index.map(phenotypes)
    df['target'] = df['target'].map({'Resistant': 1, 'Susceptible': 0})
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    log.info(f'\nMatriz final: {df.shape[0]} genomas × {df.shape[1]-1} features + target')
    vc = df['target'].value_counts()
    log.info(f'Balance: Susceptible={vc.get(0,0)} · Resistant={vc.get(1,0)} · Ratio={vc.max()/vc.min():.2f}:1')

    # Informe de features por fuente
    rgi_cols = [c for c in df.columns if c.startswith('RGI_')]
    res_cols = [c for c in df.columns if c.startswith('RES_')]
    log.info(f'\nFeatures por fuente:')
    log.info(f'  RGI:       {len(rgi_cols)} genes')
    log.info(f'  ResFinder: {len(res_cols)} genes')
    log.info(f'  Total:     {len(rgi_cols)+len(res_cols)} features')

    return df

def main():
    log.info('=== Consolidación ResFinder + Fusión de Matrices ===')

    # 1. Consolidar ResFinder
    df_res = consolidar_resfinder()
    if df_res is None:
        return

    # 2. Construir matrices individuales
    mat_res = build_matrix_resfinder(df_res)
    mat_rgi = build_matrix_rgi()

    # 3. Cargar fenotipos
    df_meta = pd.read_csv(META_DIR / 'dataset_final.tsv', sep='\t')

    # 4. Fusionar
    df_combined = fusionar_matrices(mat_rgi, mat_res, df_meta)

    # 5. Guardar
    out = ML_DIR / 'ml_matrix_combined.csv.gz'
    df_combined.to_csv(out, index=True, compression='gzip')
    log.info(f'\nMatriz combinada guardada: {out}')

    # Guardar índice de features
    feat_cols = [c for c in df_combined.columns if c != 'target']
    pd.Series(feat_cols).to_csv(ML_DIR / 'feature_index_combined.txt', index=False, header=False)

    print('\n' + '='*55)
    print('RESUMEN MATRIZ COMBINADA RGI + ResFinder')
    print('='*55)
    print(f'Genomas:          {len(df_combined):,}')
    rgi_cols = [c for c in df_combined.columns if c.startswith('RGI_')]
    res_cols = [c for c in df_combined.columns if c.startswith('RES_')]
    print(f'Features RGI:     {len(rgi_cols)}')
    print(f'Features ResFinder: {len(res_cols)}')
    print(f'Total features:   {len(rgi_cols)+len(res_cols)}')
    vc = df_combined['target'].value_counts()
    print(f'Susceptible:      {vc.get(0,0):,}')
    print(f'Resistant:        {vc.get(1,0):,}')
    print('='*55)

if __name__ == '__main__':
    main()
