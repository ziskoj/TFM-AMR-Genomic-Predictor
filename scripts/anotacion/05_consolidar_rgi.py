#!/usr/bin/env python3
"""
05_consolidar_rgi.py
Consolida todos los resultados individuales de RGI en una tabla única
Input:  /mnt/f/MIS_DATOS_TFM/results/rgi/*.txt
Output: /mnt/f/MIS_DATOS_TFM/results/rgi_consolidado.tsv
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

# ── Rutas ──────────────────────────────────────────────────────────────────
RGI_DIR  = Path('/mnt/f/MIS_DATOS_TFM/results/rgi')
OUT_DIR  = Path('/mnt/f/MIS_DATOS_TFM/results')
LOG_DIR  = Path('/mnt/f/MIS_DATOS_TFM/logs')

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'consolidacion.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Columnas relevantes del output de RGI
COLS = [
    'Best_Hit_ARO', 'ARO', 'Drug Class',
    'Resistance Mechanism', 'AMR Gene Family',
    'Best_Identities', 'Cut_Off'
]

def parse_rgi_file(txt_path: Path) -> pd.DataFrame:
    """Parsea un archivo RGI y retorna genes detectados con filtro Strict/Perfect."""
    try:
        df = pd.read_csv(txt_path, sep='\t')
        if df.empty:
            return pd.DataFrame()
        # Filtrar solo hits de alta confianza
        df = df[df['Cut_Off'].isin(['Strict', 'Perfect'])]
        if df.empty:
            return pd.DataFrame()
        df['genome_id'] = txt_path.stem
        cols_disponibles = ['genome_id'] + [c for c in COLS if c in df.columns]
        return df[cols_disponibles]
    except Exception as e:
        log.warning(f'Error parseando {txt_path.name}: {e}')
        return pd.DataFrame()

def consolidar():
    rgi_files = sorted(RGI_DIR.glob('*.txt'))
    # Excluir archivos de test
    rgi_files = [f for f in rgi_files if 'test' not in f.stem and 'tiempo' not in f.stem]
    log.info(f'Archivos RGI a consolidar: {len(rgi_files)}')

    dfs = []
    vacios = 0
    for f in tqdm(rgi_files, desc='Consolidando'):
        df = parse_rgi_file(f)
        if df.empty:
            vacios += 1
        else:
            dfs.append(df)

    if not dfs:
        log.error('No se encontraron resultados válidos')
        return None

    df_all = pd.concat(dfs, ignore_index=True)

    log.info(f'Genomas con genes detectados: {df_all["genome_id"].nunique()}')
    log.info(f'Genomas sin hits Strict/Perfect: {vacios}')
    log.info(f'Total filas: {len(df_all)}')
    log.info(f'Genes ARO únicos: {df_all["Best_Hit_ARO"].nunique()}')
    log.info(f'\nTop 10 genes más frecuentes:')
    log.info(df_all['Best_Hit_ARO'].value_counts().head(10).to_string())

    out = OUT_DIR / 'rgi_consolidado.tsv'
    df_all.to_csv(out, sep='\t', index=False)
    log.info(f'Guardado en: {out}')
    return df_all

if __name__ == '__main__':
    df = consolidar()
    if df is not None:
        print(f'\nResumen:')
        print(f'  Genomas únicos: {df["genome_id"].nunique()}')
        print(f'  Genes únicos:   {df["Best_Hit_ARO"].nunique()}')
        print(f'\nTop 5 genes:')
        print(df['Best_Hit_ARO'].value_counts().head())
        print(f'\nTop 5 clases de antibiótico:')
        print(df['Drug Class'].value_counts().head())
