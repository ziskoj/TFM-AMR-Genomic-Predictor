#!/usr/bin/env python3
"""
02_descarga_amr_fenotipos.py
Descarga genomas de K. pneumoniae con fenotipo AMR validado por laboratorio
Antibiótico objetivo: meropenem (carbapenem de referencia clínica)
"""
import requests
import pandas as pd
from pathlib import Path
import logging
import time

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR = Path('/mnt/f/MIS_DATOS_TFM')
META_DIR = BASE_DIR / 'metadata'
LOG_DIR  = BASE_DIR / 'logs'

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'descarga_amr.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Parámetros ─────────────────────────────────────────────────────────────
API_URL    = "https://www.bv-brc.org/api/genome_amr/"
ANTIBIOTICOS = ["meropenem", "imipenem"]
LIMIT      = 10000  # Pedimos más de 7353 para asegurarnos
CAMPOS_AMR = [
    "genome_id", "genome_name", "antibiotic",
    "resistant_phenotype", "evidence",
    "laboratory_typing_method", "testing_standard"
]

def descargar_fenotipos():
    dfs = []
    for antibiotico in ANTIBIOTICOS:
        log.info(f"Descargando fenotipos: {antibiotico}")
        rql = (
            f"eq(genome_name,Klebsiella%20pneumoniae*)&"
            f"eq(evidence,Laboratory%20Method)&"
            f"eq(antibiotic,{antibiotico})&"
            f"in(resistant_phenotype,(Resistant,Susceptible))&"
            f"select({','.join(CAMPOS_AMR)})&"
            f"limit({LIMIT})"
        )
        headers = {"accept": "text/tsv"}
        resp = requests.get(API_URL, params=rql, headers=headers, timeout=120)
        resp.raise_for_status()

        out = META_DIR / f'amr_fenotipos_{antibiotico}.tsv'
        out.write_text(resp.text)

        df_ab = pd.read_csv(out, sep='\t')
        df_ab['genome_id'] = df_ab['genome_id'].astype(str).str.replace('"', '').str.strip()
        log.info(f"  {antibiotico}: {len(df_ab)} registros")
        log.info(f"  {df_ab['resistant_phenotype'].value_counts().to_dict()}")
        dfs.append(df_ab)
        time.sleep(1)

    # Combinar ambos antibióticos
    df = pd.concat(dfs, ignore_index=True)

    # Si un genoma aparece en ambos, quedarse con el peor fenotipo (Resistant > Susceptible)
    # Esto es conservador y clínicamente más seguro
    df['resistant_phenotype'] = pd.Categorical(
        df['resistant_phenotype'],
        categories=['Susceptible', 'Resistant'],
        ordered=True
    )
    df_dedup = (df.sort_values('resistant_phenotype', ascending=False)
                  .drop_duplicates(subset='genome_id', keep='first')
                  .reset_index(drop=True))

    log.info(f"\nTotal genomas únicos combinados: {len(df_dedup)}")
    log.info(f"Distribución final:\n{df_dedup['resistant_phenotype'].value_counts()}")

    out_clean = META_DIR / 'amr_fenotipos_carbapenems_clean.tsv'
    df_dedup.to_csv(out_clean, sep='\t', index=False)
    log.info(f"Guardado en: {out_clean}")
    return df_dedup

if __name__ == '__main__':
    df = descargar_fenotipos()
    print(f"\nResumen final:")
    print(df['resistant_phenotype'].value_counts())
    print(f"\nGenomas únicos: {len(df)}")
