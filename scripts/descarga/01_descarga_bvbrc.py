
#!/usr/bin/env python3
"""
01_descarga_bvbrc.py
Descarga metadatos de Klebsiella pneumoniae desde BV-BRC
"""
import requests
import pandas as pd
from pathlib import Path
import logging

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR = Path('/mnt/f/MIS_DATOS_TFM')
META_DIR = BASE_DIR / 'metadata'
LOG_DIR  = BASE_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'descarga.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Parámetros ─────────────────────────────────────────────────────────────
ORGANISMO = "Klebsiella pneumoniae"
MAX_ROWS  = 500  # Empezamos con 500 para probar
API_URL   = "https://www.bv-brc.org/api/genome/"

CAMPOS = [
    "genome_id", "genome_name", "strain", "host_name",
    "isolation_country", "collection_year", "genome_quality",
    "contigs", "genome_length", "gc_content", "genome_status"
]

def descargar_metadatos():
    log.info(f"Descargando metadatos: {ORGANISMO} (max {MAX_ROWS})")
    params = {
        "eq(species,Klebsiella pneumoniae)": "",
        "select(" + ",".join(CAMPOS) + ")": "",
        "limit(" + str(MAX_ROWS) + ")": "",
        "http_accept": "text/tsv"
    }
    headers = {"accept": "text/tsv"}
    rql = f"eq(species,Klebsiella%20pneumoniae)&eq(genome_quality,Good)&in(genome_status,(WGS,Complete))&select({','.join(CAMPOS)})&limit({MAX_ROWS})"
    resp = requests.get(API_URL, params=rql, headers=headers, timeout=120)
    resp.raise_for_status()
    out = META_DIR / 'metadata_kpneumoniae.tsv'
    out.write_text(resp.text)
    df = pd.read_csv(out, sep='\t')
    log.info(f"Descargados {len(df)} genomas. Guardado en {out}")
    return df

if __name__ == '__main__':
    df = descargar_metadatos()
    print(df.head())
    print(f"\nColumnas: {list(df.columns)}")
