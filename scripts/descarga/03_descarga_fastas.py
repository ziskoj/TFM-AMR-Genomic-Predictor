#!/usr/bin/env python3
"""
03_descarga_fastas.py
Descarga los genomas FASTA de los 4432 genomas con fenotipo AMR validado
Implementa checkpointing para poder reanudar si se interrumpe
"""
import requests
import pandas as pd
from pathlib import Path
import logging
import time
from tqdm import tqdm

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path('/mnt/f/MIS_DATOS_TFM')
META_DIR   = BASE_DIR / 'metadata'
GENOME_DIR = BASE_DIR / 'genomes' / 'fasta'
LOG_DIR    = BASE_DIR / 'logs'
GENOME_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'descarga_fastas.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Parámetros ─────────────────────────────────────────────────────────────
API_URL    = "https://www.bv-brc.org/api/genome_sequence/"
PAUSA      = 0.5   # segundos entre descargas
TIMEOUT    = 120   # segundos por request
REINTENTOS = 3

def descargar_fasta(genome_id: str) -> bool:
    """Descarga un FASTA individual con reintentos. Retorna True si éxito."""
    url = f"{API_URL}?eq(genome_id,{genome_id})&http_accept=application/dna+fasta"
    for intento in range(REINTENTOS):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            if resp.text.strip() and resp.text.startswith('>'):
                out = GENOME_DIR / f"{genome_id}.fasta"
                out.write_text(resp.text)
                return True
            else:
                log.warning(f"  FASTA vacío para {genome_id}")
                return False
        except Exception as e:
            log.warning(f"  Intento {intento+1}/{REINTENTOS} fallido {genome_id}: {e}")
            time.sleep(PAUSA * (intento + 1))
    return False

def main():
    # Cargar lista de genomas
    df = pd.read_csv(META_DIR / 'amr_fenotipos_carbapenems_clean.tsv', sep='\t')
    genome_ids = df['genome_id'].astype(str).str.replace('"','').str.strip().unique().tolist()
    log.info(f"Total genomas a descargar: {len(genome_ids)}")

    # Checkpoint: saltar los ya descargados
    ya_descargados = {f.stem for f in GENOME_DIR.glob('*.fasta')}
    pendientes = [g for g in genome_ids if g not in ya_descargados]
    log.info(f"Ya descargados: {len(ya_descargados)} | Pendientes: {len(pendientes)}")

    # Descarga con barra de progreso
    exitosos, fallidos = 0, []
    for genome_id in tqdm(pendientes, desc="Descargando FASTAs"):
        if descargar_fasta(genome_id):
            exitosos += 1
        else:
            fallidos.append(genome_id)
        time.sleep(PAUSA)

        # Log de progreso cada 100 genomas
        if (exitosos + len(fallidos)) % 100 == 0:
            log.info(f"Progreso: {exitosos} OK | {len(fallidos)} fallidos")

    # Guardar lista de fallidos para reintento
    if fallidos:
        out_fail = LOG_DIR / 'fastas_fallidos.txt'
        out_fail.write_text('\n'.join(fallidos))
        log.warning(f"{len(fallidos)} genomas fallidos. Ver: {out_fail}")

    log.info(f"Descarga completada: {exitosos} exitosos | {len(fallidos)} fallidos")

if __name__ == '__main__':
    main()
