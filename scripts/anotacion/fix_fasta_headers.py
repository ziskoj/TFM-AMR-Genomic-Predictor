#!/usr/bin/env python3
"""
fix_fasta_headers.py
Limpia cabeceras problemáticas de FASTAs para compatibilidad con RGI
De: >accn|CP021742   descripción larga...
A:  >CP021742
"""
import re
from pathlib import Path

GENOME_DIR   = Path('/mnt/f/MIS_DATOS_TFM/genomes/fasta')
LOG_DIR      = Path('/mnt/f/MIS_DATOS_TFM/logs')
PROBLEMATICOS = LOG_DIR / 'genomas_a_corregir.txt'

def fix_header(fasta_path: Path) -> bool:
    """Limpia cabeceras de un FASTA. Retorna True si hubo cambios."""
    text = fasta_path.read_text()
    new_text = re.sub(r'>accn\|(\S+)\s+.*', r'>\1', text)
    if new_text != text:
        fasta_path.write_text(new_text)
        return True
    return False

def main():
    genome_ids = PROBLEMATICOS.read_text().splitlines()
    genome_ids = [g.strip() for g in genome_ids if g.strip()]
    print(f'Genomas a corregir: {len(genome_ids)}')

    corregidos, no_encontrados = 0, []
    for gid in genome_ids:
        fasta = GENOME_DIR / f'{gid}.fasta'
        if not fasta.exists():
            no_encontrados.append(gid)
            continue
        if fix_header(fasta):
            corregidos += 1
            print(f'  [OK] {gid}')
        else:
            print(f'  [SIN CAMBIOS] {gid}')

    print(f'\nCorregidos:     {corregidos}')
    print(f'No encontrados: {len(no_encontrados)}')

    if no_encontrados:
        (LOG_DIR / 'fastas_no_encontrados.txt').write_text('\n'.join(no_encontrados))

if __name__ == '__main__':
    main()
