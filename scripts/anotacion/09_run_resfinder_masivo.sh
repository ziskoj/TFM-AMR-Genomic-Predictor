#!/bin/bash
# 09_run_resfinder_masivo.sh
# Ejecuta ResFinder + PointFinder sobre todos los genomas con checkpointing

GENOME_DIR="/mnt/f/MIS_DATOS_TFM/genomes/fasta"
OUT_DIR="/mnt/f/MIS_DATOS_TFM/results/resfinder"
DB_RES="/mnt/f/MIS_DATOS_TFM/resfinder_db/resfinder_db"
DB_POINT="/mnt/f/MIS_DATOS_TFM/resfinder_db/pointfinder_db"
LOG="/mnt/f/MIS_DATOS_TFM/logs/resfinder_masivo.log"

mkdir -p "$OUT_DIR"
echo "$(date) — Iniciando pipeline ResFinder" | tee -a "$LOG"

TOTAL=$(ls "$GENOME_DIR"/*.fasta | wc -l)
CONTADOR=0
ERRORES=0

for fasta in "$GENOME_DIR"/*.fasta; do
    genome_id=$(basename "$fasta" .fasta)
    out_genome="$OUT_DIR/$genome_id"

    # Checkpoint: saltar si ya procesado
    if [ -d "$out_genome" ] && [ -f "$out_genome/ResFinder_results_tab.txt" ]; then
        continue
    fi

    mkdir -p "$out_genome"

    python /home/zisko/miniforge3/envs/resfinder_env/lib/python3.9/site-packages/resfinder/run_resfinder.py \
        -ifa "$fasta" \
        -o "$out_genome" \
        -db_res "$DB_RES" \
        -db_point "$DB_POINT" \
        -s "Klebsiella pneumoniae" \
        -acq \
        -c \
        -l 0.6 \
        -t 0.9 \
        -l_p 0.6 \
        -t_p 0.9 \
        2>> "$LOG"

    if [ $? -eq 0 ]; then
        CONTADOR=$((CONTADOR + 1))
    else
        ERRORES=$((ERRORES + 1))
        echo "$(date) [ERROR] $genome_id" | tee -a "$LOG"
    fi

    # Progreso cada 100 genomas
    if [ $(( (CONTADOR + ERRORES) % 100 )) -eq 0 ]; then
        echo "$(date) Progreso: $CONTADOR/$TOTAL OK | $ERRORES errores" | tee -a "$LOG"
    fi
done

echo "$(date) — Completado: $CONTADOR OK | $ERRORES errores" | tee -a "$LOG"
