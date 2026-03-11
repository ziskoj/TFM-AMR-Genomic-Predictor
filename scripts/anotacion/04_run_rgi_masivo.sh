#!/bin/bash
# 04_run_rgi_masivo.sh
# Ejecuta RGI sobre todos los genomas con checkpointing

GENOME_DIR="/mnt/f/MIS_DATOS_TFM/genomes/fasta"
RGI_OUT="/mnt/f/MIS_DATOS_TFM/results/rgi"
LOG="/mnt/f/MIS_DATOS_TFM/logs/rgi_masivo.log"

mkdir -p "$RGI_OUT"
echo "$(date) — Iniciando pipeline RGI" | tee -a "$LOG"

TOTAL=$(ls "$GENOME_DIR"/*.fasta | wc -l)
CONTADOR=0
ERRORES=0

for fasta in "$GENOME_DIR"/*.fasta; do
    genome_id=$(basename "$fasta" .fasta)
    out_prefix="$RGI_OUT/$genome_id"

    # Checkpoint: saltar si ya procesado
    if [ -f "${out_prefix}.txt" ]; then
        continue
    fi

    rgi main \
        --input_sequence "$fasta" \
        --output_file "$out_prefix" \
        --input_type contig \
        --alignment_tool BLAST \
        --num_threads 4 \
        --clean \
        --local 2>> "$LOG"

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
