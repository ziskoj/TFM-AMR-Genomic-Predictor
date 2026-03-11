# TFM-AMR-Genomic-Predictor

Pipeline bioinformático para la predicción de resistencia antimicrobiana (AMR)
en *Klebsiella pneumoniae* mediante Machine Learning sobre datos genómicos.

## Descripción

Este proyecto desarrolla un modelo de Machine Learning capaz de predecir
la resistencia a carbapenémicos (meropenem e imipenem) a partir de la
presencia/ausencia de genes de resistencia identificados en genomas bacterianos.

**Organismo:** *Klebsiella pneumoniae*  
**Antibióticos objetivo:** Meropenem, Imipenem (carbapenémicos)  
**Fuente de datos:** BV-BRC (Bacterial and Viral Bioinformatics Resource Center)  
**Herramienta de anotación:** RGI (Resistance Gene Identifier) + CARD v4.0.1  

## Dataset

| Parámetro | Valor |
|-----------|-------|
| Genomas totales | 4.125 |
| Fenotipos validados por laboratorio | 4.125 |
| Susceptibles | 2.514 (61%) |
| Resistentes | 1.611 (39%) |
| Ratio desbalance | 1.6:1 |

## Estructura del repositorio
```
TFM-AMR-Genomic-Predictor/
├── scripts/
│   ├── descarga/
│   │   ├── 01_descarga_bvbrc.py         # Descarga metadatos K. pneumoniae
│   │   ├── 02_descarga_amr_fenotipos.py # Fenotipos AMR validados laboratorio
│   │   └── 03_descarga_fastas.py        # Descarga masiva de genomas FASTA
│   ├── anotacion/
│   │   ├── 04_run_rgi_masivo.sh         # Pipeline anotación RGI masivo
│   │   ├── 05_consolidar_rgi.py         # Consolida resultados RGI
│   │   └── fix_fasta_headers.py         # Corrección cabeceras problemáticas
│   └── ml/
│       ├── 06_build_matrix.py           # Matriz binaria presencia/ausencia
│       ├── 07_modelo_baseline.py        # Modelos baseline RF/GBT/SVM/LR
│       └── 08_eda.py                    # Análisis exploratorio y estadístico
├── data/                                # Metadatos (sin FASTAs)
├── notebooks/                           # Jupyter notebooks de análisis
└── docs/                                # Documentación técnica por sesión
```

## Pipeline
```
BV-BRC API
    │
    ├── Metadatos K. pneumoniae (01)
    ├── Fenotipos AMR laboratorio (02)
    └── Genomas FASTA x4125 (03)
            │
            ▼
    Anotación RGI + CARD (04)
            │
            ▼
    Consolidación resultados (05)
            │
            ▼
    Matriz binaria Gen×Genoma (06)
            │
            ├── EDA + Estadística (08)
            │
            ▼
    Modelos ML baseline (07)
    Random Forest · GBT · SVM · LR
```

## Entorno

- **SO:** WSL Ubuntu 22.04
- **Python:** 3.10
- **Dependencias:** ver `requirements.txt`
- **Anotación:** Conda `rgi_env` (RGI + BLAST + DIAMOND)

## Reproducibilidad
```bash
# Activar entorno Python
source /mnt/f/TFM_Linux/envs/amr_env/bin/activate

# Activar entorno anotación
conda activate rgi_env

# Ejecutar pipeline completo en orden numérico
python scripts/descarga/01_descarga_bvbrc.py
python scripts/descarga/02_descarga_amr_fenotipos.py
python scripts/descarga/03_descarga_fastas.py
bash scripts/anotacion/04_run_rgi_masivo.sh
python scripts/anotacion/05_consolidar_rgi.py
python scripts/ml/06_build_matrix.py
python scripts/ml/08_eda.py
python scripts/ml/07_modelo_baseline.py
```

## Autor

**Trabajo Fin de Máster — Bioinformática y Ciencia de Datos**  
Repositorio privado · Licencia MIT
