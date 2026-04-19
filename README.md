# TFM-AMR-Genomic-Predictor

Pipeline bioinformático para la predicción de resistencia antimicrobiana (AMR)
en *Klebsiella pneumoniae* mediante Machine Learning sobre datos genómicos.

## Descripción

Este proyecto desarrolla modelos de Machine Learning capaces de predecir la resistencia
a tres grupos de antibióticos clínicamente relevantes — carbapenémicos, fluoroquinolonas
y cefalosporinas de 3ª generación — a partir de la presencia/ausencia de genes de
resistencia identificados en genomas bacterianos completos.

**Organismo:** *Klebsiella pneumoniae*  
**Antibióticos:** Carbapenémicos (meropenem/imipenem) · Fluoroquinolonas (ciprofloxacino) · Cefalosporinas 3G (ceftriaxona)  
**Fuente de datos:** BV-BRC (Bacterial and Viral Bioinformatics Resource Center)  
**Anotación:** RGI + CARD v4.0.1 · ResFinder · PointFinder  
**Features:** 188 (82 RGI_ · 70 RES_ · 36 PF_)

## Resultados

Evaluación mediante 5-fold StratifiedKFold con hiperparámetros optimizados (GridSearchCV).

### Carbapenémicos (n=4.044 · R=39.5%)

| Modelo | AUC | F1 | Sensibilidad | Especificidad |
|--------|-----|----|--------------|---------------|
| **Gradient Boosting** ← principal | **0.846 ± 0.009** | 0.708 | 62.9% | 90.3% |
| XGBoost | 0.845 ± 0.009 | 0.720 | 70.4% | 83.5% |
| LightGBM | 0.844 ± 0.012 | 0.711 | 63.9% | 89.6% |
| Random Forest | 0.839 ± 0.011 | 0.683 | 58.6% | 91.4% |

Top features: `blaOXA-9` (0.101) · `blaKPC-2` (0.082) · `BRP(MBL)` (0.041)

### Fluoroquinolonas — ciprofloxacino (n=3.495 · R=69.5%)

| Modelo | AUC | F1 | Sensibilidad | Especificidad |
|--------|-----|----|--------------|---------------|
| Gradient Boosting | 0.904 ± 0.013 | 0.880 | 89.6% | 81.3% |
| XGBoost | 0.904 ± 0.014 | 0.878 | 88.8% | 82.4% |
| **LightGBM** ← mejor AUC | **0.905 ± 0.012** | 0.876 | 88.1% | 85.3% |
| Random Forest | 0.897 ± 0.011 | 0.857 | 80.1% | 99.7% |

Top features: `parC` (0.127) · `gyrA Salmonella isangi` (0.072) · `ompK36` (0.043)

### Cefalosporinas 3G — ceftriaxona (n=3.763 · R=78.6%)

| Modelo | AUC | F1 | Sensibilidad | Especificidad |
|--------|-----|----|--------------|---------------|
| Gradient Boosting | 0.902 ± 0.018 | 0.911 | 94.5% | 72.4% |
| XGBoost | 0.903 ± 0.020 | 0.910 | 93.4% | 71.0% |
| **LightGBM** ← mejor AUC | **0.904 ± 0.016** | 0.909 | 92.9% | 79.1% |
| Random Forest | 0.896 ± 0.015 | 0.864 | 78.4% | 100% |

Todos los modelos superan el objetivo AUC ≥ 0.80 en los tres grupos de antibióticos.

## Dataset

| Parámetro | Valor |
|-----------|-------|
| Genomas descargados (FASTA) | 4.510 |
| Dataset carbapenémicos (fenotipos validados) | 4.044 |
| Dataset fluoroquinolonas | 3.495 |
| Dataset cefalosporinas 3G | 3.763 |
| Fuente de fenotipos | Laboratorio experimental (BV-BRC) |
| Features totales | 188 (binarias: presencia/ausencia gen) |

## Estructura del repositorio

```
TFM-AMR-Genomic-Predictor/
├── scripts/
│   ├── descarga/
│   │   ├── 01_descarga_bvbrc.py           # Descarga metadatos K. pneumoniae
│   │   ├── 02_descarga_amr_fenotipos.py   # Fenotipos AMR validados laboratorio
│   │   └── 03_descarga_fastas.py          # Descarga masiva de genomas FASTA
│   ├── anotacion/
│   │   ├── 04_run_rgi_masivo.sh           # Pipeline anotación RGI masivo (CARD)
│   │   ├── 05_consolidar_rgi.py           # Consolida resultados RGI → matriz
│   │   ├── 09_run_resfinder_masivo.sh     # Pipeline ResFinder + PointFinder
│   │   ├── 10_consolidar_resfinder.py     # Consolida resultados ResFinder
│   │   ├── 12_consolidar_pointfinder.py   # Consolida resultados PointFinder
│   │   └── fix_fasta_headers.py           # Corrección cabeceras problemáticas
│   └── ml/
│       ├── 06_build_matrix.py             # Matriz binaria presencia/ausencia
│       ├── 07_modelo_baseline.py          # Modelos baseline RF/GBT/SVM/LR
│       ├── 08_eda.py                      # Análisis exploratorio y estadístico
│       ├── 11_optimizar_modelo.py         # Optimización hiperparámetros GBT
│       ├── 13_xgboost_lightgbm.py         # XGBoost y LightGBM
│       ├── 14_deep_learning.py            # MLP (red neuronal)
│       ├── 15_metricas_clinicas.py        # Métricas clínicas (sens/spec/VPP/VPN)
│       └── 16_extension_multiAB.py        # Extensión FQ + Ceph3G (pipeline completo)
├── docs/
│   ├── figures/
│   │   ├── estudio_multiAB/               # Figuras A-E comparativas multi-AB
│   │   └── ...                            # Figuras EDA + ROC + feature importance
│   └── resultados_multiAB/                # TSVs métricas FQ + Ceph3G
├── data/                                  # Metadatos (sin FASTAs por tamaño)
├── notebooks/                             # Jupyter notebooks de análisis
└── cowork-backup/                         # Entorno reproducible con auditoría
```

## Pipeline

```
BV-BRC API
    │
    ├── Metadatos K. pneumoniae (01)
    ├── Fenotipos AMR laboratorio (02)
    └── Genomas FASTA ×4510 (03)
            │
            ▼
    ┌─────────────────────────────┐
    │      ANOTACIÓN MULTI-BASE   │
    ├─────────────────────────────┤
    │ RGI + CARD v4.0.1  (04-05) │  → 82 features RGI_
    │ ResFinder          (09-10) │  → 70 features RES_
    │ PointFinder        (09+12) │  → 36 features PF_
    └─────────────────────────────┘
            │
            ▼
    Matriz binaria 188 features (06)
            │
            ├── EDA + Estadística (08)
            │
            ▼
    ┌──────────────────────────────────┐
    │        MODELOS ML                │
    ├──────────────────────────────────┤
    │ Carbapenémicos  (07 + 11 + 13)  │
    │ Deep Learning MLP        (14)   │
    │ Métricas clínicas        (15)   │
    │ Extensión multi-AB       (16)   │
    │   └── Fluoroquinolonas          │
    │   └── Cefalosporinas 3G         │
    └──────────────────────────────────┘
```

## Entorno

- **SO:** WSL Ubuntu 22.04
- **Python:** 3.10
- **Dependencias:** ver `requirements.txt`
- **Entorno ML:** `amr_env` (scikit-learn, xgboost, lightgbm, tensorflow, pandas)
- **Entorno anotación:** Conda `rgi_env` (RGI + BLAST + DIAMOND)

## Reproducibilidad

```bash
# Activar entorno Python
source /mnt/f/TFM_Linux/envs/amr_env/bin/activate

# Activar entorno anotación
conda activate rgi_env

# Fase 1 — Descarga
python scripts/descarga/01_descarga_bvbrc.py
python scripts/descarga/02_descarga_amr_fenotipos.py
python scripts/descarga/03_descarga_fastas.py

# Fase 2 — Anotación (requiere rgi_env + ResFinder instalado)
bash scripts/anotacion/04_run_rgi_masivo.sh
python scripts/anotacion/05_consolidar_rgi.py
bash scripts/anotacion/09_run_resfinder_masivo.sh
python scripts/anotacion/10_consolidar_resfinder.py
python scripts/anotacion/12_consolidar_pointfinder.py

# Fase 3 — ML carbapenémicos
python scripts/ml/06_build_matrix.py
python scripts/ml/08_eda.py
python scripts/ml/07_modelo_baseline.py
python scripts/ml/11_optimizar_modelo.py
python scripts/ml/13_xgboost_lightgbm.py
python scripts/ml/14_deep_learning.py
python scripts/ml/15_metricas_clinicas.py

# Fase 4 — Extensión multi-AB (fluoroquinolonas + cefalosporinas 3G)
python scripts/ml/16_extension_multiAB.py
```

## Autor

**Trabajo Fin de Máster — Bioinformática y Ciencia de Datos**  
Repositorio privado · Licencia MIT
