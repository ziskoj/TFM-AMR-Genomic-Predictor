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
Valores obtenidos de los TSVs en `docs/resultados_carbapenems/` y `docs/resultados_multiAB/`.

### Carbapenémicos (n=4.044 · R=39.5%)

| Modelo | AUC | F1 | Sensibilidad | Especificidad |
|--------|-----|----|--------------|---------------|
| **Gradient Boosting** ← principal | **0.846 ± 0.009** | 0.708 | 62.9% | 90.3% |
| XGBoost | 0.845 ± 0.009 | 0.720 | 70.4% | 83.5% |
| LightGBM | 0.844 ± 0.012 | 0.711 | 63.9% | 89.6% |
| Random Forest | 0.839 ± 0.011 | 0.683 | 58.6% | 91.4% |

Top features (GBT): `blaOXA-9` (0.101) · `blaKPC-2` (0.082) · `BRP(MBL)` (0.041)

### Fluoroquinolonas — ciprofloxacino (n=3.495 · R=69.5%)

| Modelo | AUC | F1 | Sensibilidad | Especificidad |
|--------|-----|----|--------------|---------------|
| Gradient Boosting | 0.904 ± 0.013 | 0.880 | 89.6% | 81.3% |
| XGBoost | 0.904 ± 0.014 | 0.878 | 88.8% | 82.4% |
| **LightGBM** ← mejor AUC | **0.905 ± 0.012** | 0.876 | 88.1% | 85.3% |
| Random Forest | 0.897 ± 0.011 | 0.857 | 80.1% | 99.7% |

Top features (GBT): `parC` (0.126) · `gyrA Salmonella isangi` (0.072) · `ompK36 p.N218H` (0.043)

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
│       ├── 16_extension_multiAB.py        # Pipeline FQ + Ceph3G (datos + modelos)
│       ├── 17_figuras_eda.py              # Figuras EDA reproducibles (fig4–fig8)
│       ├── 18_figuras_modelos_carbapenems.py  # Figuras ROC + confusion + feat.imp.
│       └── 19_figuras_multiAB.py          # Figuras comparativas multi-AB (figA–figE)
├── docs/
│   ├── figuras_tfm/                       # fig4–fig8 + ROC + confusion + feat.imp.
│   ├── figuras_multiab/                   # figA–figE comparativas multi-antibiótico
│   ├── resultados_carbapenems/            # TSVs métricas y feature importance (carbapenémicos)
│   │   ├── resultados_baseline.tsv        # RF/GBT/SVM/LR baseline
│   │   ├── resultados_optimizacion.tsv    # GBT/RF optimizados (GridSearchCV)
│   │   ├── resultados_xgb_lgbm.tsv        # XGBoost y LightGBM
│   │   ├── resultados_dl.tsv              # MLP (deep learning)
│   │   ├── metricas_clinicas.tsv          # Sens/Spec/VPP/VPN/LR+ todos los modelos
│   │   ├── feature_importance_GBT.tsv     # Importancia de genes (GBT)
│   │   ├── feature_importance_RF.tsv      # Importancia de genes (RF)
│   │   ├── feature_importance_XGB.tsv     # Importancia de genes (XGBoost)
│   │   ├── feature_importance_LGBM.tsv    # Importancia de genes (LightGBM)
│   │   └── estadisticas_genes_eda.tsv     # Chi² + FDR para 188 genes AMR
│   └── resultados_multiAB/                # TSVs métricas FQ + Ceph3G
│       ├── comparison_all_antibiotics.csv # Tabla comparativa global 3 antibióticos
│       ├── resultados_fluoroquinolones.tsv
│       ├── resultados_cephalosporins3g.tsv
│       ├── metricas_clinicas_fluoroquinolones.tsv
│       ├── metricas_clinicas_cephalosporins3g.tsv
│       ├── feature_importance_fluoroquinolones_GBT.tsv
│       └── feature_importance_cephalosporins3g_GBT.tsv
├── data/
│   └── ml_matrix_binary.csv.gz            # Matriz ML principal (4044×188, binaria)
└── cowork-backup/                         # Entorno reproducible con auditoría MD5
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
            ├── EDA + Estadística (08 → figuras: 17)
            │
            ▼
    ┌──────────────────────────────────┐
    │        MODELOS ML                │
    ├──────────────────────────────────┤
    │ Carbapenémicos  (07 + 11 + 13)  │ → figuras: 18
    │ Deep Learning MLP        (14)   │
    │ Métricas clínicas        (15)   │
    │ Extensión multi-AB       (16)   │ → figuras: 19
    │   └── Fluoroquinolonas          │
    │   └── Cefalosporinas 3G         │
    └──────────────────────────────────┘
```

## Reproducibilidad

### Entorno

```bash
# Entorno ML (scikit-learn, xgboost, lightgbm, tensorflow, pandas, scipy)
source /mnt/f/TFM_Linux/envs/amr_env/bin/activate

# Entorno anotación (RGI + BLAST + DIAMOND)
conda activate rgi_env
```

### Ejecución completa del pipeline

```bash
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
python scripts/ml/07_modelo_baseline.py
python scripts/ml/08_eda.py
python scripts/ml/11_optimizar_modelo.py
python scripts/ml/13_xgboost_lightgbm.py
python scripts/ml/14_deep_learning.py
python scripts/ml/15_metricas_clinicas.py

# Fase 4 — Extensión multi-AB (fluoroquinolonas + cefalosporinas 3G)
python scripts/ml/16_extension_multiAB.py \
    --matrix data/ml_matrix_binary.csv.gz \
    --output_dir docs/resultados_multiAB \
    --antibiotics fluoroquinolones cephalosporins3g
```

### Regenerar figuras (desde resultados pre-computados)

Los scripts 17–19 leen los TSVs de resultados y regeneran todas las figuras
sin necesidad de re-entrenar los modelos.

```bash
source /mnt/f/TFM_Linux/envs/amr_env/bin/activate

# Fig4–Fig8: distribución, histograma, top-30 genes, volcano plot, heatmap
python scripts/ml/17_figuras_eda.py \
    --matrix data/ml_matrix_binary.csv.gz \
    --output_dir docs/figuras_tfm

# ROC, matrices de confusión, feature importance (carbapenémicos)
python scripts/ml/18_figuras_modelos_carbapenems.py \
    --matrix      data/ml_matrix_binary.csv.gz \
    --results_dir docs/resultados_carbapenems \
    --output_dir  docs/figuras_tfm \
    --from-tsv

# FigA–FigE: comparativa multi-antibiótico
python scripts/ml/19_figuras_multiAB.py \
    --results_dir docs/resultados_multiAB \
    --output_dir  docs/figuras_multiab
```

> Para generar las curvas ROC continuas (en lugar de puntos de operación),
> ejecutar el script 18 sin `--from-tsv`. Requiere los modelos `.joblib`
> en `docs/resultados_carbapenems/` y tarda ~10 minutos.

## Entorno técnico

- **SO:** WSL Ubuntu 22.04
- **Python:** 3.10
- **Entorno ML:** `amr_env` (scikit-learn, xgboost, lightgbm, tensorflow, pandas, scipy, statsmodels)
- **Entorno anotación:** Conda `rgi_env` (RGI + BLAST + DIAMOND)

## Autor

**Trabajo Fin de Máster — Bioinformática y Ciencia de Datos**  
Julián Soriano Valero · Repositorio privado · Licencia MIT
