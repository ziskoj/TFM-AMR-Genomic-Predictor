# CLAUDE.md — Contexto del proyecto TFM AMR
> Archivo de contexto para Claude. Léelo al inicio de cada sesión antes de cualquier tarea.
> Última actualización: 2026-04-23

---

## 🎯 Proyecto

**TFM:** Predicción de resistencia antimicrobiana (AMR) en *Klebsiella pneumoniae* mediante Machine Learning sobre datos genómicos.

**Autor:** Julián Soriano Valero (`zisko.juli@gmail.com`)  
**Máster:** Bioinformática y Ciencia de Datos  
**Estado actual:** Sustancialmente completo. Pendiente revisión tutor y entrega final.

---

## 🗺️ Mapa de rutas — dónde está cada cosa

### Cowork workspace (este entorno)
| Referencia | Ruta Cowork | Ruta WSL | Ruta Windows |
|---|---|---|---|
| Workspace TFM | `/sessions/.../mnt/TFM/` | `/mnt/c/Users/julia/Documents/Claude/Projects/TFM/` | `C:\Users\julia\Documents\Claude\Projects\TFM\` |

### SSD externo (datos académicos, READONLY)
| Contenido | Ruta WSL |
|---|---|
| **Datos principales** (13 GB, `dr-xr-xr-x`) | `/mnt/f/MIS_DATOS_TFM/` |
| ↳ Genomas FASTA (4.510, 7.1 GB) | `/mnt/f/MIS_DATOS_TFM/genomes/fasta/` |
| ↳ Metadata TSV (~4k genomas) | `/mnt/f/MIS_DATOS_TFM/metadata/` |
| ↳ CARD database | `/mnt/f/MIS_DATOS_TFM/card.json` |
| ↳ ResFinder + PointFinder DB | `/mnt/f/MIS_DATOS_TFM/resfinder_db/` |
| ↳ DIAMOND databases | `/mnt/f/MIS_DATOS_TFM/localDB/` |
| ↳ Resultados previos | `/mnt/f/MIS_DATOS_TFM/results/` |
| ↳ Checksums originales (backup) | `/mnt/f/MIS_DATOS_TFM/CHECKSUMS_ORIGINAL.md5` |

### Repositorio Git
| Contenido | Ruta WSL | Ruta Windows |
|---|---|---|
| **Repo principal** | `/mnt/f/TFM-AMR-Genomic-Predictor/` | `F:\TFM-AMR-Genomic-Predictor\` |
| ↳ Scripts originales | `/mnt/f/TFM-AMR-Genomic-Predictor/scripts/` | — |
| ↳ Docs + figuras | `/mnt/f/TFM-AMR-Genomic-Predictor/docs/` | — |
| ↳ Figuras multi-AB | `/mnt/f/TFM-AMR-Genomic-Predictor/docs/figures/estudio_multiAB/` | — |
| ↳ Resultados multi-AB | `/mnt/f/TFM-AMR-Genomic-Predictor/docs/resultados_multiAB/` | — |
| ↳ Cowork backup | `/mnt/f/TFM-AMR-Genomic-Predictor/cowork-backup/` | — |

### Cowork backup (scripts editables + auditoría)
| Contenido | Ruta WSL |
|---|---|
| **Backup scripts** (19 scripts, editable) | `/mnt/f/TFM-AMR-Genomic-Predictor/cowork-backup/scripts/` |
| ↳ Anotación | `/mnt/f/TFM-AMR-Genomic-Predictor/cowork-backup/scripts/anotacion/` |
| ↳ Descarga | `/mnt/f/TFM-AMR-Genomic-Predictor/cowork-backup/scripts/descarga/` |
| ↳ ML | `/mnt/f/TFM-AMR-Genomic-Predictor/cowork-backup/scripts/ml/` |
| Checksums MD5 (157.715 hashes) | `/mnt/f/TFM-AMR-Genomic-Predictor/cowork-backup/MIS_DATOS_TFM_CHECKSUMS.md5` |
| Script auditoría PRE-sesión | `/mnt/f/TFM-AMR-Genomic-Predictor/cowork-backup/audit_before_session.sh` |
| Script auditoría POST-sesión | `/mnt/f/TFM-AMR-Genomic-Predictor/cowork-backup/audit_after_session.sh` |
| Init sesión (verifica todo) | `/mnt/f/TFM-AMR-Genomic-Predictor/cowork-backup/init_cowork_session.sh` |
| Config rutas datos | `/mnt/f/TFM-AMR-Genomic-Predictor/cowork-backup/DATA_PATHS.yaml` |

---

## 📁 Contenido del workspace Cowork (archivos clave)

### Documento TFM
| Archivo | Descripción |
|---|---|
| `TFM_discusion_completa.docx` | ✅ **VERSIÓN FINAL** (3.83 MB, 19.820 palabras, 91 refs) |
| `TFM_datos_reales.docx` | Base pre-discusión |
| `sesion7_informe_19abr2026.docx` | Informe sesión 7 (datos reales) |
| `sesion6_informe_18abr2026_CORREGIDO.docx` | Informe sesión 6 corregido |
| `sesion[1-5]_pipeline_amr.docx` | Histórico sesiones anteriores |

### Scripts principales
| Archivo | Descripción |
|---|---|
| `16_extension_multiAB.py` | Pipeline completo FQ + Ceph3G (sesión 7) |
| `pipeline_multi_antibiotico.py` | Versión anterior multi-AB (referencia) |
| `run_multiAB_local.py` | Script ejecución local |
| `git_commit_sesion7.sh` | Script commit sesiones 6-7 (ya ejecutado) |

### Datos y matrices
| Archivo | Descripción |
|---|---|
| `ml_matrix_binary.csv.gz` | Matriz principal (4044×189, binaria) |
| `feature_matrix_real.csv` | Features con datos reales |
| `dataset_final.tsv` | Dataset carbapenémicos (4.126 genomas) |
| `amr_fenotipos_carbapenems_clean.tsv` | Fenotipos carbapenémicos limpios |

### Figuras
| Directorio | Contenido |
|---|---|
| `figuras_tfm/` | 8 PNG: fig4-8 + ROC + confusión + feature_importance |
| `figuras_multiab/` | 5 PNG: figA-E comparativas multi-AB (datos reales, 19-abr) |

### Resultados
| Directorio | Contenido |
|---|---|
| `resultados_reales/` | 15 arch.: métricas + modelos carbapenémicos (.joblib) |
| `resultados_multiAB/` | 13 arch.: métricas + modelos FQ + Ceph3G (.joblib) |

---

## 🔬 Dataset y resultados

### Features
- **Total:** 188 features binarias (presencia/ausencia gen)
- 82 `RGI_` — CARD v4.0.1 (RGI)
- 70 `RES_` — ResFinder
- 36 `PF_` — PointFinder
- Índice: `genome_id` → parsear como `.astype(str).str.strip()`

### Resultados reales (5-fold StratifiedKFold)

| Grupo | n | %R | Mejor modelo | AUC | F1 |
|---|---|---|---|---|---|
| Carbapenémicos | 4.044 | 39.5% | GBT | 0.846±0.009 | 0.708 |
| Fluoroquinolonas | 3.495 | 69.5% | LightGBM | 0.905±0.012 | 0.876 |
| Cefalosporinas 3G | 3.763 | 78.6% | LightGBM | 0.904±0.016 | 0.909 |

### Top features por grupo
- **Carbapenémicos:** `blaOXA-9` (0.101), `blaKPC-2` (0.082), `BRP(MBL)` (0.041)
- **Fluoroquinolonas:** `parC` (0.127), `gyrA Salmonella isangi` (0.072), `ompK36` (0.043)

---

## 📜 Scripts — inventario completo

### Descarga (`scripts/descarga/`)
| Script | Función |
|---|---|
| `01_descarga_bvbrc.py` | Metadatos K. pneumoniae desde BV-BRC API |
| `02_descarga_amr_fenotipos.py` | Fenotipos AMR validados laboratorio |
| `03_descarga_fastas.py` | Descarga masiva genomas FASTA |

### Anotación (`scripts/anotacion/`)
| Script | Función |
|---|---|
| `04_run_rgi_masivo.sh` | Pipeline RGI masivo (CARD) |
| `05_consolidar_rgi.py` | Consolida RGI → matriz features |
| `09_run_resfinder_masivo.sh` | Pipeline ResFinder + PointFinder |
| `10_consolidar_resfinder.py` | Consolida ResFinder → features |
| `12_consolidar_pointfinder.py` | Consolida PointFinder → features |
| `fix_fasta_headers.py` | Corrección cabeceras FASTA problemáticas |

### ML (`scripts/ml/`)
| Script | Función |
|---|---|
| `06_build_matrix.py` | Matriz binaria presencia/ausencia |
| `07_modelo_baseline.py` | Modelos baseline (RF/GBT/SVM/LR) |
| `08_eda.py` | EDA + análisis estadístico |
| `11_optimizar_modelo.py` | Optimización hiperparámetros GBT |
| `13_xgboost_lightgbm.py` | XGBoost + LightGBM |
| `14_deep_learning.py` | MLP (red neuronal) |
| `15_metricas_clinicas.py` | Métricas clínicas (sens/spec/VPP/VPN) |
| `16_extension_multiAB.py` | Pipeline completo FQ + Ceph3G |

---

## 🔀 Git — estado actual

**Remoto:** `https://github.com/ziskoj/TFM-AMR-Genomic-Predictor.git`  
**Rama:** `main`

| Commit | Descripción | Estado |
|---|---|---|
| `e76d350` | fix: figuras multi-AB con datos reales (figA-E sesión 7) | ✅ pusheado |
| `d86b95e` | feat: extensión multi-AB datos reales (TSVs + script 16) | ✅ pusheado |
| `3dc6612` | feat: pipeline multi-AB y figuras sesión 6 | histórico |

**Pendiente en git:**
- `TFM_discusion_completa.docx` → commit cuando tutor valide

**Comandos frecuentes en WSL:**
```bash
cd /mnt/f/TFM-AMR-Genomic-Predictor
git status --short
git log --oneline -5
git push origin main
```

---

## 🔐 Auditoría e integridad del SSD

**⚠️ NUNCA modificar `/mnt/f/MIS_DATOS_TFM/` — es READONLY**

```bash
# Verificar integridad datos SSD (ejecutar ANTES de cada sesión)
/mnt/f/TFM-AMR-Genomic-Predictor/cowork-backup/audit_before_session.sh

# Solo checksums (verificación rápida):
cd /mnt/f/MIS_DATOS_TFM
md5sum -c /mnt/f/TFM-AMR-Genomic-Predictor/cowork-backup/MIS_DATOS_TFM_CHECKSUMS.md5 2>&1 | grep -v "OK$"
# Sin output = datos íntegros ✅

# Verificar al TERMINAR sesión
/mnt/f/TFM-AMR-Genomic-Predictor/cowork-backup/audit_after_session.sh
```

---

## 💻 Entornos de ejecución (WSL)

```bash
# Entorno ML (Python 3.10 — scikit-learn, xgboost, lightgbm, tensorflow)
source /mnt/f/TFM_Linux/envs/amr_env/bin/activate

# Entorno anotación (RGI + BLAST + DIAMOND)
conda activate rgi_env
```

---

## 📋 Estado TFM — checklist

- ✅ Introducción
- ✅ Marco Teórico
- ✅ Metodología
- ✅ Resultados carbapenémicos (datos reales)
- ✅ Resultados extensión multi-AB (datos reales)
- ✅ Discusión (4 subsecciones: interpretación biológica, literatura, limitaciones, prospectiva)
- ✅ Conclusiones (6 puntos numerados)
- ✅ Referencias (91 total, 84-91 nuevas para multi-AB)
- ⚠️ TOC/Índice — actualizar con F9 en Word
- ⚠️ Revisión tutor — pendiente feedback
- ⬜ Commit `TFM_discusion_completa.docx` en git (tras validación tutor)
- ⬜ Entrega final

---

## 🔗 Referencias bibliográficas nuevas (84-91)

| Ref | Autores | Año | Tema |
|---|---|---|---|
| 84 | Cantón R et al. | 2021 | Selección resistencia antibióticos |
| 85 | Peiffer-Smadja N et al. | 2020 | ML en microbiología clínica |
| 86 | Tsang KK et al. | 2022 | Modelos generalizables multi-especie |
| 87 | Kouchaki S et al. | 2019 | ML predicción resistencia TB |
| 88 | Moradigaravand D et al. | 2018 | Predicción RAM en E. coli |
| 89 | Pataki BÁ et al. | 2020 | Predicción CMI ciprofloxacino E. coli |
| 90 | Nguyen M et al. | 2018 | Panel CMI in silico K. pneumoniae |
| 91 | Boolchandani M et al. | 2019 | Métodos secuenciación RAM |
