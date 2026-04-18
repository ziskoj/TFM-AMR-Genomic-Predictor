#!/usr/bin/env python3
"""
pipeline_multi_antibiotico.py
─────────────────────────────────────────────────────────────────────────────
Extensión del pipeline AMR de K. pneumoniae a múltiples antibióticos.

Antibióticos objetivo
---------------------
  1. Carbapenémicos  (meropenem  / imipenem)   ← estudio original
  2. Fluoroquinolonas (ciprofloxacino)          ← extensión
  3. Cefalosporinas 3ª gen (ceftriaxona)        ← extensión

Flujo de trabajo
----------------
  1. Descarga de metadatos BV-BRC con fenotipos para los 3 antibióticos
  2. Filtrado de calidad y alineación con genomas ya anotados (RGI/ResFinder/PF)
  3. Construcción de matrices fenotípicas por antibiótico
  4. Entrenamiento de 4 modelos (GBT, XGBoost, LightGBM, MLP) por antibiótico
  5. Validación cruzada estratificada 5-fold
  6. Exportación de métricas y curvas ROC

Requisitos
----------
  pip install requests pandas scikit-learn xgboost lightgbm tensorflow numpy

Uso
---
  python pipeline_multi_antibiotico.py \
      --genomes_dir  /ruta/a/genomas_anotados/ \
      --output_dir   /ruta/a/resultados/ \
      --antibiotics  carbapenems fluoroquinolones cephalosporins3g \
      --n_jobs       4

Referencia BV-BRC
-----------------
  Olson RD et al. Nucleic Acids Res. 2023;51(D1):D678-D689.
  API: https://www.bv-brc.org/api/
"""

import os
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              confusion_matrix, roc_curve)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost no disponible. Se omitirá.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM no disponible. Se omitirá.")

# ─── Configuración de logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─── Constantes BV-BRC ───────────────────────────────────────────────────────
BVBRC_API = "https://www.bv-brc.org/api"
TAXON_KPNEU = "573"          # Klebsiella pneumoniae NCBI TaxID

# Mapeo antibiótico → términos de búsqueda en BV-BRC AMR metadata
ANTIBIOTIC_MAP = {
    "carbapenems": {
        "display": "Carbapenémicos",
        "bvbrc_terms": ["meropenem", "imipenem", "ertapenem", "doripenem"],
        "primary": "meropenem",
        "drug_class": "carbapenem",
        "breakpoint_r_ugml": 2.0,   # EUCAST 2024: R > 2 µg/mL meropenem
    },
    "fluoroquinolones": {
        "display": "Fluoroquinolonas",
        "bvbrc_terms": ["ciprofloxacin", "levofloxacin", "norfloxacin"],
        "primary": "ciprofloxacin",
        "drug_class": "fluoroquinolone",
        "breakpoint_r_ugml": 0.5,   # EUCAST 2024: R > 0.5 µg/mL ciprofloxacino
    },
    "cephalosporins3g": {
        "display": "Cefalosporinas 3ª Gen.",
        "bvbrc_terms": ["ceftriaxone", "cefotaxime", "ceftazidime"],
        "primary": "ceftriaxone",
        "drug_class": "cephalosporin",
        "breakpoint_r_ugml": 2.0,   # EUCAST 2024: R > 2 µg/mL ceftriaxona (BLEE)
    },
}

# Genes clave por clase (para verificación de coherencia fenotipo-genotipo)
MARKER_GENES = {
    "carbapenems":      ["blaKPC", "blaNDM", "blaVIM", "blaIMP", "blaOXA-48",
                         "ompK36_mut", "ompK35_mut"],
    "fluoroquinolones": ["gyrA_mut", "parC_mut", "gyrB_mut", "parE_mut",
                         "qnrA", "qnrB", "qnrS", "oqxAB", "aac(6)-Ib-cr"],
    "cephalosporins3g": ["blaCTX-M-15", "blaCTX-M-14", "blaCTX-M-3",
                         "blaSHV-12", "blaSHV-2a", "blaSHV-5",
                         "blaTEM-52", "blaTEM-26"],
}

# ─── Funciones de descarga BV-BRC ────────────────────────────────────────────

def fetch_bvbrc_amr(antibiotic_terms: List[str],
                    taxon: str = TAXON_KPNEU,
                    limit: int = 50000) -> pd.DataFrame:
    """
    Descarga registros AMR desde BV-BRC para los antibióticos indicados.

    Parámetros
    ----------
    antibiotic_terms : lista de nombres de antibiótico (en inglés, BV-BRC convention)
    taxon            : NCBI Taxonomy ID de la especie objetivo
    limit            : número máximo de registros por petición

    Devuelve
    --------
    DataFrame con columnas: genome_id, antibiotic, resistant_phenotype,
                             measurement_value, measurement_unit, laboratory_typing_method
    """
    antibiotic_filter = " OR ".join(
        [f'antibiotic:"{ab}"' for ab in antibiotic_terms]
    )
    query = (
        f"select(genome_id,antibiotic,resistant_phenotype,"
        f"measurement_value,measurement_unit,laboratory_typing_method,"
        f"evidence,genome_name)"
        f"&eq(taxon_lineage_ids,{taxon})"
        f"&or({antibiotic_filter})"
        f"&limit({limit})"
        f"&http_accept=application/json"
    )
    url = f"{BVBRC_API}/genome_amr/?{query}"

    logger.info(f"  GET {url[:120]}…")
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=120,
                                headers={"Accept": "application/json"})
            resp.raise_for_status()
            records = resp.json()
            df = pd.DataFrame(records)
            logger.info(f"  → {len(df):,} registros obtenidos")
            return df
        except Exception as exc:
            logger.warning(f"  Intento {attempt+1}/3 fallido: {exc}")
            time.sleep(5 * (attempt + 1))

    raise RuntimeError("No se pudo conectar con BV-BRC tras 3 intentos.")


def fetch_bvbrc_genomes(genome_ids: List[str],
                        fields: str = "genome_id,genome_name,collection_year,geographic_location") -> pd.DataFrame:
    """Descarga metadatos de genomas desde BV-BRC."""
    ids_str = ",".join(genome_ids[:5000])   # máximo 5000 por petición
    url = (f"{BVBRC_API}/genome/?in(genome_id,({ids_str}))"
           f"&select({fields})&limit(5000)"
           f"&http_accept=application/json")
    resp = requests.get(url, timeout=120, headers={"Accept": "application/json"})
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


# ─── Preprocesamiento ────────────────────────────────────────────────────────

def clean_amr_records(df: pd.DataFrame,
                      valid_phenotypes: List[str] = ("Resistant", "Susceptible")) -> pd.DataFrame:
    """
    Filtra registros AMR:
      - Solo fenotipos válidos (Resistant / Susceptible)
      - Elimina duplicados (genome_id + antibiotic → fenotipo mayoritario)
      - Requiere método de laboratorio estándar (MIC o disk diffusion)
    """
    df = df[df["resistant_phenotype"].isin(valid_phenotypes)].copy()
    df = df[df["laboratory_typing_method"].str.lower().isin(
        ["mic", "disk diffusion", "broth microdilution", "agar dilution",
         "vitek", "microscan", "sensititre", "etest"]
    )].copy()

    # Si hay duplicados, conservar el fenotipo mayoritario
    dedup = (
        df.groupby(["genome_id", "antibiotic"])["resistant_phenotype"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )
    logger.info(f"  Registros tras limpieza: {len(dedup):,} "
                f"({dedup['resistant_phenotype'].value_counts().to_dict()})")
    return dedup


def build_feature_matrix(feature_csv: str,
                         phenotype_df: pd.DataFrame,
                         min_freq: float = 0.01,
                         max_freq: float = 0.99) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construye la matriz X (genes AMR binarios) y el vector y (fenotipo).

    Parámetros
    ----------
    feature_csv  : ruta al CSV de features (genome_id + columnas de genes)
    phenotype_df : DataFrame con genome_id y resistant_phenotype
    min_freq     : frecuencia mínima de un gen para incluirlo
    max_freq     : frecuencia máxima (elimina genes ubicuos)
    """
    features = pd.read_csv(feature_csv, index_col="genome_id")
    merged = features.join(
        phenotype_df.set_index("genome_id")["resistant_phenotype"],
        how="inner"
    ).dropna(subset=["resistant_phenotype"])

    y = (merged["resistant_phenotype"] == "Resistant").astype(int)
    X = merged.drop(columns=["resistant_phenotype"])

    # Filtrado por frecuencia
    freq = X.mean()
    mask = (freq >= min_freq) & (freq <= max_freq)
    X = X.loc[:, mask]
    logger.info(f"  Features tras filtrado de frecuencia: {X.shape[1]} "
                f"(eliminadas: {(~mask).sum()})")

    return X, y


# ─── Modelos ─────────────────────────────────────────────────────────────────

def get_models(random_state: int = 42) -> Dict:
    """Devuelve el diccionario de modelos a evaluar."""
    models = {
        "GBT": GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=random_state
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu", max_iter=500,
            early_stopping=True, random_state=random_state
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=random_state
        )
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=random_state, verbose=-1
        )
    return models


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series,
                         n_splits: int = 5,
                         random_state: int = 42) -> Dict:
    """
    Validación cruzada estratificada con cálculo de métricas.

    Devuelve
    --------
    dict con AUC_mean, AUC_sd, F1_mean, F1_sd, Accuracy_mean,
             Sensitivity_mean, Specificity_mean y roc_curves
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    aucs, f1s, accs, sens, specs = [], [], [], [], []
    roc_data = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data.append({"fpr": fpr.tolist(), "tpr": tpr.tolist()})

        aucs.append(roc_auc_score(y_test, y_prob))
        f1s.append(f1_score(y_test, y_pred))
        accs.append(accuracy_score(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

        logger.debug(f"    Fold {fold+1}: AUC={aucs[-1]:.3f} F1={f1s[-1]:.3f}")

    return {
        "AUC_mean": np.mean(aucs),    "AUC_sd": np.std(aucs),
        "F1_mean":  np.mean(f1s),     "F1_sd":  np.std(f1s),
        "Acc_mean": np.mean(accs),
        "Sens_mean": np.mean(sens),   "Spec_mean": np.mean(specs),
        "roc_curves": roc_data,
    }


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_pipeline(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for ab_key in args.antibiotics:
        if ab_key not in ANTIBIOTIC_MAP:
            logger.error(f"Antibiótico desconocido: {ab_key}. Ignorado.")
            continue

        ab_cfg = ANTIBIOTIC_MAP[ab_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"ANTIBIÓTICO: {ab_cfg['display']} ({ab_cfg['primary']})")
        logger.info(f"{'='*60}")

        # 1. Descargar metadatos fenotípicos
        logger.info("Paso 1: Descarga de metadatos BV-BRC…")
        amr_df = fetch_bvbrc_amr(ab_cfg["bvbrc_terms"])
        amr_clean = clean_amr_records(amr_df)

        # Guardar metadatos crudos
        amr_clean.to_csv(output_dir / f"phenotypes_{ab_key}.csv", index=False)
        logger.info(f"  Guardado: phenotypes_{ab_key}.csv")

        # 2. Construir matriz de features
        logger.info("Paso 2: Construcción de matriz de features…")
        feature_csv = Path(args.genomes_dir) / "feature_matrix_188.csv"
        if not feature_csv.exists():
            logger.error(f"No se encuentra {feature_csv}. "
                         "Asegúrate de haber ejecutado el pipeline original primero.")
            continue

        X, y = build_feature_matrix(str(feature_csv), amr_clean)
        logger.info(f"  Dataset: {len(X):,} genomas, {X.shape[1]} features, "
                    f"Resistentes={y.sum():,} ({y.mean()*100:.1f}%)")

        # 3. Entrenar y evaluar modelos
        logger.info("Paso 3: Entrenamiento y validación cruzada…")
        models = get_models(random_state=42)
        ab_results = {"config": ab_cfg, "dataset": {
            "n_total": len(X), "n_resistant": int(y.sum()),
            "n_susceptible": int((y == 0).sum()),
            "n_features": X.shape[1],
            "pct_resistant": float(y.mean() * 100),
        }, "models": {}}

        for model_name, model in models.items():
            logger.info(f"  → {model_name}…")
            metrics = cross_validate_model(model, X, y)
            ab_results["models"][model_name] = metrics
            logger.info(
                f"    AUC={metrics['AUC_mean']:.3f}±{metrics['AUC_sd']:.3f}  "
                f"F1={metrics['F1_mean']:.3f}±{metrics['F1_sd']:.3f}  "
                f"Sens={metrics['Sens_mean']:.3f}  Spec={metrics['Spec_mean']:.3f}"
            )

        all_results[ab_key] = ab_results

        # 4. Guardar resultados por antibiótico
        result_path = output_dir / f"results_{ab_key}.json"
        with open(result_path, "w") as f:
            # Convertir numpy a Python nativo para serialización
            def convert(obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                raise TypeError
            json.dump(ab_results, f, indent=2, default=convert)
        logger.info(f"  Guardado: {result_path}")

    # 5. Tabla comparativa global
    logger.info("\nGenerando tabla comparativa global…")
    rows = []
    for ab_key, ab_data in all_results.items():
        for model_name, metrics in ab_data["models"].items():
            rows.append({
                "Antibiótico": ab_data["config"]["display"],
                "Modelo": model_name,
                "N_genomas": ab_data["dataset"]["n_total"],
                "Pct_R": ab_data["dataset"]["pct_resistant"],
                "AUC": metrics["AUC_mean"],
                "AUC_SD": metrics["AUC_sd"],
                "F1": metrics["F1_mean"],
                "Sensibilidad": metrics["Sens_mean"],
                "Especificidad": metrics["Spec_mean"],
            })

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(output_dir / "comparison_all_antibiotics.csv", index=False)
    logger.info(f"Guardado: {output_dir}/comparison_all_antibiotics.csv")
    logger.info("\n✅ Pipeline completado.")


# ─── Argumentos CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline multi-antibiótico AMR para K. pneumoniae"
    )
    parser.add_argument(
        "--genomes_dir", required=True,
        help="Directorio con feature_matrix_188.csv (salida del pipeline original)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directorio de salida para resultados"
    )
    parser.add_argument(
        "--antibiotics", nargs="+",
        default=["carbapenems", "fluoroquinolones", "cephalosporins3g"],
        choices=list(ANTIBIOTIC_MAP.keys()),
        help="Antibióticos a analizar (default: los 3 grupos)"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1,
        help="Número de cores para paralelización"
    )
    args = parser.parse_args()
    run_pipeline(args)
