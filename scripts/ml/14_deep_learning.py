#!/usr/bin/env python3
"""
14_deep_learning.py
Modelos de Deep Learning para predicción AMR
- MLP (Multi-Layer Perceptron) con Keras/TensorFlow
- Validación cruzada estratificada 5-fold
Input:  ml_matrix_binary.csv.gz
Output: resultados_dl.tsv + modelo guardado
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import warnings
warnings.filterwarnings('ignore')
os_environ = {'TF_CPP_MIN_LOG_LEVEL': '3'}
import os
os.environ.update(os_environ)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

# ── Rutas ──────────────────────────────────────────────────────────────────
ML_DIR  = Path('/mnt/f/TFM_Linux/ml_matrices')
OUT_DIR = Path('/mnt/f/TFM_Linux/ml_matrices/resultados')
LOG_DIR = Path('/mnt/f/MIS_DATOS_TFM/logs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'deep_learning.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

SEED     = 42
N_SPLITS = 5
tf.random.set_seed(SEED)
np.random.seed(SEED)

def load_data():
    log.info('Cargando matriz completa...')
    df = pd.read_csv(ML_DIR / 'ml_matrix_binary.csv.gz', index_col=0)
    X = df.drop(columns=['target']).values.astype(np.float32)
    y = df['target'].values.astype(np.float32)
    feature_names = [c for c in df.columns if c != 'target']
    log.info(f'  Shape: {X.shape} · Balance: {int((y==0).sum())} S / {int((y==1).sum())} R')
    return X, y, feature_names

def build_mlp(input_dim, dropout_rate=0.3, l2_reg=0.001):
    """MLP con BatchNormalization, Dropout y regularización L2."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        # Bloque 1
        layers.Dense(256, kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),

        # Bloque 2
        layers.Dense(128, kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),

        # Bloque 3
        layers.Dense(64, kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate / 2),

        # Bloque 4
        layers.Dense(32, kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Activation('relu'),

        # Salida
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def cross_validate_mlp(X, y):
    """Validación cruzada estratificada del MLP."""
    log.info('\n=== Entrenando MLP (Deep Learning) ===')
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    aucs, f1s, accs = [], [], []
    histories = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        log.info(f'  Fold {fold+1}/{N_SPLITS}...')

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Escalar features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        # Calcular class weights
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        class_weight = {0: 1.0, 1: n_neg / n_pos}

        # Construir y compilar modelo
        model = build_mlp(X_train.shape[1])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['AUC']
        )

        # Callbacks
        cb = [
            callbacks.EarlyStopping(
                monitor='val_auc', patience=20,
                restore_best_weights=True, mode='max'
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_auc', factor=0.5,
                patience=10, mode='max', verbose=0
            )
        ]

        # Entrenar
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=64,
            class_weight=class_weight,
            callbacks=cb,
            verbose=0
        )
        histories.append(history)

        # Evaluar
        y_prob = model.predict(X_val, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_val, y_prob)
        f1  = f1_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)

        aucs.append(auc)
        f1s.append(f1)
        accs.append(acc)
        log.info(f'    AUC={auc:.4f} · F1={f1:.4f} · Acc={acc:.4f}')

        # Limpiar memoria
        keras.backend.clear_session()

    log.info(f'\n  RESULTADO FINAL MLP:')
    log.info(f'    AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    log.info(f'    F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
    log.info(f'    Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}')

    return {
        'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs),
        'f1_mean':  np.mean(f1s),  'f1_std':  np.std(f1s),
        'acc_mean': np.mean(accs)
    }

def entrenar_modelo_final(X, y):
    """Entrena el MLP final sobre todos los datos."""
    log.info('\nEntrenando modelo final sobre todos los datos...')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    class_weight = {0: 1.0, 1: n_neg / n_pos}

    model = build_mlp(X.shape[1])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    model.fit(
        X_scaled, y,
        epochs=100,
        batch_size=64,
        class_weight=class_weight,
        verbose=0
    )

    # Guardar modelo y scaler
    model.save(OUT_DIR / 'modelo_mlp.keras')
    joblib.dump(scaler, OUT_DIR / 'scaler_mlp.joblib')
    log.info(f'  Modelo guardado: {OUT_DIR}/modelo_mlp.keras')
    return model, scaler

def main():
    log.info('=== DEEP LEARNING — MLP para predicción AMR ===')
    X, y, feature_names = load_data()

    # Validación cruzada
    resultados = cross_validate_mlp(X, y)

    # Modelo final
    entrenar_modelo_final(X, y)

    # Comparativa
    print('\n' + '='*65)
    print('COMPARATIVA COMPLETA — TODOS LOS MODELOS')
    print('='*65)
    print(f'GBT optimizado:    AUC = 0.8460')
    print(f'XGBoost:           AUC = 0.8449')
    print(f'LightGBM:          AUC = 0.8437')
    print(f'MLP (Deep Learning): AUC = {resultados["auc_mean"]:.4f} ± {resultados["auc_std"]:.4f}')
    print(f'                     F1  = {resultados["f1_mean"]:.4f} ± {resultados["f1_std"]:.4f}')
    print('='*65)

    # Guardar resultados
    df_res = pd.DataFrame([{
        'modelo': 'MLP',
        'auc_mean': resultados['auc_mean'],
        'auc_std':  resultados['auc_std'],
        'f1_mean':  resultados['f1_mean'],
        'f1_std':   resultados['f1_std'],
        'accuracy': resultados['acc_mean'],
    }])
    df_res.to_csv(OUT_DIR / 'resultados_dl.tsv', sep='\t', index=False)

if __name__ == '__main__':
    main()
