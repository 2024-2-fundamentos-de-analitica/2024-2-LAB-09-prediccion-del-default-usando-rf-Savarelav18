# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
import zipfile

# Rutas de los archivos ZIP
train_zip_path = "files/input/train_data.csv.zip"
test_zip_path = "files/input/test_data.csv.zip"


# Función para cargar y limpiar datos
def load_and_clean_data(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(z.namelist()[0]) as file:
            df = pd.read_csv(file)

    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    df.dropna(inplace=True)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return df


# Cargar y limpiar los datasets
train_df = load_and_clean_data(train_zip_path)
test_df = load_and_clean_data(test_zip_path)

# Mostrar información después de la limpieza
print("Información del conjunto de entrenamiento después de limpieza:")
print(train_df.info())

print("\nInformación del conjunto de prueba después de limpieza:")
print(test_df.info())

# Mostrar primeras filas
print("\nPrimeras filas del conjunto de entrenamiento:")
print(train_df.head())

print("\nPrimeras filas del conjunto de prueba:")
print(test_df.head())


# Separar características (X) y variable objetivo (y)
X_train = train_df.drop(columns=["default"])
y_train = train_df["default"]

X_test = test_df.drop(columns=["default"])
y_test = test_df["default"]

# Mostrar las dimensiones de los conjuntos
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Identificar variables categóricas y numéricas
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]  # Variables categóricas
numerical_features = [
    col for col in X_train.columns if col not in categorical_features
]  # Variables numéricas

# Preprocesamiento: One-Hot Encoding para las variables categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough",  # Deja las variables numéricas sin cambios
)

# Pipeline con preprocesamiento y modelo Random Forest
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

# Entrenar el pipeline
pipeline.fit(X_train, y_train)

# Evaluación inicial del modelo
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"Precisión en entrenamiento: {train_score:.4f}")
print(f"Precisión en prueba: {test_score:.4f}")


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score

# Definir los hiperparámetros a optimizar
param_grid = {
    "classifier__n_estimators": [50, 100, 200],  # Número de árboles
    "classifier__max_depth": [10, 20, None],  # Profundidad máxima de los árboles
    "classifier__min_samples_split": [
        2,
        5,
        10,
    ],  # Mínimo de muestras para dividir un nodo
    "classifier__min_samples_leaf": [1, 2, 4],  # Mínimo de muestras en una hoja
}

# Definir la métrica de optimización
scorer = make_scorer(balanced_accuracy_score)

# GridSearch con validación cruzada de 10 folds
grid_search = GridSearchCV(
    pipeline, param_grid, cv=10, scoring=scorer, n_jobs=-1, verbose=1
)

# Ajustar el modelo con los mejores hiperparámetros
grid_search.fit(X_train, y_train)

# Imprimir los mejores hiperparámetros y la precisión balanceada
print(f"Mejores hiperparámetros: {grid_search.best_params_}")
print(f"Mejor precisión balanceada en validación: {grid_search.best_score_:.4f}")

# Evaluar el modelo optimizado en el conjunto de prueba
test_score = balanced_accuracy_score(
    y_test, grid_search.best_estimator_.predict(X_test)
)
print(f"Precisión balanceada en prueba: {test_score:.4f}")


import os
import gzip
import pickle

# Definir la ruta del directorio y del archivo del modelo
model_dir = "files/models"
model_path = os.path.join(model_dir, "model.pkl.gz")

# Crear la carpeta si no existe
os.makedirs(model_dir, exist_ok=True)

# Guardar el modelo comprimido
with gzip.open(model_path, "wb") as f:
    pickle.dump(grid_search.best_estimator_, f)

print(f"Modelo guardado en {model_path}")


import os
import json
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
)

# Asegurar que la carpeta de salida exista
output_dir = "files/output"
os.makedirs(output_dir, exist_ok=True)

# Hacer predicciones en los conjuntos de entrenamiento y prueba
y_train_pred = grid_search.best_estimator_.predict(X_train)
y_test_pred = grid_search.best_estimator_.predict(X_test)

# Calcular métricas
metrics = [
    {
        "dataset": "train",
        "precision": precision_score(y_train, y_train_pred),
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred),
        "f1_score": f1_score(y_train, y_train_pred),
    },
    {
        "dataset": "test",
        "precision": precision_score(y_test, y_test_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1_score": f1_score(y_test, y_test_pred),
    },
]

# Guardar métricas en JSON
metrics_path = os.path.join(output_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Métricas guardadas en {metrics_path}")


import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix

# Cargar el archivo de métricas existente
metrics_path = "files/output/metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
else:
    metrics = []

# Calcular matrices de confusión
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)


# Función para convertir matriz de confusión a diccionario
def format_confusion_matrix(cm, dataset_name):
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }


# Agregar matrices de confusión al JSON
metrics.append(format_confusion_matrix(cm_train, "train"))
metrics.append(format_confusion_matrix(cm_test, "test"))

# Guardar el archivo actualizado
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Matrices de confusión agregadas en {metrics_path}")
