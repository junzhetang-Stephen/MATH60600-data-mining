#!/usr/bin/env python
# coding: utf-8

# # Partie B — Optimisation des hyperparamètres
# 
# ## Objectif
# 
# Ce notebook applique **trois méthodes d'optimisation** aux données continues de l'exemple de jeux en ligne du cours :
# 
# 1. **Grid Search** avec `GridSearchCV`;
# 2. **Optimisation bayésienne** avec `BayesSearchCV`;
# 3. **Keras Tuner (Hyperband)** pour optimiser un réseau de neurones de régression.
# 
# Les trois expériences utilisent le même fichier, la même variable cible continue `y` et la même séparation entraînement/validation. Les deux premières optimisent une forêt aléatoire de régression (`RandomForestRegressor`); la troisième optimise un réseau de neurones de régression.
# 
# **Important :** ne pas remplacer les données du cours par `make_classification`, `make_regression` ou les fichiers Kaggle de la Partie A. La Partie B exige le Y continu de l'exemple du cours.

# ## 1. Les trois méthodes utilisées
# 
# ### 1.1 Grid Search (recherche par grille)
# Grid Search définit à l'avance plusieurs valeurs pour chaque hyperparamètre et évalue toutes les combinaisons possibles. Cette méthode est simple et systématique, mais le nombre d'essais peut devenir très élevé lorsque l'espace de recherche augmente.
# 
# ### 1.2 Bayesian Optimization (optimisation bayésienne)
# L'optimisation bayésienne exploite les résultats des essais précédents pour choisir les prochains hyperparamètres à tester. Elle construit une approximation probabiliste de la performance et privilégie les essais susceptibles d'améliorer le résultat. Elle peut donc utiliser le budget de calcul plus efficacement qu'une recherche exhaustive.
# 
# ### 1.3 Keras Tuner — Hyperband
# Keras Tuner est une bibliothèque d'optimisation pour les modèles Keras. Hyperband compare plusieurs configurations et alloue progressivement davantage de ressources aux configurations prometteuses, tout en arrêtant plus tôt celles qui semblent moins performantes. Dans cet exemple, les hyperparamètres du réseau de neurones (nombre d'unités, taux d'apprentissage et régularisation) sont optimisés pour une tâche de régression.
# 
# **Différence importante :** Grid Search et Bayesian Optimization optimisent ici une forêt aléatoire de régression; Keras Tuner optimise un réseau de neurones de régression. Les trois méthodes sont appliquées aux mêmes données du cours et évaluées avec les mêmes mesures.

# ## 2. Packages requis
# 
# Exécuter la cellule d'installation si certains packages ne sont pas disponibles. TensorFlow et Keras Tuner sont nécessaires uniquement pour la troisième méthode.

# In[ ]:


# Installation (décommenter au besoin)
# %pip install numpy pandas scikit-learn scikit-optimize matplotlib joblib tensorflow keras-tuner


# In[5]:


# ============================================================
# 1) Imports et paramètres généraux
# ============================================================

from pathlib import Path
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

import tensorflow as tf
import keras_tuner as kt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Chemin exact fourni pour le fichier d'entraînement du cours.
DATA_PATH = Path(
    r"C:\Users\Stephen\Desktop\Math60600数据挖掘课件\documents_etudiants_A2026\donnees\clv_mobile_data_train.txt"
)

# Nombre d'essais : ces valeurs peuvent être augmentées si le temps de calcul le permet.
BAYES_N_ITER = 20
KERAS_MAX_TRIALS = 20
KERAS_MAX_EPOCHS = 30

OUTPUT_DIR = Path.cwd() / "partB_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Fichier de données :", DATA_PATH)
print("Dossier des résultats :", OUTPUT_DIR.resolve())


# ## 3. Charger les données du cours
# 
# Le code essaie plusieurs séparateurs courants. Il exige une colonne appelée `y`. Après la lecture, il vérifie que `y` est numérique et qu'il ne s'agit pas du Y catégoriel 1, 2, 3, 4 de la Partie A.

# In[6]:


# ============================================================
# 2) Lecture du fichier
# ============================================================

def read_course_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}\n"
            "Vérifiez que le fichier existe à cet emplacement."
        )

    candidates = []
    # sep=None détecte souvent les séparateurs; les autres essais couvrent
    # les fichiers séparés par espaces, tabulations, virgules ou points-virgules.
    for sep in [None, r"\s+", "\\t", ",", ";"]:
        try:
            temp = pd.read_csv(
                file_path, sep=sep, engine="python", quotechar='"'
            )
            temp.columns = [str(c).strip() for c in temp.columns]
            temp = temp.dropna(axis=1, how="all")
            y_cols = [c for c in temp.columns if c.lower() == "y"]
            if len(y_cols) == 1 and temp.shape[1] >= 2:
                score = (temp.shape[1], -int(temp.isna().sum().sum()))
                candidates.append((score, temp, sep))
        except Exception:
            pass

    if not candidates:
        raise ValueError(
            "Lecture impossible ou colonne 'y' introuvable. "
            "Vérifiez le séparateur et l'en-tête du fichier."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, data, detected_sep = candidates[0]
    y_col = [c for c in data.columns if c.lower() == "y"][0]
    data = data.rename(columns={y_col: "y"})

    print("Séparateur utilisé :", repr(detected_sep))
    return data


df = read_course_data(DATA_PATH)

print("Dimensions du fichier :", df.shape)
print("Colonnes :", df.columns.tolist())
display(df.head())
display(df.dtypes.to_frame("Type de variable"))


# ## 4. Préparer X et le Y continu
# 
# Les valeurs de `y` qui sont manquantes ou non numériques sont retirées. Les variables explicatives numériques et catégorielles sont prétraitées séparément. Les imputations et transformations sont placées dans un pipeline afin que les paramètres de prétraitement soient appris uniquement à partir des données d'entraînement.
# 
# Si le fichier contient exactement les quatre catégories 1, 2, 3 et 4, le code s'arrête : cela indiquerait probablement que le fichier de la Partie A a été utilisé par erreur.

# In[7]:


# ============================================================
# 3) Nettoyage et séparation de X et y
# ============================================================

y_numeric = pd.to_numeric(df["y"], errors="coerce")
valid_rows = y_numeric.notna()

if (~valid_rows).sum() > 0:
    print("Lignes retirées : y manquant/non numérique =", int((~valid_rows).sum()))

df = df.loc[valid_rows].copy()
y = y_numeric.loc[valid_rows].astype(float)
X = df.drop(columns=["y"]).copy()
X = X.dropna(axis=1, how="all")

# Convertir les colonnes majoritairement numériques.
for col in X.columns:
    converted = pd.to_numeric(X[col], errors="coerce")
    non_missing = X[col].notna().sum()
    if non_missing > 0 and converted.notna().sum() / non_missing >= 0.95:
        X[col] = converted

if X.shape[1] == 0:
    raise ValueError("Aucune variable explicative utilisable.")
if y.nunique() < 2:
    raise ValueError("La variable y doit contenir au moins deux valeurs distinctes.")

unique_y = set(pd.unique(y))
if len(unique_y) == 4 and unique_y.issubset({1.0, 2.0, 3.0, 4.0}):
    raise ValueError(
        "Le Y contient les quatre classes 1, 2, 3, 4 de la Partie A. "
        "La Partie B demande le Y continu du cours."
    )

print("Dimensions de X :", X.shape)
print("Nombre d'observations :", len(y))
print("Nombre de valeurs distinctes de y :", y.nunique())
print("Minimum de y :", y.min())
print("Maximum de y :", y.max())
print("Moyenne de y :", y.mean())
print("Écart-type de y :", y.std())

if y.nunique() <= 10:
    warnings.warn(
        "Peu de valeurs distinctes pour y : vérifiez que le Y est bien continu.",
        UserWarning
    )

display(y.describe().to_frame("Statistiques de y"))


# ## 5. Séparation entraînement / validation et prétraitement commun
# 
# Les trois méthodes utilisent la même séparation des données. Les variables numériques sont imputées par la médiane. Les variables catégorielles sont imputées par la modalité la plus fréquente et encodées en One-Hot.
# 
# Pour Grid Search et Bayesian Optimization, le prétraitement est intégré au pipeline de la forêt aléatoire. Pour Keras Tuner, le prétraitement est ajusté sur l'ensemble d'entraînement uniquement, puis appliqué à l'ensemble de validation.

# In[8]:


# ============================================================
# 4) Séparation des données
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [c for c in X_train.columns if c not in numeric_features]

print("Taille entraînement :", X_train.shape)
print("Taille validation :", X_valid.shape)
print("Variables numériques :", len(numeric_features))
print("Variables catégorielles :", len(categorical_features))

# Préprocesseur pour les modèles sklearn.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

# Même prétraitement de base pour les deux recherches sklearn.
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
])

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# ## 6. Méthode 1 — Grid Search
# 
# Grid Search teste toutes les combinaisons définies dans `param_grid`. La performance est mesurée par le RMSE négatif (`neg_root_mean_squared_error`), car Scikit-learn maximise les scores : le signe négatif permet donc de sélectionner l'erreur quadratique moyenne la plus faible.

# In[9]:


# ============================================================
# 5) GRID SEARCH — RandomForestRegressor
# ============================================================

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

grid_best_model = grid_search.best_estimator_
grid_best_cv_rmse = -grid_search.best_score_

print("Meilleurs paramètres Grid Search :")
print(grid_search.best_params_)
print("Meilleur RMSE CV Grid Search :", grid_best_cv_rmse)


# ## 7. Méthode 2 — Bayesian Optimization
# 
# Bayesian Optimization teste un nombre limité de configurations et utilise les résultats précédents pour guider les essais suivants. Ici, `BayesSearchCV` optimise les mêmes hyperparamètres d'une forêt aléatoire de régression, avec la même validation croisée à 5 plis.

# In[10]:


# ============================================================
# 6) BAYESIAN OPTIMIZATION — RandomForestRegressor
# ============================================================

search_spaces = {
    "model__n_estimators": Integer(80, 300),
    "model__max_depth": Categorical([None, 5, 10, 15, 20, 30]),
    "model__min_samples_split": Integer(2, 12)
}

bayes_search = BayesSearchCV(
    estimator=rf_pipeline,
    search_spaces=search_spaces,
    n_iter=BAYES_N_ITER,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0
)

bayes_search.fit(X_train, y_train)

bayes_best_model = bayes_search.best_estimator_
bayes_best_cv_rmse = -bayes_search.best_score_

print("Meilleurs paramètres Bayesian Optimization :")
print(bayes_search.best_params_)
print("Meilleur RMSE CV Bayesian Optimization :", bayes_best_cv_rmse)


# ## 8. Méthode 3 — Keras Tuner (Hyperband)
# 
# Cette méthode optimise un réseau de neurones pour la régression. La couche de sortie contient un seul neurone sans fonction d'activation, ce qui permet de prédire une valeur continue. La fonction de perte est l'erreur quadratique moyenne (MSE), et l'indicateur suivi est le RMSE de validation.
# 
# Les variables sont encodées puis standardisées pour faciliter l'entraînement du réseau de neurones.

# In[11]:


# ============================================================
# 7) Prétraitement spécifique au réseau de neurones
# ============================================================

nn_numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

nn_categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

nn_preprocessor = ColumnTransformer(
    transformers=[
        ("num", nn_numeric_transformer, numeric_features),
        ("cat", nn_categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

X_train_nn = nn_preprocessor.fit_transform(X_train)
X_valid_nn = nn_preprocessor.transform(X_valid)

# Convertir les matrices creuses en matrices denses pour Keras.
if hasattr(X_train_nn, "toarray"):
    X_train_nn = X_train_nn.toarray()
if hasattr(X_valid_nn, "toarray"):
    X_valid_nn = X_valid_nn.toarray()

X_train_nn = np.asarray(X_train_nn, dtype=np.float32)
X_valid_nn = np.asarray(X_valid_nn, dtype=np.float32)
y_train_nn = np.asarray(y_train, dtype=np.float32)
y_valid_nn = np.asarray(y_valid, dtype=np.float32)

print("Dimensions après prétraitement pour Keras :")
print("X_train_nn :", X_train_nn.shape)
print("X_valid_nn :", X_valid_nn.shape)


# In[12]:


# ============================================================
# 8) Définition du modèle et de l'espace de recherche Keras Tuner
# ============================================================

def build_regression_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train_nn.shape[1],)))

    # Nombre d'unités optimisé par Hyperband.
    units = hp.Choice("units", values=[32, 64, 128])
    model.add(tf.keras.layers.Dense(units, activation="relu"))

    # Option d'une deuxième couche cachée.
    if hp.Boolean("use_second_layer"):
        second_units = hp.Choice("second_units", values=[16, 32, 64])
        model.add(tf.keras.layers.Dense(second_units, activation="relu"))

    # Régularisation L2 optimisée.
    l2_value = hp.Choice("l2", values=[0.0, 0.0001, 0.001])
    if l2_value > 0:
        model.add(tf.keras.layers.Dense(
            1,
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(l2_value)
        ))
    else:
        model.add(tf.keras.layers.Dense(1, activation="linear"))

    learning_rate = hp.Choice(
        "learning_rate", values=[0.001, 0.0005, 0.0001]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                 tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


tuner = kt.Hyperband(
    hypermodel=build_regression_model,
    objective=kt.Objective("val_rmse", direction="min"),
    max_epochs=KERAS_MAX_EPOCHS,
    factor=3,
    hyperband_iterations=1,
    directory=str(OUTPUT_DIR / "keras_tuner"),
    project_name="course_continuous_y_regression",
    overwrite=True
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

tuner.search(
    X_train_nn,
    y_train_nn,
    validation_data=(X_valid_nn, y_valid_nn),
    epochs=KERAS_MAX_EPOCHS,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
keras_best_model = tuner.get_best_models(num_models=1)[0]

print("Meilleurs hyperparamètres Keras Tuner :")
print(best_hp.values)


# ## 9. Comparer les trois méthodes sur l'ensemble de validation
# 
# Les trois modèles sont évalués sur le même ensemble de validation, avec RMSE, MAE et R². Une valeur RMSE ou MAE plus faible indique une erreur de prédiction plus petite. Un R² plus élevé indique généralement une meilleure capacité explicative sur cet échantillon.

# In[15]:


# ============================================================
# 9) Prédictions et mesures de performance
# ============================================================

def regression_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred))
    }

grid_pred = grid_best_model.predict(X_valid)
bayes_pred = bayes_best_model.predict(X_valid)
keras_pred = keras_best_model.predict(X_valid_nn, verbose=0).reshape(-1)

results = pd.DataFrame([
    {"Méthode": "Grid Search - Random Forest", **regression_metrics(y_valid, grid_pred)},
    {"Méthode": "Bayesian Optimization - Random Forest", **regression_metrics(y_valid, bayes_pred)},
    {"Méthode": "Keras Tuner Hyperband - Neural Network", **regression_metrics(y_valid, keras_pred)}
])

results = results.sort_values("RMSE", ascending=True).reset_index(drop=True)
display(results)

best_method = results.iloc[0]["Méthode"]
print("Méthode avec le RMSE de validation le plus faible :", best_method)


# ## 10. Visualisation des prédictions
# 
# Les graphiques ci-dessous permettent d'examiner les prédictions du modèle ayant obtenu le RMSE le plus faible sur l'ensemble de validation.

# In[16]:


# ============================================================
# 10) Graphiques du meilleur modèle selon le RMSE de validation
# ============================================================

prediction_map = {
    "Grid Search - Random Forest": grid_pred,
    "Bayesian Optimization - Random Forest": bayes_pred,
    "Keras Tuner Hyperband - Neural Network": keras_pred
}

best_pred = prediction_map[best_method]

plt.figure(figsize=(7, 6))
plt.scatter(y_valid, best_pred, alpha=0.6)
min_value = min(float(np.min(y_valid)), float(np.min(best_pred)))
max_value = max(float(np.max(y_valid)), float(np.max(best_pred)))
plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
plt.xlabel("Valeurs réelles de y")
plt.ylabel("Valeurs prédites")
plt.title(f"Valeurs réelles vs prédites — {best_method}")
plt.tight_layout()
plt.show()

residuals = np.asarray(y_valid) - best_pred
plt.figure(figsize=(7, 5))
plt.scatter(best_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Valeurs prédites")
plt.ylabel("Résidus")
plt.title("Graphique des résidus")
plt.tight_layout()
plt.show()


# ## 11. Sauvegarder les résultats
# 
# Les tableaux de comparaison et les meilleurs paramètres sont sauvegardés dans le dossier `partB_outputs`. Le modèle Keras et les deux modèles Scikit-learn sont également enregistrés.

# In[17]:


# ============================================================
# 11) Sauvegarde des résultats et des modèles
# ============================================================

results.to_csv(OUTPUT_DIR / "comparison_three_methods.csv", index=False)

with open(OUTPUT_DIR / "best_parameters.json", "w", encoding="utf-8") as f:
    json.dump({
        "Grid_Search": {k: (None if v is None else str(v)) for k, v in grid_search.best_params_.items()},
        "Bayesian_Optimization": {k: (None if v is None else str(v)) for k, v in bayes_search.best_params_.items()},
        "Keras_Tuner": {k: (None if v is None else str(v)) for k, v in best_hp.values.items()}
    }, f, ensure_ascii=False, indent=4)

import joblib
joblib.dump(grid_best_model, OUTPUT_DIR / "grid_search_random_forest.joblib")
joblib.dump(bayes_best_model, OUTPUT_DIR / "bayesian_random_forest.joblib")
keras_best_model.save(OUTPUT_DIR / "keras_tuner_regression_model.keras")

print("Fichiers sauvegardés dans :", OUTPUT_DIR.resolve())
print("Notebook terminé : les trois méthodes ont été appliquées au Y continu du fichier du cours.")


# ## 12. Points à commenter dans le rapport
# 
# Dans le rapport, présenter :
# 
# 1. Le fichier de données du cours, le nombre d'observations et de variables, ainsi que la nature continue de `y`.
# 2. Les étapes de préparation des données et la séparation entraînement/validation.
# 3. Les hyperparamètres explorés par chaque méthode.
# 4. Le tableau comparatif des RMSE, MAE et R² de validation.
# 5. La méthode qui obtient le meilleur RMSE sur cet ensemble de validation, en précisant qu'il s'agit du résultat observé sur cette séparation.
# 
# Ne pas inventer les résultats : recopier les valeurs réellement obtenues après exécution du notebook.
