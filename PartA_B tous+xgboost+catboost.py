#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Load the datasets
train = pd.read_csv("/Users/tangjuanzhe/Desktop/train.csv")
test = pd.read_csv("/Users/tangjuanzhe/Desktop/test.csv")

# Display the dimensions
print("Train:", train.shape)
print("Test:", test.shape)

# Display the first rows
display(train.head())
display(test.head())


# In[2]:


# Check categorical variables
categorical_vars = ['difflevel', 'country', 'opsys', 'acquis']

for col in categorical_vars:
    print("\n", col)
    print(train[col].value_counts().sort_index())

# Check target variable distribution
print("\nTarget variable y:")
print(train['y'].value_counts().sort_index())

# Check target variable proportions
print("\nTarget variable proportions:")
print(train['y'].value_counts(normalize=True).sort_index())


# In[3]:


from sklearn.model_selection import train_test_split

# Separate features and target
X = train.drop(columns=['y'])
y = train['y']

# Split the training data
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the dimensions
print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("y_train:", y_train.shape)
print("y_valid:", y_valid.shape)


# In[4]:


print("Training set proportions:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nValidation set proportions:")
print(y_valid.value_counts(normalize=True).sort_index())


# In[5]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# In[6]:


# Train the model
rf_model.fit(X_train, y_train)


# In[7]:


# Predict the validation set
y_pred = rf_model.predict(X_valid)

# Calculate accuracy
accuracy = accuracy_score(y_valid, y_pred)

print("Random Forest Accuracy:", accuracy)


# In[8]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculate confusion matrix
cm = confusion_matrix(y_valid, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[1, 2, 3, 4]
)

disp.plot(cmap="Blues")
plt.title("Random Forest - Confusion Matrix")
plt.show()


# In[9]:


from sklearn.metrics import classification_report

print(classification_report(y_valid, y_pred))


# Model 1: Random Forest
# 
# Method: Direct multiclass classification
# Number of trees: 100
# Validation Accuracy: 69.2%
# 
# Observation:
# The model performs better on classes 1 and 2
# than on classes 3 and 4.
# Classes 3 and 4 are often confused with each other.

# In[10]:


#utiliser OVA model dans le mémoire
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create the OVA model
ova_model = OneVsRestClassifier(
    RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
)

# Train the model
ova_model.fit(X_train, y_train)

# Predict the validation set
y_pred_ova = ova_model.predict(X_valid)

# Calculate accuracy
accuracy_ova = accuracy_score(y_valid, y_pred_ova)

print("OVA Random Forest Accuracy:", accuracy_ova)


# In[11]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculate confusion matrix
cm_ova = confusion_matrix(y_valid, y_pred_ova)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_ova,
    display_labels=[1, 2, 3, 4]
)

disp.plot(cmap="Blues")
plt.title("OVA Random Forest - Confusion Matrix")
plt.show()


# In[12]:


from sklearn.metrics import classification_report
print(classification_report(y_valid, y_pred_ova))


# In[13]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Create the Boosting model
boost_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# Train the model
boost_model.fit(X_train, y_train)

# Predict the validation set
y_pred_boost = boost_model.predict(X_valid)

# Calculate accuracy
accuracy_boost = accuracy_score(y_valid, y_pred_boost)

print("Boosting Accuracy:", accuracy_boost)


# In[14]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculate confusion matrix
cm_boost = confusion_matrix(y_valid, y_pred_boost)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_boost,
    display_labels=[1, 2, 3, 4]
)

disp.plot(cmap="Blues")
plt.title("Gradient Boosting - Confusion Matrix")
plt.show()


# In[15]:


from sklearn.metrics import classification_report
print(classification_report(y_valid, y_pred_boost))


# In[16]:


# Compare the models

results = pd.DataFrame({
    'Model': [
        'Random Forest',
        'OVA + Random Forest',
        'Gradient Boosting'
    ],
    'Accuracy': [
        accuracy,
        accuracy_ova,
        accuracy_boost
    ]
})

print(results)


# 
# ## Model Comparison – Baseline Models
# 
# Three classification approaches were evaluated using the same training and validation sets.
# 
# ### 1. Methods
# 
# - **Random Forest:** Direct multiclass classification using an ensemble of decision trees.
# - **One-vs-All (OVA):** Four binary classifiers were trained, each separating one class from the remaining classes, using Random Forest as the base classifier.
# - **Gradient Boosting:** A boosting model was trained directly for multiclass classification.
# 
# ### 2. Validation Results
# 
# | Model | Accuracy |
# | Random Forest | 69.2% |
# | OVA + Random Forest | 69.5% |
# | Gradient Boosting | 73.3% |
# 
# ### 3. Analysis
# 
# Gradient Boosting achieved the highest validation accuracy (73.3%), outperforming Random Forest (69.2%) and OVA + Random Forest (69.5%).
# 
# The classification report also shows that Gradient Boosting achieved better F1-scores for all four classes than the two Random Forest approaches.
# 
# However, class 3 remains more difficult to predict than the other classes.
# 
# ### 4. Conclusion
# 
# Among the three baseline models, Gradient Boosting obtained the best overall validation performance. Therefore, it will be selected for further hyperparameter optimization.
# 

# In[17]:


# Hyperparameter Optimization
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)


# In[18]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Create the Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [2, 3]
}

# Create Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit Grid Search on the training set
grid_search.fit(X_train, y_train)

# Display the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)


# In[19]:


# Get the best model
best_gb_model = grid_search.best_estimator_

# Predict the validation set
y_pred_best = best_gb_model.predict(X_valid)

# Calculate validation accuracy
best_accuracy = accuracy_score(y_valid, y_pred_best)

print("Optimized Gradient Boosting Accuracy:", best_accuracy)


# In[20]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculate confusion matrix
cm_best = confusion_matrix(y_valid, y_pred_best)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_best,
    display_labels=[1, 2, 3, 4]
)

disp.plot(cmap="Blues")
plt.title("Optimized Gradient Boosting - Confusion Matrix")
plt.show()


# In[21]:


from sklearn.metrics import classification_report
print(classification_report(y_valid, y_pred_best))


# 
# ## Optimized Gradient Boosting – Evaluation
# 
# ### 1. Confusion Matrix
# 
# The optimized Gradient Boosting model correctly classified 761 out of 1,000 validation observations.
# 
# ### 2. Classification Performance
# 
# The model achieved:
# 
# - Accuracy: 76.1%
# - Macro-average Precision: 0.76
# - Macro-average Recall: 0.76
# - Macro-average F1-score: 0.76
# 
# Class 1 achieved the highest F1-score (0.86), while class 3 remained the most difficult class to predict (0.65).
# 
# ### 3. Comparison with the Baseline Model
# 
# Validation accuracy increased from 73.3% to 76.1%, representing an improvement of 2.8 percentage points.
# 
# F1-scores also improved for classes 2, 3 and 4, while class 1 remained unchanged.
# 
# ### 4. Conclusion
# 
# Hyperparameter optimization improved the overall classification performance of Gradient Boosting. The optimized model achieved the best validation accuracy among the models evaluated so far.
# 
# 

# In[22]:


print(train.columns.tolist())
print(train.shape)


# In[23]:


print("Test columns:")
print(test.columns.tolist())
print("Test shape:")
print(test.shape)
display(test.head())


# In[24]:


# Separate features and target
X = train.drop(columns=['y'])
y = train['y']

# Remove id from test features
X_test = test.drop(columns=['id'])

# Check feature columns
print("Training features:", X.columns.tolist())
print("Test features:", X_test.columns.tolist())

# Check whether the columns match
print("Same columns:", X.columns.tolist() == X_test.columns.tolist())

print("X shape:", X.shape)
print("X_test shape:", X_test.shape)


# In[26]:


from sklearn.ensemble import GradientBoostingClassifier

# Create the final model
final_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Train on all 5,000 observations
final_model.fit(X, y)

print("Final model training completed.")


# In[27]:


# Predict the test set
y_test_pred = final_model.predict(X_test)

# Check predictions
print("Prediction shape:", y_test_pred.shape)
print("Predicted classes:")
print(pd.Series(y_test_pred).value_counts().sort_index())


# In[28]:


# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'y': y_test_pred
})

# Save as CSV
submission.to_csv(
    '/Users/tangjuanzhe/Desktop/submission.csv',
    index=False
)

print("Submission file created successfully!")

# Check the submission file
print(submission.shape)
display(submission.head())


# In[29]:


# Check submission structure
print("Columns:", submission.columns.tolist())
print("Number of rows:", len(submission))
print("Missing values:")
print(submission.isnull().sum())

print("Predicted classes:")
print(submission['y'].value_counts().sort_index())


# In[30]:


#utiliser OVO
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create the OVO model
ovo_model = OneVsOneClassifier(
    RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
)

# Train the model
ovo_model.fit(X_train, y_train)

# Predict the validation set
y_pred_ovo = ovo_model.predict(X_valid)

# Calculate accuracy
accuracy_ovo = accuracy_score(y_valid, y_pred_ovo)

print("OVO Random Forest Accuracy:", accuracy_ovo)


# In[31]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculate confusion matrix
cm_ovo = confusion_matrix(y_valid, y_pred_ovo)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_ovo,
    display_labels=[1, 2, 3, 4]
)

disp.plot(cmap="Blues")
plt.title("OVO Random Forest - Confusion Matrix")
plt.show()


# In[32]:


from sklearn.metrics import classification_report
print(classification_report(y_valid, y_pred_ovo))


# In[3]:


#Pour améliorer le accuracy je utilise xgboost
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Load datasets
train = pd.read_csv("/Users/tangjuanzhe/Desktop/train.csv")
test = pd.read_csv("/Users/tangjuanzhe/Desktop/test.csv")

# Separate features and target
X = train.drop(columns=["y"])
y = train["y"]

X_test = test.drop(columns=["id"])

# Split training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)


# In[4]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# Train the model
xgb_model.fit(X_train, y_train - 1)

# Predict on validation set
y_pred_xgb = xgb_model.predict(X_valid) + 1

# Evaluate
accuracy_xgb = accuracy_score(y_valid, y_pred_xgb)

print("XGBoost Accuracy:", accuracy_xgb)
print(classification_report(y_valid, y_pred_xgb))


# In[5]:


#optimiser xgboost paramètre
from sklearn.model_selection import GridSearchCV

param_grid_xgb = {
    "n_estimators": [200, 300, 500],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.8]
}

grid_xgb = GridSearchCV(
    estimator=XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    ),
    param_grid=param_grid_xgb,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_xgb.fit(X_train, y_train - 1)

print("Best parameters:", grid_xgb.best_params_)
print("Best CV Accuracy:", grid_xgb.best_score_)


# In[6]:


# Get the best model
best_xgb_model = grid_xgb.best_estimator_

# Predict on validation set
y_pred_best_xgb = best_xgb_model.predict(X_valid) + 1

# Calculate accuracy
best_xgb_accuracy = accuracy_score(y_valid, y_pred_best_xgb)

print("Optimized XGBoost Accuracy:", best_xgb_accuracy)

# Classification report
print(classification_report(y_valid, y_pred_best_xgb))


# In[7]:


#OVA + XGBoost 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Create OVA + XGBoost model
ova_xgb_model = OneVsRestClassifier(
    XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
)

# Train the model
ova_xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_ova_xgb = ova_xgb_model.predict(X_valid)

# Evaluate the model
accuracy_ova_xgb = accuracy_score(y_valid, y_pred_ova_xgb)

print("OVA + XGBoost Accuracy:", accuracy_ova_xgb)
print(classification_report(y_valid, y_pred_ova_xgb))


# In[8]:


#OVO + XGBoost
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report

ovo_xgb_model = OneVsOneClassifier(
    XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
)

ovo_xgb_model.fit(X_train, y_train)

y_pred_ovo_xgb = ovo_xgb_model.predict(X_valid)

accuracy_ovo_xgb = accuracy_score(y_valid, y_pred_ovo_xgb)

print("OVO + XGBoost Accuracy:", accuracy_ovo_xgb)
print(classification_report(y_valid, y_pred_ovo_xgb))


# ### Comparaison des stratégies de classification multiclasses avec XGBoost
# 
# Nous avons comparé trois stratégies de classification multiclasses en utilisant XGBoost comme classificateur de base : la classification directe, One-vs-All (OVA) et One-vs-One (OVO).
# 
# Les résultats obtenus sur l'ensemble de validation sont présentés ci-dessous.
# 
# | Stratégie | Accuracy | Macro Precision | Macro Recall | Macro F1-score |
# |---|---:|---:|---:|---:|
# | XGBoost direct (paramètres initiaux) | 76.1% | 0.77 | 0.76 | 0.76 |
# | XGBoost direct (paramètres optimisés) | 77.2% | 0.78 | 0.77 | 0.77 |
# | XGBoost + OVA | 73.4% | 0.74 | 0.74 | 0.73 |
# | XGBoost + OVO | **79.8%** | **0.80** | **0.80** | **0.80** |
# 
# ### Analyse des résultats
# 
# La stratégie OVO obtient la meilleure performance, avec une accuracy de 79.8%, suivie de XGBoost direct avec des paramètres optimisés (77.2%). La stratégie OVA obtient la performance la plus faible, avec une accuracy de 73.4%.
# 
# Par rapport à XGBoost direct optimisé, OVO améliore l'accuracy de 2.6 points de pourcentage.
# 
# En examinant les F1-scores par classe, OVO donne également de meilleurs résultats pour les quatre classes. En particulier, le F1-score de la classe 3 passe de 0.65 avec XGBoost direct optimisé à 0.73 avec OVO. Celui de la classe 4 passe de 0.70 à 0.74.
# 
# ### Conclusion
# 
# Dans nos expériences, la stratégie OVO combinée à XGBoost donne les meilleurs résultats parmi les trois stratégies comparées. Elle permet notamment d'améliorer la classification des classes 3 et 4, qui sont plus difficiles à prédire. Ces résultats suggèrent que la décomposition du problème en plusieurs classifications binaires par paires est avantageuse pour notre jeu de données.

# In[9]:


#optimiser ovo+xgboost paramètre
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Create OVO + XGBoost
ovo_xgb_base = OneVsOneClassifier(
    XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1
    )
)

# Define parameter grid
param_grid_ovo = {
    "estimator__n_estimators": [300, 500],
    "estimator__learning_rate": [0.05, 0.1],
    "estimator__max_depth": [3, 4]
}

# Grid Search with 3-fold cross-validation
grid_ovo = GridSearchCV(
    estimator=ovo_xgb_base,
    param_grid=param_grid_ovo,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_ovo.fit(X_train, y_train)

print("Best parameters:", grid_ovo.best_params_)
print("Best CV Accuracy:", grid_ovo.best_score_)


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
train = pd.read_csv("/Users/tangjuanzhe/Desktop/train.csv")

# Separate features and target
X = train.drop(columns=["y"])
y = train["y"]

# Split data
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


# In[5]:


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=30)

print("Best parameters:", study.best_params)
print("Best CV Accuracy:", study.best_value)


# In[6]:


from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Create optimized OVO + XGBoost
best_ovo_xgb = OneVsOneClassifier(
    XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=600,
        learning_rate=0.09492962821365351,
        max_depth=4,
        subsample=0.8205765095424473,
        colsample_bytree=0.8232659142660376,
        random_state=42,
        n_jobs=1
    )
)

# Train model
best_ovo_xgb.fit(X_train, y_train)

# Predict
y_pred_bayes = best_ovo_xgb.predict(X_valid)

# Evaluate
accuracy_bayes = accuracy_score(y_valid, y_pred_bayes)

print("Bayesian Optimized OVO + XGBoost Accuracy:", accuracy_bayes)
print(classification_report(y_valid, y_pred_bayes))


# ### Bayesian Optimization pour OVO + XGBoost
# 
# Nous avons utilisé l'optimisation bayésienne avec Optuna et l'échantillonneur TPE afin de rechercher de meilleurs hyperparamètres pour le modèle OVO + XGBoost.
# 
# L'optimisation a été réalisée avec 30 essais et une validation croisée à 3 plis.
# 
# Les meilleurs hyperparamètres obtenus sont :
# 
# - n_estimators = 600
# - learning_rate = 0.0949
# - max_depth = 4
# - subsample = 0.8206
# - colsample_bytree = 0.8233
# 
# La meilleure accuracy moyenne en validation croisée est de 79.50%.
# 
# Sur l'ensemble de validation indépendant, le modèle optimisé obtient une accuracy de 79.5%, contre 79.8% pour le modèle initial.
# 
# Ainsi, l'optimisation bayésienne n'a pas amélioré la performance du modèle dans cette expérience. Le modèle OVO + XGBoost avec les paramètres initiaux demeure le meilleur modèle testé.

# In[9]:


# Predict using the Bayesian optimized OVO + XGBoost model
y_pred_bayes = best_ovo_xgb.predict(X_valid)

# Calculate confusion matrix
cm_best = confusion_matrix(y_valid, y_pred_bayes)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_best,
    display_labels=[1, 2, 3, 4]
)

disp.plot(cmap="Blues")
plt.title("Bayesian Optimized OVO + XGBoost - Confusion Matrix")
plt.show()


# In[11]:


# Reload test dataset
test = pd.read_csv("/Users/tangjuanzhe/Desktop/test.csv")

# Remove id because it is not a predictive feature
X_test = test.drop(columns=["id"])

# Check the data
print("Test data shape:", test.shape)
print("X_test shape:", X_test.shape)
print("Feature columns match:", list(X_test.columns) == list(X.columns))


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load training data
train = pd.read_csv("/Users/tangjuanzhe/Desktop/train.csv")

# Separate features and target
X = train.drop(columns=["y"])
y = train["y"]

# Split training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)


# In[5]:


categorical_features = [
    "difflevel",
    "country",
    "opsys",
    "acquis"
]

print("Categorical features:", categorical_features)


# In[6]:


from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=100
)

cat_model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=100,
    use_best_model=True
)


# In[7]:


from sklearn.metrics import accuracy_score, classification_report

y_pred_cat = cat_model.predict(X_valid).ravel()

accuracy_cat = accuracy_score(y_valid, y_pred_cat)

print("CatBoost Accuracy:", round(accuracy_cat, 4))
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_cat))


# In[8]:


from sklearn.model_selection import GridSearchCV

# Create CatBoost model
cat_grid_model = CatBoostClassifier(
    loss_function="MultiClass",
    random_seed=42,
    verbose=0,
    thread_count=1
)

# Define parameter grid
param_grid_cat = {
    "iterations": [500, 1000],
    "learning_rate": [0.03, 0.05, 0.1],
    "depth": [4, 6, 8]
}

print("Number of parameter combinations:", 18)


# In[ ]:


cat_grid = GridSearchCV(
    estimator=cat_grid_model,
    param_grid=param_grid_cat,
    scoring="accuracy",
    cv=3,
    n_jobs=1,
    verbose=1
)

cat_grid.fit(
    X_train,
    y_train,
    cat_features=categorical_features
)

print("Best parameters:", cat_grid.best_params_)
print("Best CV Accuracy:", round(cat_grid.best_score_, 4))


# In[1]:


import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Define the objective function
def objective_cat(trial):

    params = {
        "iterations": trial.suggest_int("iterations", 300, 1500, step=100),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2, log=True
        ),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0, 2),
        "loss_function": "MultiClass",
        "random_seed": 42,
        "verbose": 0,
        "thread_count": 1
    }

    model = CatBoostClassifier(**params)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=3,
        scoring="accuracy",
        n_jobs=1
    )

    return scores.mean()

print("CatBoost Bayesian Optimization objective ready!")


# In[4]:


import pandas as pd
from pathlib import Path

# Find the Desktop folder
desktop = Path.home() / "Desktop"

# Check whether the files exist
train_path = desktop / "train.csv"
test_path = desktop / "test.csv"

print("Train file exists:", train_path.exists())
print("Test file exists:", test_path.exists())

# Load the datasets
if train_path.exists() and test_path.exists():
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("\nTrain shape:", train.shape)
    print("Test shape:", test.shape)
    print("\nColumns:", train.columns.tolist())
else:
    print("Files not found. Please check the Desktop folder.")


# In[8]:


# Training features and target
X_train = train.drop(columns=["y"])
y_train = train["y"]

# Test features: remove id from model inputs
X_test = test.drop(columns=["id"])

# Keep IDs for Kaggle submission
test_ids = test["id"]

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)

print("\nTarget distribution:")
print(y_train.value_counts().sort_index())

print("\nMissing values in training data:")
print(X_train.isnull().sum())

print("\nMissing values in test data:")
print(X_test.isnull().sum())


# In[9]:


# Identify categorical features
cat_features = ["difflevel", "country", "opsys", "acquis"]

# Convert categorical columns to strings
for col in cat_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

print("Categorical features:", cat_features)

print("\nTraining data types:")
print(X_train[cat_features].dtypes)

print("\nData preparation completed!")


# In[10]:


from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
import optuna

# Define the optimization objective
def objective_cat(trial):

    params = {
        "iterations": trial.suggest_int("iterations", 500, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0, 2),
        "loss_function": "MultiClass",
        "verbose": 0,
        "thread_count": 1,
        "random_seed": 42
    }

    model = CatBoostClassifier(**params)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=3,
        scoring="accuracy",
        n_jobs=1
    )

    return scores.mean()

print("CatBoost optimization function is ready!")


# In[11]:


# Create an Optuna study
study_cat = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

# Run 30 optimization trials
study_cat.optimize(
    objective_cat,
    n_trials=30
)

# Display the best results
print("\nBest cross-validation accuracy:")
print(study_cat.best_value)

print("\nBest parameters:")
print(study_cat.best_params)


# In[12]:


# Train CatBoost using the best parameters all datas

best_model = CatBoostClassifier(
    **study_cat.best_params,
    loss_function="MultiClass",
    verbose=100,
    random_seed=42
)

best_model.fit(X_train, y_train)

# Predict Kaggle test data
test_predictions = best_model.predict(X_test)

# Convert predictions to integers
test_predictions = test_predictions.flatten().astype(int)

print("Prediction completed!")
print("Number of predictions:", len(test_predictions))
print("Predicted classes:")
print(pd.Series(test_predictions).value_counts().sort_index())


# In[13]:


# Create Kaggle submission file

submission = pd.DataFrame({
    "id": test_ids,
    "y": test_predictions
})

# Save submission file to Desktop
output_path = desktop / "submission_catboost_optuna.csv"
submission.to_csv(output_path, index=False)

print("Submission file saved!")
print("File:", output_path)

print("\nSubmission preview:")
display(submission.head())

print("\nSubmission shape:", submission.shape)


# In[16]:


# Check whether trained models are still available in the notebook
for name, obj in list(globals().items()):
    class_name = type(obj).__name__

    if any(key in class_name.lower() for key in
           ["catboost", "xgb", "onevsone", "onevsrest", "voting"]):
        print(f"Variable: {name} | Model type: {class_name}")


# In[17]:


# Check whether each model supports probability prediction
models = {
    "CatBoost": best_model,
    "XGBoost": xgb_model,
    "XGBoost OVO": ovo_model
}

for name, model in models.items():
    print(f"{name}: {hasattr(model, 'predict_proba')}")


# In[19]:


from sklearn.utils.validation import check_is_fitted

# Check the fitted status of each model
models = {
    "CatBoost": best_model,
    "XGBoost": xgb_model,
    "XGBoost OVO": ovo_model
}

for name, model in models.items():
    try:
        check_is_fitted(model)
        print(f"{name}: Fitted")
    except Exception:
        print(f"{name}: Not fitted")

# Check available training and validation datasets
data_names = [
    "X_train", "X_val", "y_train", "y_val",
    "X_train_xgb", "X_test_xgb"
]

for name in data_names:
    if name in globals():
        obj = globals()[name]
        print(name, obj.shape)


# In[20]:


from sklearn.model_selection import train_test_split

# Create a consistent training and validation split
X = train.drop(columns=["y"]).copy()
y = train["y"].copy()

X_train_blend, X_val_blend, y_train_blend, y_val_blend = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set:", X_train_blend.shape)
print("Validation set:", X_val_blend.shape)
print("Training labels:", y_train_blend.shape)
print("Validation labels:", y_val_blend.shape)


# In[21]:


from xgboost import XGBClassifier

# Train XGBoost using the selected hyperparameters
xgb_blend = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_blend.fit(X_train_blend, y_train_blend - 1)

print("XGBoost training completed.")


# In[22]:


from sklearn.metrics import accuracy_score

# Generate predictions on the validation set
cat_pred = best_model.predict(X_val_blend).flatten()
xgb_pred = xgb_blend.predict(X_val_blend) + 1

# Calculate validation accuracy
cat_accuracy = accuracy_score(y_val_blend, cat_pred)
xgb_accuracy = accuracy_score(y_val_blend, xgb_pred)

print(f"CatBoost accuracy: {cat_accuracy:.4%}")
print(f"XGBoost accuracy: {xgb_accuracy:.4%}")


# In[23]:


# Display the current CatBoost model parameters
print(best_model.get_params())


# In[24]:


from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Identify categorical features
cat_features = ["difflevel", "country", "opsys", "acquis"]

# Prepare training and validation data
X_train_cat = X_train_blend.copy()
X_val_cat = X_val_blend.copy()

for col in cat_features:
    X_train_cat[col] = X_train_cat[col].astype(str)
    X_val_cat[col] = X_val_cat[col].astype(str)

# Train CatBoost using the best Optuna parameters
cat_blend = CatBoostClassifier(
    iterations=1100,
    learning_rate=0.02682442119315144,
    depth=5,
    l2_leaf_reg=9.041718988294106,
    loss_function="MultiClass",
    random_seed=42,
    random_strength=0.1860260920815992,
    verbose=100
)

cat_blend.fit(
    X_train_cat,
    y_train_blend,
    cat_features=cat_features
)

# Evaluate CatBoost on the validation set
cat_pred_blend = cat_blend.predict(X_val_cat).flatten()
cat_accuracy_blend = accuracy_score(y_val_blend, cat_pred_blend)

print(f"CatBoost validation accuracy: {cat_accuracy_blend:.4%}")


# In[25]:


import numpy as np
from sklearn.metrics import accuracy_score

# Generate prediction probabilities
cat_proba = cat_blend.predict_proba(X_val_cat)
xgb_proba = xgb_blend.predict_proba(X_val_blend)

# Test different probability blending weights
results = []

for cat_weight in np.arange(0, 1.01, 0.1):
    xgb_weight = 1 - cat_weight

    # Combine prediction probabilities
    blended_proba = (
        cat_weight * cat_proba +
        xgb_weight * xgb_proba
    )

    # Convert probabilities into class predictions
    blended_pred = np.argmax(blended_proba, axis=1) + 1

    # Calculate validation accuracy
    accuracy = accuracy_score(y_val_blend, blended_pred)

    results.append((cat_weight, xgb_weight, accuracy))

# Display results sorted by accuracy
results.sort(key=lambda x: x[2], reverse=True)

for cat_weight, xgb_weight, accuracy in results:
    print(
        f"CatBoost: {cat_weight:.1f} | "
        f"XGBoost: {xgb_weight:.1f} | "
        f"Accuracy: {accuracy:.4%}"
    )


# In[26]:


# Test finer blending weights near the current best result
results_fine = []

for cat_weight in np.arange(0.85, 1.001, 0.025):
    xgb_weight = 1 - cat_weight

    # Combine prediction probabilities
    blended_proba = (
        cat_weight * cat_proba +
        xgb_weight * xgb_proba
    )

    # Generate class predictions
    blended_pred = np.argmax(blended_proba, axis=1) + 1

    # Calculate validation accuracy
    accuracy = accuracy_score(y_val_blend, blended_pred)

    results_fine.append((cat_weight, xgb_weight, accuracy))

# Sort results by accuracy
results_fine.sort(key=lambda x: x[2], reverse=True)

for cat_weight, xgb_weight, accuracy in results_fine:
    print(
        f"CatBoost: {cat_weight:.3f} | "
        f"XGBoost: {xgb_weight:.3f} | "
        f"Accuracy: {accuracy:.4%}"
    )


# In[27]:


from sklearn.metrics import classification_report, confusion_matrix

# Generate predictions using the best blending ratio
best_blended_proba = (
    0.85 * cat_proba +
    0.15 * xgb_proba
)

best_blended_pred = np.argmax(best_blended_proba, axis=1) + 1

# Display classification performance for each class
print("Classification Report:")
print(classification_report(y_val_blend, best_blended_pred))

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_val_blend, best_blended_pred))


# In[28]:


import pandas as pd

# Calculate CatBoost feature importance
feature_importance = pd.DataFrame({
    "Feature": X_train_cat.columns,
    "Importance": cat_blend.get_feature_importance()
})

# Sort features by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.to_string(index=False))


# In[29]:


import numpy as np

# Create new training and validation datasets
X_train_fe = X_train_blend.copy()
X_val_fe = X_val_blend.copy()

# Calculate average purchase amount per purchase
X_train_fe["avg_purchase"] = np.divide(
    X_train_fe["totpurchases"],
    X_train_fe["numpurchases"],
    out=np.zeros(len(X_train_fe)),
    where=X_train_fe["numpurchases"] > 0
)

X_val_fe["avg_purchase"] = np.divide(
    X_val_fe["totpurchases"],
    X_val_fe["numpurchases"],
    out=np.zeros(len(X_val_fe)),
    where=X_val_fe["numpurchases"] > 0
)

# Calculate average playtime per session
X_train_fe["avg_playtime"] = np.divide(
    X_train_fe["totplaytime"],
    X_train_fe["numsessions"],
    out=np.zeros(len(X_train_fe)),
    where=X_train_fe["numsessions"] > 0
)

X_val_fe["avg_playtime"] = np.divide(
    X_val_fe["totplaytime"],
    X_val_fe["numsessions"],
    out=np.zeros(len(X_val_fe)),
    where=X_val_fe["numsessions"] > 0
)

# Calculate average sessions per day
X_train_fe["sessions_per_day"] = np.divide(
    X_train_fe["numsessions"],
    X_train_fe["numdays"],
    out=np.zeros(len(X_train_fe)),
    where=X_train_fe["numdays"] > 0
)

X_val_fe["sessions_per_day"] = np.divide(
    X_val_fe["numsessions"],
    X_val_fe["numdays"],
    out=np.zeros(len(X_val_fe)),
    where=X_val_fe["numdays"] > 0
)

print("Original features:", X_train_blend.shape[1])
print("New features:", X_train_fe.shape[1])
print("Training shape:", X_train_fe.shape)
print("Validation shape:", X_val_fe.shape)


# In[30]:


from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Identify categorical features
cat_features = ["difflevel", "country", "opsys", "acquis"]

# Convert categorical features to strings
X_train_fe_cat = X_train_fe.copy()
X_val_fe_cat = X_val_fe.copy()

for col in cat_features:
    X_train_fe_cat[col] = X_train_fe_cat[col].astype(str)
    X_val_fe_cat[col] = X_val_fe_cat[col].astype(str)

# Train CatBoost with engineered features
cat_fe_model = CatBoostClassifier(
    iterations=1100,
    learning_rate=0.02682442119315144,
    depth=5,
    l2_leaf_reg=9.041718988294106,
    loss_function="MultiClass",
    random_seed=42,
    random_strength=0.1860260920815992,
    verbose=100
)

cat_fe_model.fit(
    X_train_fe_cat,
    y_train_blend,
    cat_features=cat_features
)

# Evaluate the model
cat_fe_pred = cat_fe_model.predict(X_val_fe_cat).flatten()
cat_fe_accuracy = accuracy_score(y_val_blend, cat_fe_pred)

print(f"CatBoost with feature engineering: {cat_fe_accuracy:.4%}")


# In[31]:


# Select the eight most important original features
selected_features = [
    "totpurchases",
    "trendpurchase",
    "country",
    "numpurchases",
    "numsessions",
    "acquis",
    "trendsession",
    "opsys"
]

# Create reduced training and validation datasets
X_train_selected = X_train_blend[selected_features].copy()
X_val_selected = X_val_blend[selected_features].copy()

# Convert categorical features to strings
cat_features_selected = ["country", "acquis", "opsys"]

for col in cat_features_selected:
    X_train_selected[col] = X_train_selected[col].astype(str)
    X_val_selected[col] = X_val_selected[col].astype(str)

print("Selected features:", X_train_selected.columns.tolist())
print("Training shape:", X_train_selected.shape)
print("Validation shape:", X_val_selected.shape)


# In[32]:


from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Train CatBoost using the selected features
cat_selected_model = CatBoostClassifier(
    iterations=1100,
    learning_rate=0.02682442119315144,
    depth=5,
    l2_leaf_reg=9.041718988294106,
    loss_function="MultiClass",
    random_seed=42,
    random_strength=0.1860260920815992,
    verbose=100
)

cat_selected_model.fit(
    X_train_selected,
    y_train_blend,
    cat_features=cat_features_selected
)

# Evaluate the model on the validation set
selected_pred = cat_selected_model.predict(X_val_selected).flatten()
selected_accuracy = accuracy_score(y_val_blend, selected_pred)

print(f"CatBoost with selected features: {selected_accuracy:.4%}")


# In[33]:


# Select ten features for the next experiment
selected_features_10 = selected_features + ["skill1", "skill2"]

# Create training and validation datasets
X_train_10 = X_train_blend[selected_features_10].copy()
X_val_10 = X_val_blend[selected_features_10].copy()

# Convert categorical features to strings
cat_features_10 = ["country", "acquis", "opsys"]

for col in cat_features_10:
    X_train_10[col] = X_train_10[col].astype(str)
    X_val_10[col] = X_val_10[col].astype(str)

print("Training shape:", X_train_10.shape)
print("Validation shape:", X_val_10.shape)


# In[34]:


from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Train CatBoost using ten selected features
cat_10_model = CatBoostClassifier(
    iterations=1100,
    learning_rate=0.02682442119315144,
    depth=5,
    l2_leaf_reg=9.041718988294106,
    loss_function="MultiClass",
    random_seed=42,
    random_strength=0.1860260920815992,
    verbose=100
)

cat_10_model.fit(
    X_train_10,
    y_train_blend,
    cat_features=cat_features_10
)

# Evaluate the model on the validation set
pred_10 = cat_10_model.predict(X_val_10).flatten()
accuracy_10 = accuracy_score(y_val_blend, pred_10)

print(f"CatBoost with ten features: {accuracy_10:.4%}")


# In[35]:


from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Train CatBoost with eight features and deeper trees
cat_depth6 = CatBoostClassifier(
    iterations=1100,
    learning_rate=0.02682442119315144,
    depth=6,
    l2_leaf_reg=9.041718988294106,
    loss_function="MultiClass",
    random_seed=42,
    random_strength=0.1860260920815992,
    verbose=100
)

cat_depth6.fit(
    X_train_selected,
    y_train_blend,
    cat_features=cat_features_selected
)

# Evaluate the model
pred_depth6 = cat_depth6.predict(X_val_selected).flatten()
accuracy_depth6 = accuracy_score(y_val_blend, pred_depth6)

print(f"CatBoost with depth=6: {accuracy_depth6:.4%}")


# In[36]:


from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Train CatBoost with stronger L2 regularization
cat_l2_15 = CatBoostClassifier(
    iterations=1100,
    learning_rate=0.02682442119315144,
    depth=5,
    l2_leaf_reg=15,
    loss_function="MultiClass",
    random_seed=42,
    random_strength=0.1860260920815992,
    verbose=100
)

cat_l2_15.fit(
    X_train_selected,
    y_train_blend,
    cat_features=cat_features_selected
)

# Evaluate the model
pred_l2_15 = cat_l2_15.predict(X_val_selected).flatten()
accuracy_l2_15 = accuracy_score(y_val_blend, pred_l2_15)

print(f"CatBoost with l2_leaf_reg=15: {accuracy_l2_15:.4%}")


# In[38]:


from sklearn.metrics import classification_report, confusion_matrix

# Generate predictions using the best eight-feature model
best_selected_pred = cat_selected_model.predict(X_val_selected).flatten()

# Display classification metrics
print("Classification Report:")
print(classification_report(y_val_blend, best_selected_pred))

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_val_blend, best_selected_pred))


# In[39]:


import pandas as pd

# Extract feature importance
importance_df = pd.DataFrame({
    "Feature": X_train_selected.columns,
    "Importance": cat_selected_model.get_feature_importance()
})

# Sort features by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.to_string(index=False))


# In[40]:


from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Remove country from the selected features
features_no_country = [
    col for col in selected_features
    if col != "country"
]

# Prepare training and validation datasets
X_train_no_country = X_train_blend[features_no_country].copy()
X_val_no_country = X_val_blend[features_no_country].copy()

# Convert categorical features to strings
cat_features_no_country = ["acquis", "opsys"]

for col in cat_features_no_country:
    X_train_no_country[col] = X_train_no_country[col].astype(str)
    X_val_no_country[col] = X_val_no_country[col].astype(str)

# Train CatBoost without country
cat_no_country = CatBoostClassifier(
    iterations=1100,
    learning_rate=0.02682442119315144,
    depth=5,
    l2_leaf_reg=9.041718988294106,
    loss_function="MultiClass",
    random_seed=42,
    random_strength=0.1860260920815992,
    verbose=100
)

cat_no_country.fit(
    X_train_no_country,
    y_train_blend,
    cat_features=cat_features_no_country
)

# Evaluate the model
pred_no_country = cat_no_country.predict(X_val_no_country).flatten()
accuracy_no_country = accuracy_score(y_val_blend, pred_no_country)

print(f"CatBoost without country: {accuracy_no_country:.4%}")


# In[41]:


# Compare purchase-related features across target classes
purchase_summary = train.groupby("y")[
    ["totpurchases", "trendpurchase", "numpurchases"]
].mean().round(2)

print(purchase_summary)


# In[42]:


import numpy as np

# Create training and validation datasets
X_train_avg = X_train_blend[selected_features].copy()
X_val_avg = X_val_blend[selected_features].copy()

# Calculate average purchase amount
X_train_avg["avg_purchase"] = np.divide(
    X_train_avg["totpurchases"],
    X_train_avg["numpurchases"],
    out=np.zeros(len(X_train_avg), dtype=float),
    where=X_train_avg["numpurchases"].to_numpy() != 0
)

X_val_avg["avg_purchase"] = np.divide(
    X_val_avg["totpurchases"],
    X_val_avg["numpurchases"],
    out=np.zeros(len(X_val_avg), dtype=float),
    where=X_val_avg["numpurchases"].to_numpy() != 0
)

# Convert categorical features to strings
for col in cat_features_selected:
    X_train_avg[col] = X_train_avg[col].astype(str)
    X_val_avg[col] = X_val_avg[col].astype(str)

print("Training shape:", X_train_avg.shape)
print("Validation shape:", X_val_avg.shape)


# 
# ## Model Optimization: CatBoost
# 
# ### 1. Initial Model Evaluation
# 
# We first evaluated several models using a fixed stratified train-validation split (80% training and 20% validation). The dataset contains 5,000 observations and four balanced target classes.
# 
# The initial CatBoost model achieved 77.6% validation accuracy. We also tested a probability-based ensemble of CatBoost and XGBoost, which achieved 79.5% validation accuracy.
# 
# ### 2. Feature Engineering
# 
# We created three additional features:
# 
# - `avg_purchase`: average purchase amount.
# - `avg_playtime`: average playtime per session.
# - `sessions_per_day`: average number of sessions per day.
# 
# However, the model with all three additional features achieved only 76.9%. Therefore, adding more features does not necessarily improve model performance.
# 
# ### 3. Feature Selection
# 
# We tested a smaller feature set based on CatBoost feature importance.
# 
# The eight selected features were:
# 
# - `totpurchases`
# - `trendpurchase`
# - `country`
# - `numpurchases`
# - `numsessions`
# - `acquis`
# - `trendsession`
# - `opsys`
# 
# This model achieved 78.6% validation accuracy, outperforming the original CatBoost model.
# 
# Adding `skill1` and `skill2` reduced accuracy to 77.7%.
# 
# Removing `country` reduced accuracy to 69.1%, indicating that this feature provides important predictive information.
# 
# ### 4. Hyperparameter Experiments
# 
# We tested different model configurations:
# 
# | Experiment | Validation Accuracy |
# |---|---:|
# | Eight features, depth = 5, 1100 iterations | 78.6% |
# | Eight features, depth = 6 | 78.0% |
# | Eight features, L2 = 15 | 77.3% |
# | Eight features, 700 iterations | 76.4% |
# 
# The original configuration with depth = 5 and 1100 iterations performed best among these experiments.
# 
# ### 5. Classification Analysis
# 
# The eight-feature model achieved:
# 
# - Class 1 F1-score: 0.89
# - Class 2 F1-score: 0.84
# - Class 3 F1-score: 0.67
# - Class 4 F1-score: 0.75
# 
# The main classification errors occurred between classes 2, 3, and 4.
# 
# We also compared the average purchase-related feature values across the four classes. Classes 2 and 3 showed relatively similar purchasing behavior, which may contribute to their classification confusion.
# 
# ### 6. Current Optimization Direction
# 
# Our next experiment introduces only one additional feature, `avg_purchase`, calculated as:
# 
# `totpurchases / numpurchases`
# 
# The objective is to determine whether average purchase amount provides additional information for distinguishing the four classes.
# 
# We will compare its validation accuracy with the existing eight-feature model.
# 

# In[43]:


from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Train CatBoost with the additional average purchase feature
cat_avg_model = CatBoostClassifier(
    iterations=1100,
    learning_rate=0.02682442119315144,
    depth=5,
    l2_leaf_reg=9.041718988294106,
    loss_function="MultiClass",
    random_seed=42,
    random_strength=0.1860260920815992,
    verbose=100
)

cat_avg_model.fit(
    X_train_avg,
    y_train_blend,
    cat_features=cat_features_selected
)

# Evaluate the model
pred_avg = cat_avg_model.predict(X_val_avg).flatten()
accuracy_avg = accuracy_score(y_val_blend, pred_avg)

print(f"CatBoost with avg_purchase: {accuracy_avg:.4%}")


# In[44]:


from xgboost import XGBClassifier

# Prepare numeric training and validation datasets
X_train_xgb8 = X_train_blend[selected_features].copy()
X_val_xgb8 = X_val_blend[selected_features].copy()

# Train XGBoost using eight selected features
xgb_8_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_8_model.fit(
    X_train_xgb8,
    y_train_blend - 1
)

# Evaluate XGBoost
xgb_8_pred = xgb_8_model.predict(X_val_xgb8) + 1
xgb_8_accuracy = accuracy_score(y_val_blend, xgb_8_pred)

print(f"XGBoost with eight features: {xgb_8_accuracy:.4%}")


# In[45]:


# Obtain class probabilities from both models
cat_proba_8 = cat_selected_model.predict_proba(X_val_selected)
xgb_proba_8 = xgb_8_model.predict_proba(X_val_xgb8)

# Test different CatBoost and XGBoost weights
best_accuracy_8 = 0
best_cat_weight_8 = 0

for cat_weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    xgb_weight = 1 - cat_weight

    # Combine predicted probabilities
    blended_proba = (
        cat_weight * cat_proba_8
        + xgb_weight * xgb_proba_8
    )

    # Convert probabilities into class predictions
    blended_pred = blended_proba.argmax(axis=1) + 1

    # Evaluate the blended model
    blended_accuracy = accuracy_score(
        y_val_blend,
        blended_pred
    )

    print(
        f"CatBoost weight: {cat_weight:.1f}, "
        f"XGBoost weight: {xgb_weight:.1f}, "
        f"Accuracy: {blended_accuracy:.4%}"
    )

    if blended_accuracy > best_accuracy_8:
        best_accuracy_8 = blended_accuracy
        best_cat_weight_8 = cat_weight

print("\nBest CatBoost weight:", best_cat_weight_8)
print("Best XGBoost weight:", 1 - best_cat_weight_8)
print(f"Best blended accuracy: {best_accuracy_8:.4%}")


# In[46]:


from sklearn.metrics import classification_report, confusion_matrix

# Generate predictions from the eight-feature ensemble
blend_8_proba = (
    0.7 * cat_proba_8
    + 0.3 * xgb_proba_8
)

blend_8_pred = blend_8_proba.argmax(axis=1) + 1

# Display classification report
print("Eight-feature ensemble:")
print(classification_report(y_val_blend, blend_8_pred))

# Display confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_val_blend, blend_8_pred))


# In[47]:


from sklearn.metrics import classification_report, confusion_matrix

# Generate XGBoost predictions
xgb_8_pred = xgb_8_model.predict(X_val_xgb8) + 1

# Display classification report
print("XGBoost classification report:")
print(classification_report(y_val_blend, xgb_8_pred))

# Display confusion matrix
print("XGBoost confusion matrix:")
print(confusion_matrix(y_val_blend, xgb_8_pred))


# In[48]:


from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Prepare the eight-feature dataset
X_cv = train[selected_features].copy()
y_cv = train["y"].copy()

# Convert categorical features to strings
for col in cat_features_selected:
    X_cv[col] = X_cv[col].astype(str)

# Define the CatBoost model
cat_cv_model = CatBoostClassifier(
    iterations=1100,
    learning_rate=0.02682442119315144,
    depth=5,
    l2_leaf_reg=9.041718988294106,
    loss_function="MultiClass",
    random_seed=42,
    random_strength=0.1860260920815992,
    verbose=0,
    allow_writing_files=False
)

# Create stratified five-fold cross-validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Evaluate the model
cv_scores = cross_val_score(
    cat_cv_model,
    X_cv,
    y_cv,
    cv=skf,
    scoring="accuracy",
    n_jobs=1
)

print("Cross-validation scores:", cv_scores)
print(f"Mean CV accuracy: {cv_scores.mean():.4%}")
print(f"Standard deviation: {cv_scores.std():.4%}")


# In[49]:


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# Define the XGBoost model
xgb_base = XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# Define hyperparameter search space
param_dist = {
    "n_estimators": [400, 600, 800, 1000],
    "learning_rate": [0.03, 0.05, 0.07, 0.1],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9]
}

# Define stratified cross-validation
xgb_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Search for better hyperparameters
xgb_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=20,
    scoring="accuracy",
    cv=xgb_cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Run hyperparameter optimization
xgb_search.fit(
    X_train_xgb8,
    y_train_blend - 1
)

print("Best parameters:")
print(xgb_search.best_params_)

print(f"Best CV accuracy: {xgb_search.best_score_:.4%}")


# In[50]:


from sklearn.metrics import accuracy_score
import numpy as np

# Get the best tuned XGBoost model
xgb_tuned = xgb_search.best_estimator_

# Predict probabilities on the validation set
xgb_tuned_proba = xgb_tuned.predict_proba(X_val_xgb8)

# Evaluate XGBoost alone
xgb_tuned_pred = xgb_tuned.predict(X_val_xgb8) + 1
xgb_tuned_acc = accuracy_score(y_val_blend, xgb_tuned_pred)

print(f"Tuned XGBoost validation accuracy: {xgb_tuned_acc:.4%}")

# Test different CatBoost and XGBoost blending weights
best_accuracy = 0
best_cat_weight = 0

for cat_weight in np.arange(0, 1.01, 0.05):
    xgb_weight = 1 - cat_weight

    blended_proba = (
        cat_weight * cat_proba_8
        + xgb_weight * xgb_tuned_proba
    )

    blended_pred = np.argmax(blended_proba, axis=1) + 1
    accuracy = accuracy_score(y_val_blend, blended_pred)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_cat_weight = cat_weight

print(f"Best CatBoost weight: {best_cat_weight:.2f}")
print(f"Best XGBoost weight: {1 - best_cat_weight:.2f}")
print(f"Best blended validation accuracy: {best_accuracy:.4%}")


# In[51]:


from sklearn.metrics import confusion_matrix, classification_report

# Use the best previous 8-feature ensemble
best_ensemble_proba = (
    0.7 * cat_proba_8
    + 0.3 * xgb_proba_8
)

best_ensemble_pred = np.argmax(best_ensemble_proba, axis=1) + 1

print("Confusion Matrix:")
print(confusion_matrix(y_val_blend, best_ensemble_pred))

print("\nClassification Report:")
print(classification_report(y_val_blend, best_ensemble_pred))


# In[52]:


# Compare prediction probabilities for each true class
for class_label in [1, 2, 3, 4]:
    mask = (y_val_blend == class_label)

    avg_proba = best_ensemble_proba[mask].mean(axis=0)

    print(f"\nTrue class {class_label}:")
    for i, prob in enumerate(avg_proba):
        print(f"Predicted class {i + 1}: {prob:.4f}")


# In[55]:


# Search blending weights with smaller intervals
best_accuracy = 0
best_cat_weight = 0

for cat_weight in np.arange(0.50, 1.00, 0.01):
    xgb_weight = 1 - cat_weight

    blended_proba = (
        cat_weight * cat_proba_8
        + xgb_weight * xgb_proba_8
    )

    blended_pred = np.argmax(blended_proba, axis=1) + 1
    accuracy = accuracy_score(y_val_blend, blended_pred)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_cat_weight = cat_weight

print(f"Best CatBoost weight: {best_cat_weight:.2f}")
print(f"Best XGBoost weight: {1 - best_cat_weight:.2f}")
print(f"Best validation accuracy: {best_accuracy:.4%}")


# In[58]:


import numpy as np
import pandas as pd
import lightgbm as lgb

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

# Define the selected features
selected_features = [
    "totpurchases",
    "trendpurchase",
    "country",
    "numpurchases",
    "numsessions",
    "acquis",
    "trendsession",
    "opsys"
]

cat_features = ["country", "acquis", "opsys"]

# Prepare the training data
X_cb = X_train_blend[selected_features].copy()
X_xgb = X_train_blend[selected_features].copy()
X_lgb = X_train_blend[selected_features].copy()

# Convert categorical columns for CatBoost
for col in cat_features:
    X_cb[col] = X_cb[col].astype(str)

# Set consistent categorical levels for LightGBM
for col in cat_features:
    categories = X_lgb[col].dropna().unique()
    X_lgb[col] = pd.Categorical(
        X_lgb[col],
        categories=categories
    )

# Define 5-fold cross-validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Create out-of-fold probability arrays
n_samples = len(X_train_blend)

oof_cat = np.zeros((n_samples, 4))
oof_xgb = np.zeros((n_samples, 4))
oof_lgb = np.zeros((n_samples, 4))

# Train the three base models using 5 folds
for fold, (train_idx, valid_idx) in enumerate(
    skf.split(X_cb, y_train_blend), 1
):
    print(f"Training fold {fold}/5...")

    # Split fold data
    y_fold_train = y_train_blend.iloc[train_idx].to_numpy()

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=1100,
        learning_rate=0.02682442119315144,
        depth=5,
        l2_leaf_reg=9.041718988294106,
        random_strength=0.1860260920815992,
        loss_function="MultiClass",
        verbose=False,
        random_seed=42
    )

    cat_model.fit(
        X_cb.iloc[train_idx],
        y_fold_train,
        cat_features=cat_features
    )

    oof_cat[valid_idx] = cat_model.predict_proba(
        X_cb.iloc[valid_idx]
    )

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(
        X_xgb.iloc[train_idx],
        y_fold_train - 1
    )

    oof_xgb[valid_idx] = xgb_model.predict_proba(
        X_xgb.iloc[valid_idx]
    )

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    lgb_model.fit(
        X_lgb.iloc[train_idx],
        y_fold_train - 1,
        categorical_feature=cat_features
    )

    oof_lgb[valid_idx] = lgb_model.predict_proba(
        X_lgb.iloc[valid_idx]
    )

    print(f"Fold {fold} completed.")

# Combine OOF predictions for the meta-model
meta_train = np.hstack([
    oof_cat,
    oof_xgb,
    oof_lgb
])

print("\nOOF training completed.")
print("Meta training shape:", meta_train.shape)
print("CatBoost OOF shape:", oof_cat.shape)
print("XGBoost OOF shape:", oof_xgb.shape)
print("LightGBM OOF shape:", oof_lgb.shape)


# In[59]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import clone

# Prepare validation data
X_cb_val = X_val_blend[selected_features].copy()
X_xgb_val = X_val_blend[selected_features].copy()
X_lgb_val = X_val_blend[selected_features].copy()

# Convert categorical columns for CatBoost
for col in cat_features:
    X_cb_val[col] = X_cb_val[col].astype(str)

# Match LightGBM categorical levels with training data
for col in cat_features:
    X_lgb_val[col] = pd.Categorical(
        X_lgb_val[col],
        categories=X_lgb[col].cat.categories
    )

# Train CatBoost on all training data
cat_final = CatBoostClassifier(
    iterations=1100,
    learning_rate=0.02682442119315144,
    depth=5,
    l2_leaf_reg=9.041718988294106,
    random_strength=0.1860260920815992,
    loss_function="MultiClass",
    verbose=False,
    random_seed=42
)

cat_final.fit(
    X_cb,
    y_train_blend,
    cat_features=cat_features
)

# Train XGBoost on all training data
xgb_final = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_final.fit(X_xgb, y_train_blend - 1)

# Train LightGBM on all training data
lgb_final = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=4,
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

lgb_final.fit(
    X_lgb,
    y_train_blend - 1,
    categorical_feature=cat_features
)

# Generate validation probabilities
val_cat = cat_final.predict_proba(X_cb_val)
val_xgb = xgb_final.predict_proba(X_xgb_val)
val_lgb = lgb_final.predict_proba(X_lgb_val)

# Create validation meta features
meta_val = np.hstack([
    val_cat,
    val_xgb,
    val_lgb
])

# Train the Logistic Regression meta-model
meta_model = LogisticRegression(
    C=1.0,
    max_iter=2000,
    random_state=42
)

meta_model.fit(meta_train, y_train_blend)

# Evaluate Stacking
stack_pred = meta_model.predict(meta_val)
stack_acc = accuracy_score(y_val_blend, stack_pred)

print(f"Stacking accuracy: {stack_acc:.4%}")
print(classification_report(y_val_blend, stack_pred))

# Evaluate simple average ensemble
average_proba = (val_cat + val_xgb + val_lgb) / 3
average_pred = np.argmax(average_proba, axis=1) + 1

average_acc = accuracy_score(y_val_blend, average_pred)

print(f"Simple average accuracy: {average_acc:.4%}")


# In[60]:


from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# Define the CatBoost meta-model
meta_cat = CatBoostClassifier(
    loss_function="MultiClass",
    verbose=False,
    random_seed=42,
    thread_count=-1
)

# Define hyperparameter search space
meta_param_dist = {
    "iterations": [300, 500, 700, 900],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "depth": [2, 3, 4, 5],
    "l2_leaf_reg": [3, 5, 8, 10],
    "random_strength": [0.1, 0.5, 1.0]
}

# Define stratified cross-validation
meta_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Search for the best meta-model parameters
meta_search = RandomizedSearchCV(
    estimator=meta_cat,
    param_distributions=meta_param_dist,
    n_iter=15,
    scoring="accuracy",
    cv=meta_cv,
    random_state=42,
    n_jobs=1,
    verbose=1
)

# Train and optimize the meta-model using OOF predictions
meta_search.fit(meta_train, y_train_blend)

# Evaluate on the independent validation set
best_meta_cat = meta_search.best_estimator_

meta_cat_pred = best_meta_cat.predict(meta_val).flatten()

meta_cat_acc = accuracy_score(
    y_val_blend,
    meta_cat_pred
)

print("\nBest parameters:")
print(meta_search.best_params_)

print(f"\nMeta-model CV accuracy: {meta_search.best_score_:.4%}")
print(f"CatBoost Stacking validation accuracy: {meta_cat_acc:.4%}")

print("\nClassification Report:")
print(classification_report(y_val_blend, meta_cat_pred))

print("\nPrevious Logistic Regression Stacking: 80.2000%")


# In[61]:


import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

# Prepare the selected features
X_extra = X_train_blend[selected_features].copy()
X_extra_val = X_val_blend[selected_features].copy()

# Define categorical and numerical features
cat_cols = ["country", "acquis", "opsys"]
num_cols = [c for c in selected_features if c not in cat_cols]

# Encode categorical features
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("numerical", "passthrough", num_cols)
])

# Define ExtraTrees model
extra_model = ExtraTreesClassifier(
    n_estimators=600,
    max_features=0.8,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

# Create OOF predictions
oof_extra = np.zeros((len(X_extra), 4))

skf_extra = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Train ExtraTrees across 5 folds
for fold, (train_idx, valid_idx) in enumerate(
    skf_extra.split(X_extra, y_train_blend), 1
):
    print(f"Training ExtraTrees fold {fold}/5...")

    extra_pipeline = make_pipeline(
        clone(preprocessor),
        clone(extra_model)
    )

    extra_pipeline.fit(
        X_extra.iloc[train_idx],
        y_train_blend.iloc[train_idx]
    )

    oof_extra[valid_idx] = extra_pipeline.predict_proba(
        X_extra.iloc[valid_idx]
    )

    print(f"Fold {fold} completed.")

# Train ExtraTrees on all training data
extra_final = make_pipeline(
    clone(preprocessor),
    clone(extra_model)
)

extra_final.fit(X_extra, y_train_blend)

# Generate validation probabilities
val_extra = extra_final.predict_proba(X_extra_val)

print("\nExtraTrees OOF shape:", oof_extra.shape)
print("ExtraTrees validation shape:", val_extra.shape)
print("ExtraTrees OOF completed.")


# In[63]:


import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Combine four base models
meta_train_16 = np.hstack([
    oof_cat,
    oof_xgb,
    oof_lgb,
    oof_extra
])

meta_val_16 = np.hstack([
    val_cat,
    val_xgb,
    val_lgb,
    val_extra
])

print("Meta training shape:", meta_train_16.shape)
print("Meta validation shape:", meta_val_16.shape)

# Train the CatBoost meta-model
meta_cat_4 = CatBoostClassifier(
    iterations=700,
    learning_rate=0.02,
    depth=2,
    l2_leaf_reg=3,
    random_strength=0.5,
    loss_function="MultiClass",
    verbose=False,
    random_seed=42,
    thread_count=-1
)

meta_cat_4.fit(meta_train_16, y_train_blend)

# Predict validation labels
pred_4 = meta_cat_4.predict(meta_val_16).astype(int).ravel()

# Evaluate performance
acc_4 = accuracy_score(y_val_blend, pred_4)

print("\nFour-model Stacking Accuracy:", round(acc_4 * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_val_blend, pred_4))

print("Confusion Matrix:")
print(confusion_matrix(y_val_blend, pred_4))

print("\nPrevious best accuracy: 81.60%")
print("Accuracy improvement:", round((acc_4 - 0.816) * 100, 2), "percentage points")


# In[64]:


import numpy as np

from xgboost import XGBClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Prepare the same 8 features used in Stacking
X_ovo_train = X_train_blend[selected_features].copy()
X_ovo_val = X_val_blend[selected_features].copy()

# Define categorical and numerical features
cat_cols = ["country", "acquis", "opsys"]
num_cols = [c for c in selected_features if c not in cat_cols]

# Encode categorical features
ovo_preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("numerical", "passthrough", num_cols)
])

# Define the binary XGBoost classifier
ovo_base = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)

# Build the One-vs-One classifier
ovo_model = make_pipeline(
    ovo_preprocessor,
    OneVsOneClassifier(ovo_base, n_jobs=-1)
)

# Train the OvO model
ovo_model.fit(X_ovo_train, y_train_blend)

# Predict validation labels
ovo_pred = ovo_model.predict(X_ovo_val)

# Evaluate performance
ovo_acc = accuracy_score(y_val_blend, ovo_pred)

print("OvO + XGBoost Accuracy:", round(ovo_acc * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_val_blend, ovo_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_val_blend, ovo_pred))

print("\nPrevious best Stacking accuracy: 81.60%")
print("OvO difference:", round((ovo_acc - 0.816) * 100, 2), "percentage points")


# In[65]:


from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Prepare the same 8 features
X_ova_train = X_train_blend[selected_features].copy()
X_ova_val = X_val_blend[selected_features].copy()

# Define categorical and numerical features
cat_cols = ["country", "acquis", "opsys"]
num_cols = [c for c in selected_features if c not in cat_cols]

# Encode categorical features
ova_preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("numerical", "passthrough", num_cols)
])

# Define the binary XGBoost classifier
ova_base = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)

# Build the One-vs-Rest classifier
ova_model = make_pipeline(
    ova_preprocessor,
    OneVsRestClassifier(ova_base, n_jobs=-1)
)

# Train the OvA model
ova_model.fit(X_ova_train, y_train_blend)

# Predict validation labels
ova_pred = ova_model.predict(X_ova_val)

# Evaluate performance
ova_acc = accuracy_score(y_val_blend, ova_pred)

print("OvA + XGBoost Accuracy:", round(ova_acc * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_val_blend, ova_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_val_blend, ova_pred))

print("\nOvO accuracy: 80.40%")
print("Three-model Stacking accuracy: 81.60%")


# In[66]:


import numpy as np
from itertools import combinations
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Prepare the selected features
X_ovo = X_train_blend[selected_features].copy()
X_ovo_val = X_val_blend[selected_features].copy()

cat_cols = ["country", "acquis", "opsys"]
num_cols = [c for c in selected_features if c not in cat_cols]

# Define preprocessing
ovo_preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("numerical", "passthrough", num_cols)
])

# Define the XGBoost binary classifier
ovo_base = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)

# Generate pairwise probability scores
def ovo_probability_scores(X_fit, y_fit, X_predict):
    classes = np.array([1, 2, 3, 4])
    scores = np.zeros((len(X_predict), len(classes)))

    for class_i, class_j in combinations(classes, 2):
        pair_mask = y_fit.isin([class_i, class_j])

        X_pair = X_fit.loc[pair_mask]
        y_pair = (y_fit.loc[pair_mask] == class_j).astype(int)

        pair_model = make_pipeline(
            clone(ovo_preprocessor),
            clone(ovo_base)
        )

        pair_model.fit(X_pair, y_pair)

        prob_j = pair_model.predict_proba(X_predict)[:, 1]

        scores[:, class_i - 1] += 1 - prob_j
        scores[:, class_j - 1] += prob_j

    # Normalize the pairwise scores
    scores = scores / scores.sum(axis=1, keepdims=True)

    return scores

# Generate five-fold OOF predictions
oof_ovo = np.zeros((len(X_ovo), 4))

skf_ovo = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for fold, (train_idx, valid_idx) in enumerate(
    skf_ovo.split(X_ovo, y_train_blend), 1
):
    print(f"Training OvO fold {fold}/5...")

    X_fold_train = X_ovo.iloc[train_idx]
    y_fold_train = y_train_blend.iloc[train_idx]
    X_fold_valid = X_ovo.iloc[valid_idx]

    oof_ovo[valid_idx] = ovo_probability_scores(
        X_fold_train,
        y_fold_train,
        X_fold_valid
    )

    print(f"Fold {fold} completed.")

# Generate validation predictions using all training data
val_ovo = ovo_probability_scores(
    X_ovo,
    y_train_blend,
    X_ovo_val
)

# Combine four base-model predictions
meta_train_ovo = np.hstack([
    oof_cat,
    oof_xgb,
    oof_lgb,
    oof_ovo
])

meta_val_ovo = np.hstack([
    val_cat,
    val_xgb,
    val_lgb,
    val_ovo
])

print("\nMeta training shape:", meta_train_ovo.shape)
print("Meta validation shape:", meta_val_ovo.shape)

# Train the CatBoost meta-model
meta_ovo_model = CatBoostClassifier(
    iterations=700,
    learning_rate=0.02,
    depth=2,
    l2_leaf_reg=3,
    random_strength=0.5,
    loss_function="MultiClass",
    verbose=False,
    random_seed=42,
    thread_count=-1
)

meta_ovo_model.fit(meta_train_ovo, y_train_blend)

# Evaluate the new Stacking model
pred_ovo_stack = meta_ovo_model.predict(
    meta_val_ovo
).astype(int).ravel()

acc_ovo_stack = accuracy_score(
    y_val_blend,
    pred_ovo_stack
)

print("\nOvO + Three-model Stacking Accuracy:",
      round(acc_ovo_stack * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_val_blend, pred_ovo_stack))

print("Confusion Matrix:")
print(confusion_matrix(y_val_blend, pred_ovo_stack))

print("\nPrevious best accuracy: 81.60%")
print("Accuracy difference:",
      round((acc_ovo_stack - 0.816) * 100, 2),
      "percentage points")


# In[67]:


from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import randint, uniform, loguniform

# Define the CatBoost meta-model
meta_base = CatBoostClassifier(
    loss_function="MultiClass",
    verbose=False,
    random_seed=42,
    thread_count=-1
)

# Define the hyperparameter search space
param_dist = {
    "iterations": [400, 600, 800, 1000, 1200],
    "depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.015, 0.02, 0.03, 0.05],
    "l2_leaf_reg": [2, 3, 5, 7, 9, 12],
    "random_strength": [0.1, 0.3, 0.5, 1.0, 1.5]
}

# Define cross-validation
cv_meta = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Randomized hyperparameter search
meta_search_new = RandomizedSearchCV(
    estimator=meta_base,
    param_distributions=param_dist,
    n_iter=30,
    scoring="accuracy",
    cv=cv_meta,
    random_state=42,
    n_jobs=1,
    verbose=1
)

# Fit using OOF meta-training features only
meta_search_new.fit(meta_train, y_train_blend)

# Evaluate the best meta-model on the independent validation set
best_meta_new = meta_search_new.best_estimator_

meta_pred_new = best_meta_new.predict(meta_val).astype(int).ravel()

meta_acc_new = accuracy_score(
    y_val_blend,
    meta_pred_new
)

print("\nBest Parameters:")
print(meta_search_new.best_params_)

print("\nBest CV Accuracy:",
      round(meta_search_new.best_score_ * 100, 2), "%")

print("\nValidation Accuracy:",
      round(meta_acc_new * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_val_blend, meta_pred_new))

print("Confusion Matrix:")
print(confusion_matrix(y_val_blend, meta_pred_new))

print("\nPrevious best validation accuracy: 81.60%")


# In[68]:


import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold

# Prepare full training and test data
X_full = train[selected_features].copy()
X_test_full = test[selected_features].copy()

y_full = train["y"].copy()

cat_cols = ["country", "acquis", "opsys"]

# Prepare CatBoost data
X_cb_full = X_full.copy()
X_cb_test = X_test_full.copy()

for col in cat_cols:
    X_cb_full[col] = X_cb_full[col].astype(str)
    X_cb_test[col] = X_cb_test[col].astype(str)

# Prepare LightGBM categorical data
X_lgb_full = X_full.copy()
X_lgb_test = X_test_full.copy()

for col in cat_cols:
    categories = pd.Index(
        pd.concat([
            X_lgb_full[col].astype(str),
            X_lgb_test[col].astype(str)
        ]).unique()
    )

    X_lgb_full[col] = pd.Categorical(
        X_lgb_full[col].astype(str),
        categories=categories
    )

    X_lgb_test[col] = pd.Categorical(
        X_lgb_test[col].astype(str),
        categories=categories
    )

# Prepare XGBoost data
X_xgb_full = X_full.copy()
X_xgb_test = X_test_full.copy()

# Define base models
cat_params = {
    "iterations": 1100,
    "learning_rate": 0.02682442119315144,
    "depth": 5,
    "l2_leaf_reg": 9.041718988294106,
    "random_strength": 0.1860260920815992
}

xgb_params = {
    "n_estimators": 500,
    "learning_rate": 0.1,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

lgb_params = {
    "n_estimators": 700,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5
}

# Generate OOF predictions
oof_cat_full = np.zeros((len(X_full), 4))
oof_xgb_full = np.zeros((len(X_full), 4))
oof_lgb_full = np.zeros((len(X_full), 4))

skf_full = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for fold, (train_idx, valid_idx) in enumerate(
    skf_full.split(X_full, y_full), 1
):
    print(f"Training fold {fold}/5...")

    # CatBoost
    cat_model = CatBoostClassifier(
        **cat_params,
        loss_function="MultiClass",
        verbose=False,
        random_seed=42,
        thread_count=-1
    )

    cat_model.fit(
        X_cb_full.iloc[train_idx],
        y_full.iloc[train_idx],
        cat_features=cat_cols
    )

    oof_cat_full[valid_idx] = cat_model.predict_proba(
        X_cb_full.iloc[valid_idx]
    )

    # XGBoost
    xgb_model = XGBClassifier(
        **xgb_params,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(
        X_xgb_full.iloc[train_idx],
        y_full.iloc[train_idx] - 1
    )

    oof_xgb_full[valid_idx] = xgb_model.predict_proba(
        X_xgb_full.iloc[valid_idx]
    )

    # LightGBM
    lgb_model = LGBMClassifier(
        **lgb_params,
        objective="multiclass",
        num_class=4,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    lgb_model.fit(
        X_lgb_full.iloc[train_idx],
        y_full.iloc[train_idx]
    )

    oof_lgb_full[valid_idx] = lgb_model.predict_proba(
        X_lgb_full.iloc[valid_idx]
    )

    print(f"Fold {fold} completed.")

# Train final base models on all 5000 observations
cat_final_full = CatBoostClassifier(
    **cat_params,
    loss_function="MultiClass",
    verbose=False,
    random_seed=42,
    thread_count=-1
)

cat_final_full.fit(
    X_cb_full,
    y_full,
    cat_features=cat_cols
)

xgb_final_full = XGBClassifier(
    **xgb_params,
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_final_full.fit(X_xgb_full, y_full - 1)

lgb_final_full = LGBMClassifier(
    **lgb_params,
    objective="multiclass",
    num_class=4,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

lgb_final_full.fit(X_lgb_full, y_full)

# Generate test probabilities
test_cat = cat_final_full.predict_proba(X_cb_test)
test_xgb = xgb_final_full.predict_proba(X_xgb_test)
test_lgb = lgb_final_full.predict_proba(X_lgb_test)

# Build meta features
meta_train_full = np.hstack([
    oof_cat_full,
    oof_xgb_full,
    oof_lgb_full
])

meta_test_full = np.hstack([
    test_cat,
    test_xgb,
    test_lgb
])

# Use optimized meta parameters when available
if "meta_search_new" in globals():
    final_meta_params = meta_search_new.best_params_
else:
    final_meta_params = {
        "iterations": 700,
        "learning_rate": 0.02,
        "depth": 2,
        "l2_leaf_reg": 3,
        "random_strength": 0.5
    }

print("\nMeta-model parameters:", final_meta_params)

# Train final meta-model
final_meta_model = CatBoostClassifier(
    **final_meta_params,
    loss_function="MultiClass",
    verbose=False,
    random_seed=42,
    thread_count=-1
)

final_meta_model.fit(meta_train_full, y_full)

# Predict Kaggle test labels
test_predictions = final_meta_model.predict(
    meta_test_full
).astype(int).ravel()

# Create submission file
submission = pd.DataFrame({
    "id": test["id"],
    "y": test_predictions
})

submission.to_csv("submission_stacking.csv", index=False)

print("\nSubmission file created: submission_stacking.csv")
print("\nPrediction distribution:")
print(submission["y"].value_counts().sort_index())

print("\nSubmission preview:")
print(submission.head())

print("\nSubmission shape:", submission.shape)


# In[ ]:




