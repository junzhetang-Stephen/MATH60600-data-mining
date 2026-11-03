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


# In[ ]:


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


# In[ ]:


study_cat = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study_cat.optimize(
    objective_cat,
    n_trials=30
)

print("Best parameters:")
print(study_cat.best_params)

print("Best CV Accuracy:")
print(round(study_cat.best_value, 4))


# In[ ]:




