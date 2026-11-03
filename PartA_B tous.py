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


# In[ ]:




