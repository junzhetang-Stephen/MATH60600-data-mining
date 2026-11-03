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


# In[ ]:




