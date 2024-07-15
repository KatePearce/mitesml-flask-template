import pickle
import sklearn

import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import KFold, cross_val_score, train_test_split

#import matplotlib.pyplot as plt


# Load Datasets
bc_x, bc_y = datasets.load_breast_cancer(return_X_y = True, as_frame = True)

# Feature Selection
bc_features = bc_x[['mean radius', 'mean area', 'worst radius', 'worst area', 'worst concavity', 'perimeter error']]
bc_features.head()

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(bc_features, bc_y, test_size=0.20, random_state=42)

# Fit the model
regr = linear_model.LogisticRegression(fit_intercept = True, penalty = 'l2', solver = 'liblinear')
regr.fit(X_train.to_numpy().reshape(-6,6), y_train)
y_pred = regr.predict(X_test.to_numpy().reshape(-6,6))

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(regr, f)