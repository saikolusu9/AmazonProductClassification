#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:14:57 2020

@author: saikolusu
"""

from sklearn.ensemble import RandomForestClassifier
from scripts.train.decorator import Decorator
import numpy as np
decorator = Decorator()
from sklearn.model_selection import RandomizedSearchCV
########################################################################################################

class RF:
    @decorator.def_decorator
    def build_RF(self,train_x,train_y,val_x):


        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=50, stop=500, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        # min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        # min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       # 'min_samples_split': min_samples_split,
                       # 'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        #####################

        # Use the gridsearch to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_grid = RandomizedSearchCV(estimator=rf, param_distributions =random_grid, cv=5, random_state=1)
        # Fit the random search model
        rf_grid.fit(train_x, train_y)
        print(rf_grid.best_estimator_.get_params())
        y_pred = rf_grid.predict(val_x)

        return y_pred


