#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:14:09 2020

@author: saikolusu
"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class SupportVector:
    def build_svc(self,train_x,train_y,val_x):
        # defining parameter range
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['rbf','linear']}

        grid = GridSearchCV(estimator=SVC(), param_grid = param_grid, cv = 5)
        print("Error is here")
        # fitting the model for grid search
        grid.fit(train_x, train_y)
        # print best parameter after tuning
        print(grid.best_params_)

        # View the accuracy score
        print('Best score for data1:', grid.best_score_)

        # print how our model looks after hyper-parameter tuning
        print('Best C:', grid.best_estimator_.C)
        print('Best Kernel:',grid.best_estimator_.kernel)
        print('Best Gamma:',grid.best_estimator_.gamma)

         # Train a new classifier using the best parameters found by the grid search
        predicitions = SVC(C=grid.best_estimator_.C, kernel=grid.best_estimator_.kernel, gamma=grid.best_estimator_.gamma).fit(train_x, train_y).predict(val_x)

        # print classification report
        #print(classification_report(val_y, grid_predictions))

        return predicitions
