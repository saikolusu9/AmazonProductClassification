#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:07:03 2020

@author: saikolusu
"""

model = MultinomialNB()
model.fit(train_x, train_y)
preds = model.predict(val_x)
accuracy_score(preds, val_y)
