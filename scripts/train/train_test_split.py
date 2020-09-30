#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:56:34 2020

@author: saikolusu
"""
from sklearn.model_selection import train_test_split
from scripts.train.decorator import Decorator
decorator = Decorator()

class Split:
    def train_test_split(self, word_vectors_tfidf, target):
        train_x, val_x, train_y, val_y = train_test_split(word_vectors_tfidf, target, test_size=0.33, random_state=42)
        return train_x, val_x, train_y, val_y
