#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:27:56 2020r

@author: saikolusu
"""

import os
import configparser as cp
from train.read_data import ReadData
from train.clean_data import CleanData
from train.feature_engineering import FeatureExtraction
from train.train_test_split import Split
from sklearn import metrics
from flask import Flask, render_template, url_for, request
import pandas as pd

app = Flask(__name__)


class Execute:

    def __init__(self):
        self.cur_path = os.path.dirname(__file__)
        self.config = cp.RawConfigParser()
        self.properties_file_path = os.path.relpath('../resources/application.properties', self.cur_path)
        self.config.read(self.properties_file_path)
        self.data_folder = self.config.get('Properties', 'folder')
        self.data_file = self.config.get('Properties', 'file_name')
        self.columns = ['Category', 'Description']
        self.readdata = ReadData()
        self.cleaneddata = CleanData()
        self.features_extract = FeatureExtraction()
        self.split_df = Split()

    def main(self, df):
        df["cleaned"] = df["Text"].astype(str).apply(self.cleaneddata._clean)

        word_vectors_tfidf = self.features_extract.featrue_extract(df)
        target = self.features_extract.label_encoding(df)

        train_x, val_x, train_y, val_y = self.split_df.train_test_split(word_vectors_tfidf, target)
        pred_y = self.rf.build_RF(train_x, train_y, val_x)

        print("Accuracy:", metrics.accuracy_score(val_y, pred_y))




if __name__ == '__main__':
    ex = Execute()
    ex.main()
# df = ReadData.read_data(path, file_name, col_names=col_names, header=None)
# df = df[df['Category'].isin(['Household','Books','Electronics','Clothing & Accessories'])]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)