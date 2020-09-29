#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:27:56 2020r

@author: saikolusu
"""

from scripts.train.clean_data import CleanData
from scripts.train.feature_engineering import FeatureExtraction
from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from scripts.predict.make_prediction import ExecutePrediction

app = Flask(__name__)
ex = ExecutePrediction()

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        df = pd.DataFrame(np.array([message]), columns=['Text'])
        my_prediction = ex.prediction(df)
        print("The Predicition is :" + my_prediction)

    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)