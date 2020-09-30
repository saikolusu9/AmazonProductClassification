#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:33:58 2020

@author: saikolusu
"""

import pandas as pd
import os
from scripts.train.decorator import Decorator

class ReadData:
    decorator = Decorator()

    def read_data(self, path, file, columns=None, header=None):
        #df = pd.read_csv(os.path.relpath(path + "/" + file, os.path.dirname(__file__)), header=header)
        df = pd.read_csv(path + "/" + file, header=header)
        if header == None:
            df.columns = columns
        return df



