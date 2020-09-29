#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:42:11 2020

@author: saikolusu
"""
'''
import matplotlib.pyplot as plt
import seaborn as sns

def analyse_na_value(df, var):

   sns.set(font_scale=1.4)
   df['Category'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
   plt.xlabel("Category", labelpad=14)
   plt.ylabel("Count of Category", labelpad=14)
   plt.title("Count of Categoryin dataset", y=1.02);


# let's run the function on each variable with missing data
analyse_na_value(df, "Category")
'''