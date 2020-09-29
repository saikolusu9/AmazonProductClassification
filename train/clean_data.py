#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:46:37 2020

@author: saikolusu
"""

import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from scripts.train.decorator import Decorator
decorator = Decorator()
class CleanData:
    # Removing StopWords and Punctuation
    def __init__(self):
        self.punctuations = string.punctuation
        #nltk.download('stopwords')
        #nltk.download('wordnet')
        self.stopword_list = stopwords.words("english")
        self.lem = WordNetLemmatizer()


    def _clean(self, text):
        cleaned_text = text.lower()

        cleaned_text = "".join(c for c in cleaned_text if c not in self.punctuations)

        words = cleaned_text.split()
        words = [w for w in words if w not in self.stopword_list]

        words = [self.lem.lemmatize(word, "v") for word in words]
        words = [self.lem.lemmatize(word, "n") for word in words]

        cleaned_text = " ".join(words)

        return cleaned_text


