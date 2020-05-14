# -*- coding: utf-8 -*-
"""
Created on Mon May 11 23:16:44 2020

@author: RajdevKapoor
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting=3)

#cleaning the texts
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
for i in range(1000):
    review = re.sub("[^a-zA-Z]"," ",dataset.Review[i])
    review=review.lower()
    review=review.split()
    ps = PorterStemmer()
    review = [ ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = ' '.join(review)
    corpus.append(review)
    
###Bags of Words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X= cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values