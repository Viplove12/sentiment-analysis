# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

dataset=pd.read_csv('train.csv')

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
 
#text preprocessing /cleaning
for i in range(0,31962):
    review=re.sub('[^a-zA-Z]',' ',dataset['tweet'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
#Creating a bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

#splitting the dataset
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=0)    

#implementing naive bayes
from sklearn.naive_bayes import GaussianNB
classifierNB=GaussianNB()
classifierNB.fit(xtrain,ytrain)

#implementing random forest
from sklearn.ensemble import RandomForestClassifier
classifierRF=RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=0)
classifierRF.fit(xtrain,ytrain)


#predicting test results
ypredNB=classifierNB.predict(xtest)
ypredRF=classifierRF.predict(xtest)

#confusion matrix
from sklearn.metrics import confusion_matrix
cmNB=confusion_matrix(ytest,ypredNB)
cmRF=confusion_matrix(ytest,ypredRF)
   