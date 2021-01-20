# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:14:42 2020

@author: Anson
"""


"""Report"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import statsmodels.api as sm
import pylab as pl
from pandas import DataFrame
##Data Cleaning
df = pd.read_csv('D:/Graduate study/Spring 2020/6290_10 AI Application/Project/list_attr_celeba.csv')
X,y= df.iloc[:, np.r_[1:21, 22:41]].values,df.iloc[:,21].values
df.replace(to_replace=-1, value=0, inplace=True)
col=df.columns[np.r_[1:21, 22:41]]#列名
df.replace(to_replace=-1, value=0, inplace=True)

##Feature Select
#Total Feature
l_X,l_y= pd.DataFrame(df,columns=['Pale_Skin','Mouth_Slightly_Open','Brown_Hair','Oval_Face','Wearing_Lipstick', 'Pointy_Nose','Smiling',
                                  'Straight_Hair','Heavy_Makeup', 'No_Beard','High_Cheekbones','Big_Lips','Black_Hair','Chubby',
                                  'Bushy_Eyebrows','Wavy_Hair','Young','Bags_Under_Eyes','Goatee','Sideburns','Blurry','Mustache',
                                  '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Wearing_Necktie','Blond_Hair','Bangs','Double_Chin',
                                  'Big_Nose', 'Wearing_Earrings','Attractive','intercept','Gray_Hair','Receding_Hairline','Bald'
                                  ]),pd.DataFrame(df,columns=['Male'])
#l_X,l_y= df.iloc[:, np.r_[1:21, 22:42]],df.iloc[:,21]
#Top 10 Features
#l_X,l_y= pd.DataFrame(df,columns=['Wearing_Lipstick', 'Heavy_Makeup', 'No_Beard']),pd.DataFrame(df,columns=['Male'])

##Split dataset
l_X_train=X[0:162770,: ]
l_X_valid=X[162770:182637,:]
l_X_test=X[182637:202599,:]
l_y_train=y[0:162770]
l_y_valid=y[162770:182637]
l_y_test=y[182637:202599]

##predict
from sklearn.metrics import confusion_matrix
import pyspark.sql.functions as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

##Modeling RF
rf=RandomForestClassifier(
        n_estimators= 40, max_depth=13, min_samples_split=50,
        min_samples_leaf=10 ,oob_score=True,random_state=10)
rf.fit(l_X_train,l_y_train)
y_pred = rf.predict(l_X_valid)
y_true = l_y_valid.tolist()
cnf_matrix = confusion_matrix(y_true, y_pred)
print(cnf_matrix)
print("Prediction Accuracy is ", (cnf_matrix[0,0]+cnf_matrix[1,1])/sum(sum(cnf_matrix)) )
print("True Positive Accuracy is ", (cnf_matrix[1,1]/ (cnf_matrix[1,1]+cnf_matrix[1,0])))

"""
Full:
RF With Total Feature
[[10169  1240]
 [  172  8286]]
Prediction Accuracy is  0.9289273669904867
True Positive Accuracy is  0.9796642232206195

RF With Top 3 Feature
[[9071 2338]
 [  35 8423]]
Prediction Accuracy is  0.8805556953742387
True Positive Accuracy is  0.9958619058879168
"""