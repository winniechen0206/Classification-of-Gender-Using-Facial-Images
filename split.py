# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:17:20 2020

@author: Anson
"""
import os,shutil
import pandas as pd

root=os.getcwd().replace('\\','/')
floder_t=os.getcwd()+'\\training\\'
floder_v=os.getcwd()+'\\validation\\'
if not os.path.exists(floder_t):
    os.makedirs(floder_t)
    os.makedirs(floder_t+'\\male')
    os.makedirs(floder_t+'\\female')
if not os.path.exists(floder_v):
    os.makedirs(floder_v)
    os.makedirs(floder_v+'\\male')
    os.makedirs(floder_v+'\\female')    
dt = pd.read_csv(root+'/'+'list_attr_celeba.csv')
for i in range(21015,26015):#Training 0-16013; Validation:16014-21014
    name = dt.iloc[i,0]
    m=dt.iloc[i,21]
    if m == 1:
        shutil.move(root+'/img_align_celeba/img_align_celeba/'
                    +name, root + '/validation/male')
    else:shutil.move(root+'/img_align_celeba/img_align_celeba/'
                    +name, root + '/validation/female')
  
#shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")


