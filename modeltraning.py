# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:29:04 2020

@author: praneeth
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import matplotlib.image as mpimg  
import streamlit as sl
from joblib import dump, load
#Model Training
def redp():
    df=pd.read_csv("ipl.csv")
    df.drop(['mid', 'venue', 'striker','non-striker','batsman','bowler'],inplace=True,axis=1)
    df['date']=df['date'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))
    consistent=['Kolkata Knight Riders', 'Deccan Chargers', 'Chennai Super Kings', 'Rajasthan Royals',
       'Mumbai Indians', 'Kings XI Punjab',
       'Royal Challengers Bangalore', 'Delhi Daredevils','Sunrisers Hyderabad']
    df=df[(df['bat_team']).isin(consistent)&df['bowl_team'].isin(consistent)]
    df=df[df['overs']>5]
    new_df=pd.get_dummies(df,columns=['bat_team','bowl_team'])
    new_df = new_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]
    X_train = new_df.drop(labels='total', axis=1)[new_df['date'].dt.year <= 2016]
    X_test = new_df.drop(labels='total', axis=1)[new_df['date'].dt.year >= 2017]
    y_train = new_df[new_df['date'].dt.year <= 2016]['total'].values
    y_test = new_df[new_df['date'].dt.year >= 2017]['total'].values
    X_train.drop(labels='date', axis=True, inplace=True)
    X_test.drop(labels='date', axis=True, inplace=True)
    return X_train,X_test,y_train,y_test
X_train,X_test,y_train,y_test=redp()
model=Ridge()
parms={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,2,10,20,30,35]}
rr=GridSearchCV(model,parms,scoring='neg_mean_squared_error',cv=4)
rr.fit(X_train,y_train)
prediction=rr.predict(X_test)
dump(rr, 'filename1.joblib')