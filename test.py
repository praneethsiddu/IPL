# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:06:43 2020

@author: praneeth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn import metrics
import matplotlib.image as mpimg  
import streamlit as sl
sl.title('INDIAN PREMIER LEAGUE SCORE PREDICTION')
img = mpimg.imread('ipl.jpg')
sl.image(img,use_column_width=True,caption='INDIAN PREMIER LEAGUE')
sl.subheader('About IPL')
sl.info("The Indian Premier League (IPL) is a professional Twenty20 cricket league in India contested during March or April and May of every year by eight teams representing eight different cities in India. The league was founded by the Board of Control for Cricket in India (BCCI) in 2008. The IPL has an exclusive window in ICC Future Tours Programme.The IPL is the most-attended cricket league in the world and in 2014 ranked sixth by average attendance among all sports leagues. In 2010, the IPL became the first sporting event in the world to be broadcast live on YouTube. The brand value of the IPL in 2019 was ₹475 billion (US$6.7 billion), according to Duff & Phelps. According to BCCI, the 2015 IPL season contributed ₹11.5 billion (US$160 million) to the GDP of the Indian economy.") 
if sl.button("LIST OF TEAMS"):
    sl.text_area("Teams","Chennai Super Kings-CSK\nDelhi Daredevils-DD\nKings XI Punjab-KXIP\nKolkata Knight Riders-KKR\nRajasthan Royals-RR\nMumbai Indians-MI\nSunrisers Hyderabad-SRH\nRoyal Challengers Bangalore-RCB")
else:
    sl.write("Hit The Button To List The Teams Participating In The Tournament ")    
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
model=Ridge()
parms={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,2,10,20,30,35]}
rr=GridSearchCV(model,parms,scoring='neg_mean_squared_error',cv=4)
rr.fit(X_train,y_train)
prediction=rr.predict(X_test)
batting_team=sl.selectbox('Select The Batting Team?',('Chennai Super Kings','Delhi Daredevils','Kings XI Punjab','Kolkata Knight Riders','Rajasthan Royals','Mumbai Indians','Sunrisers Hyderabad','Royal Challengers Bangalore'))
sl.write('You selected:',batting_team)
result_array=[]
if batting_team == 'Chennai Super Kings':
    result_array = result_array + [1,0,0,0,0,0,0,0]
elif batting_team == 'Delhi Daredevils':
    result_array = result_array + [0,1,0,0,0,0,0,0]
elif batting_team == 'Kings XI Punjab':
    result_array = result_array + [0,0,1,0,0,0,0,0]
elif batting_team == 'Kolkata Knight Riders':
    result_array = result_array + [0,0,0,1,0,0,0,0]
elif batting_team == 'Mumbai Indians':
    result_array = result_array + [0,0,0,0,1,0,0,0]
elif batting_team == 'Rajasthan Royals':
    result_array = result_array + [0,0,0,0,0,1,0,0]
elif batting_team == 'Royal Challengers Bangalore':
    result_array = result_array + [0,0,0,0,0,0,1,0]
elif batting_team == 'Sunrisers Hyderabad':
    result_array = result_array + [0,0,0,0,0,0,0,1]    
else:
    pass

sl.write(batting_team)
bowling_team=sl.selectbox('Select The Bowling Team?',('Chennai Super Kings','Delhi Daredevils','Kings XI Punjab','Kolkata Knight Riders','Rajasthan Royals','Mumbai Indians','Sunrisers Hyderabad','Royal Challengers Bangalore'))
sl.write('You selected:',bowling_team)
if bowling_team == 'Chennai Super Kings':
    result_array = result_array + [1,0,0,0,0,0,0,0]
elif bowling_team == 'Delhi Daredevils':
    result_array = result_array + [0,1,0,0,0,0,0,0]
elif bowling_team == 'Kings XI Punjab':
    result_array = result_array + [0,0,1,0,0,0,0,0]
elif bowling_team == 'Kolkata Knight Riders':
    result_array = result_array + [0,0,0,1,0,0,0,0]
elif bowling_team == 'Mumbai Indians':
    result_array = result_array + [0,0,0,0,1,0,0,0]
elif bowling_team == 'Rajasthan Royals':
    result_array = result_array + [0,0,0,0,0,1,0,0]
elif bowling_team == 'Royal Challengers Bangalore':
    result_array = result_array + [0,0,0,0,0,0,1,0]
elif bowling_team == 'Sunrisers Hyderabad':
    result_array = result_array + [0,0,0,0,0,0,0,1]
else:
    pass
if batting_team==bowling_team:
    sl.error("Please Select Two Different Teams For Prediction")
sl.title("Lets Start Our Score Prediction")
sl.warning('please choice the overs correctly')
overs=sl.number_input('Enter Number Of Overs Compeleted',min_value=5.0,max_value=20.0)
runs = sl.number_input('Enter Number Of Runs Scored')
wickets= sl.slider('Select numbetr of wickets down?', 0,10, 1)
wickets_in_prev_5 =sl.number_input('Enter Number Of wickwts Down In Previous 5 Overs')
runs_in_prev_5 =sl.number_input('Enter Number Of Runs Scored In Previous 5 Overs')
result_array = result_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]
data = np.array([result_array])
lower_limit =int(rr.predict(data))-5
upper_limit =int(rr.predict(data))+10
if runs_in_prev_5:
     sl.write("THE PREDICTED SCORE IS BETWEEN "+str(lower_limit)+" - "+str(upper_limit))
     sl.success('we predicted succesfully')  
     sl.balloons()    
sl.info("This is a purely informational message for the users.\n    That the predictions are completely based on data collected from previous scores of IPL seasons. \n Sometimes The prediction May not Meet the reality as it is also Depended upon the form of the players,weather, pitch conditions and other sport factors")
