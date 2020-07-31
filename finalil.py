# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:31:26 2020

@author: praneeth
"""
"""
Created by:Partapu Praneeth
"""

import pandas as pd
import numpy as np
import matplotlib.image as mpimg  
import streamlit as sl
from joblib import dump, load


rr = load('filename.joblib')

sl.write("""
# IPL SCORE PREDICTION APP
This app predicts the **IPL score!**
Data obtained from previous ipl seasons.
""")

img = mpimg.imread('ipl.jpg')
sl.image(img,use_column_width=True,caption='Indian Premier league') 
team = mpimg.imread('teams.jpg')
sl.image(team,use_column_width=True,caption='Consistent Franchises') 
sl.write("""
## List OF Consistent IPL Teams
1.Chennai Super Kings-CSK\n                      
2.Delhi Daredevils-DD\n 
3.Kings XI Punjab-KXIP\n 
4.Kolkata Knight Riders-KKR\n 
5.Rajasthan Royals-RR\n 
6.Mumbai Indians-MI\n 
7.Sunrisers Hyderabad-SRH\n 
8.Royal Challengers Bangalore-RCB\n 
""") 
log = mpimg.imread('logo.jpg')
sl.sidebar.image(log,use_column_width=True)
def user_input_features():
    sl.sidebar.subheader('**Teams:**')
    batting_team=sl.sidebar.selectbox('Select The Batting Team?',('Chennai Super Kings','Delhi Daredevils','Kings XI Punjab','Kolkata Knight Riders','Rajasthan Royals','Mumbai Indians','Sunrisers Hyderabad','Royal Challengers Bangalore'))
    bowling_team=sl.sidebar.selectbox('Select The Bowling Team?',('Chennai Super Kings','Delhi Daredevils','Kings XI Punjab','Kolkata Knight Riders','Rajasthan Royals','Mumbai Indians','Sunrisers Hyderabad','Royal Challengers Bangalore'))
    if batting_team==bowling_team:
        sl.sidebar.error("Please Select Two Different Teams For Prediction")
    sl.sidebar.subheader('**Overs:**')
    overs=sl.sidebar.number_input('Enter Number Of Overs Compeleted',min_value=5.0,max_value=20.0,step=0.1,format='%.1f')
    if round(overs,1)%1>0.6:
        sl.sidebar.error("There are Only **Six Balls** Per Over.Please Choose the overs Correctly!")
    sl.sidebar.subheader('**Runs:**')
    runs = sl.sidebar.number_input('Enter Number Of Runs Scored',min_value=-1,max_value=720,value=0)
    sl.sidebar.subheader('**Wickets:**')
    wickets= sl.sidebar.slider('Number of Wickets Down', 0,10, 1)
    return batting_team,bowling_team,overs,runs,wickets
    


batting_team,bowling_team,overs,runs,wickets=user_input_features();


sl.title("Lets Start Our Score Prediction")
sl.warning("The Predictions are based on Statistisc Of First 5 overs")
wickets_in_prev_5 = sl.slider('Number Of Wickets Down In Previous 5 Overs', 0,10, 1)
runs_in_prev_5 =sl.number_input('Enter Number Of Runs Scored In Previous 5 Overs',min_value=-1,max_value=720,value=0)

result_array=[]

def bt():
    global result_array
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
    return 
bt()

def bwt():
    global result_array
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
    return
bwt() 

result_array = result_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]
data = np.array([result_array])
lower_limit =int(rr.predict(data))-5
upper_limit =int(rr.predict(data))+10
@sl.cache
def return_input_data():
    data={'bat_team':batting_team,
          'bowl_team':bowling_team,
          'overs':round(overs,1),
          'runs':runs,
          'wickets':wickets,
          'wickets_in_prev_5':wickets_in_prev_5,
          'runs_in_prev_5':runs_in_prev_5
          }
    features = pd.DataFrame(data, index=[0])
    return features
dfi=return_input_data()
sl.write("**Selected Input values**")
sl.write(dfi)







if sl.button('Predict'):
    if wickets==10:
        sl.write("**Our Predicted Score Is Between** "+str(runs))
        sl.info("**This is a purely informational message for the users.That these predictions are completely based on data collected from previous scores of IPL seasons.Sometimes The prediction May not Meet the reality as it is also Depended upon the form of the players,weather, pitch conditions and other sport factors**")
        sl.success('Done!!')
        sl.balloons()
    else:
        sl.info("**Our Predicted Score**"+str(lower_limit)+" - "+str(upper_limit))
        sl.info("**This is a purely informational message for the users.That the predictions are completely based on data collected from previous scores of IPL seasons.Sometimes The prediction May not Meet the reality as it is also Depended upon the form of the players,weather, pitch conditions and other sport factors**")
        sl.success('Done!!')  
        sl.balloons()    
else:
    pass
@sl.cache
def winnerlist():
    win=pd.read_csv('win.csv')
    win.index=win['Year']
    win.index.name='Year'
    return win
winners=winnerlist()
if sl.button("Click Here To Check The IPL Winners List"):
    sl.write("**IPL Winners List**")
    sl.write(""" Here we present the list of IPL winners in the table which include winner, runner-up, venue, number of teams participated, player of the match and player of the series. 
         Let's have look at the table of IPL Winners List from **2008** to **2020**.""")
    sl.write(winners) 
    win = mpimg.imread('win.jpg')
    sl.image(win,use_column_width=True,caption='Triumphs') 
    sl.header("IPL Winner 2008: Winner Rajasthan Royals")
    sl.write(""" The Shane Warne led Rajasthan Royals won the first edition of the Indian Premier League. A thoroughly well balanced team with experienced batsmen like Shane Watson and Graeme Smith, an explosive batsman like Yusuf Pathan in the middle order, and deceptive bowler like Sohail Tanvir were the backbone of the Rajasthan Royals in the first edition. But in the final against the Chennai Super Kings, Yusuf Pathan played a match of his life with contributions in both bowling and the batting department. Yusuf first took 3 crucial wickets with run-rate less than a bowl and scored a quickfire 56 from 39.""")
    sl.header("IPL Winner 2009: Winner Deccan Chargers")
    sl.write("The Chargers led by Adam Gilchrist won the second edition of the Indian Premier League. The team had some incredible hitters in them – Herschelle Gibbs, Andrew Symonds, young Rohit Sharma. Pragyan Ojha played a very crucial role in the team’s journey with his left-arm spin bowling scalping 18 wickets in the tournament. Adam Gilchrist notched up a total of 495 runs in 16 matches and stood second in the overall list of highest run-scorers in the season behind Mathew Hayden.")
    sl.header("IPL Winner 2010: Winner Chennai Super Kings")
    sl.write("Channai Super Kings become the first team of IPL to win back to back finals in the history. The defending champions held the title against the batting titans of the tournament RCB. CSK’s batting unit was lit up by the presence of Mr Cricket, Michael Hussey, who played some important innings throughout the tournament. Hussey and Murali Vijay’s 159 run opening partnership against the Royal Challengers Bangalore was the main reason for CSK defending the title.")
    sl.header("IPL Winner 2012: Kolkata Knight Riders")
    sl.write("Kolkata Knight Riders defeated the defending champions Chennai Super Kings in the final. With KKR batsman left with the daunting task of chasing 190 in the final, Bisla took the onus of chase on himself and replied with a swashbuckling inning of 89 from 48 balls. CSK’s run feast led by Suresh Raina’s 73 from 38 wasn’t enough to stop the Knights from snatching the title from them.")
    sl.header("IPL Winner 2013: Winner Mumbai Indians")
    sl.write("The hitman, Rohit Sharma led Mumbai Indians to their first glory at the IPL. MI did so by defeating Chennai Super Kings in the final of 2013 IPL. The side was clinically well balanced throughout the tournament. The likes of Kieron Pollard, Lasith Malinga and Mitchell Johnson were a consistent part of the team’s playing XI.")
    sl.header("IPL Winner 2014: Winner Kolkata Knight Riders")
    sl.write("Kolkata Knight Riders become the second team to won the IPL trophy more than once. If Manvinder Bisla was the hero in the final of the 2012 edition, this time it was Manish Pandey’s heroics that led KKR to chase a huge total of 199. Pandey scored a brilliant 94 from 50. Robin Uthappa from KKR won the orange cap for scoring 660 runs in 16 innings.")
    sl.header("IPL Winner 2015: Winner Mumbai Indians")
    sl.write("Mumbai Indians became the third team after CSK and KKR to win the title more than once. Again, the apt balancing of the Mumbai Indians helped them win the trophy. Simmons, Rohit Sharma, Ambati Rayadu and Kieron Pollard carried the batting unit. The bowling department was led by Slinga Malinga. Lasith Malinga took 24 wickets in the tournament but couldn’t get the purple cap. Dwayne Bravo went ahead of him by scalping two extra wickets.")
    sl.header("IPL Winner 2016: Winner Sunrisers Hyderabad")
    sl.write("The Sunrisers Hyderabad team managed to win the final against Royal Challengers Bangalore by just 8 runs. David Warner masterfully crafted the team journey through the season as a captain. But the season was more about the Virat Kohli - Ab de Villiers partnership in almost all the RCB matches. Virat scored 4 centuries and 7 half-centuries in 16 innings he played! There should be no doubt on who was the orange cap winner for the season.")
    sl.header("IPL Winner 2017: Winner Mumbai Indian")
    sl.write("Mumbai Indians became the 1st team in IPL history to win the title more than two times. They first win was in 2013 and the second in 2015. The victory margin of Mumbai Indians against the Rising Pune Supergiants in the final was just one run. Rohit Sharma, as the captain of Mumbai Indians, was exceptional.")
    sl.header("IPL Winner 2018: Winner Chennai Super Kings")
    sl.write("Chennai Super Kings become the second team to win more than two trophies of IPL. Veterans Ambati Rayadu and Shane Watson were the backbones of the batting unit. Dhoni, as usual, provided best finishes in most of the games. CSK’s bowling unit was lucky to get inputs from MS Dhoni, the captain and Dwanye Bravo, who led the bowling attack for CSK.")
    sl.header("IPL Winner 2019: Winner Mumbai Indians")
    sl.write("The final of the IPL 2019 again saw the clash of the titans. Mumbai Indians and Chennai Super Kings met again in the finals. The final went down to the last bowl. Shardul Thakur who was hoping for at least one run for the super over failed to counter Malinga’s off-cutter. Thus, Mumbai Indians became the champions for a record 4th time.")
if sl.button("Click Here To Know Latest News About IPL"):    
    uae = mpimg.imread('uae.jpg')
    sl.image(uae,use_column_width=True,caption='IPL 2020 in UAE to start on September 19, final on November 8') 
    sl.header("IPL 2020 in UAE to start on September 19, final on November 8")
    sl.write("“We are yet to hear from the BCCI officially. We have heard the statements by Brijesh Patel (IPL Governing Council chairman) in the media and we welcome it. At the ECB, we are ready to support the BCCI. Once we officially hear from them, we will actively get into the preparation process, seeking necessary government approvals.”UAE’s domestic D10 League started from Friday and is slated to run till August 7 in Dubai and Usmani believes the tournament will help them prepare for the IPL. Usmani also elaborated on other procedures that will be put in place for the smooth running of the IPL.")
    sl.header("What will be the protocols for teams?")
    sl.write("We have set some standard protocols for the D10 League and if the teams want to come here a month in advance, we will extend all those protocols. We will take more precautions and make things even better, following government and ICC guidelines.")
    sl.header("Will there be spectators at venues?")
    sl.write("We will propose a few plans to our governments and will seek their approval on what protocols need to be followed to host the entire tournament. As far as fans’ entry is concerned, we would want our Asian diaspora in the UAE and also the Emiratis to come and watch the IPL. They are excited to see such a prestigious event. We will ask the government to allow some flexibility to ensure fans at the venues. The curfew has been lifted in Dubai and even tourists are allowed. (There are still protocols and restrictions in Abu Dhabi and Sharjah.).In restaurants, 30 to 50 per cent occupancy is permitted in Dubai, with proper social distancing and other safety measures. In malls and other places, the numbers are even higher, so we are hopeful of allowing some fans.")
    sl.header("Will there be travel restrictionns")
    sl.write("With UAE opening its airports and allowing tourists, a quarantine is not required if people travelling here carry a negative COVID-19 report. If your reports are negative then you are allowed to move freely (Tests will be arranged for those who fail to provide the report. Players and stakeholders will also need to download the DXB app to follow the health and safety protocol).")
    sl.header("What about the practice facilities?")
    sl.write("As an associate member nation, we have some of the best facilities. We have three grounds, in which there are two Ovals. The ICC Academy and the Abu Dhabi facility have their own practice grounds. The ICC Academy is the world’s biggest practice facility. In Dubai, the ICC Academy has 38 wickets, including a few simulated turfs. There are eight teams, but it won’t be a problem in facilitating their training.")
    sl.header("How will the weather be?")
    sl.write("In October-November, the weather is usually very pleasant. By the time IPL starts, winters will be approaching in this part of the world.")
