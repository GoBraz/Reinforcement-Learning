# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:58:58 2018
@author: Luiz
"""
import math
import matplotlib.pyplot as plt
from ai_LSTM2 import Dqn

# Defining function to create the walls
def Wall(t):
    T = 5
    w = (2*math.pi/T)   
    upperWall = 4 + 5*math.sin(w*t)
    bottonWall = -4 + 5*math.sin(w*t)
    upperRewardLim = 2 + 5*math.sin(w*t)
    bottonRewardLim = -2 + 5*math.sin(w*t)
    return upperWall, bottonWall, upperRewardLim, bottonRewardLim

# importing the class that applies DQN
brain = Dqn(2,3,0.9)

#initial position:
pos = 0
# how much the car can move up or down in a loop
ramp = 0.5
t = 0.0
reward = 0.0

# Declarating vectors to store information
V_upperWall = []
V_bottonWall = []
V_pos = []
V_bottonRewardLim = []
V_upperRewardLim = []
V_action = []
V_time = []
V_reward = []

# defining the loop to control the position of the car along time
while (True):
    t+=0.1
    
    #Getting positions of the wall and reward limits for each timestep
    upperWall, bottonWall, upperRewardLim, bottonRewardLim = Wall(t)
    action = int(brain.update(reward, [t,pos]))
    
    V_upperWall.append(upperWall)
    V_bottonWall.append(bottonWall)
    V_upperRewardLim.append(upperRewardLim)
    V_bottonRewardLim.append(bottonRewardLim)
    V_action.append(action)
    
    # Go up
    if action == 0:
        ramp = 1
    # stand still
    elif action == 1:
        ramp = 0
    # Go down
    elif action == 2:
        ramp = -1
    
    # Actualizing position of the agent
    pos = pos + ramp
    
    # defining the rewards for the agent 
    if bottonRewardLim <= pos <= upperRewardLim:
        reward = 1
    elif pos >= upperWall or pos <= bottonWall:
        reward = -3
        if pos >= upperWall : pos = upperWall
        if pos <= bottonWall : pos = bottonWall
    elif pos > bottonWall and pos < bottonRewardLim:
        reward = -1 
    elif pos < upperWall and pos > upperRewardLim:
        reward = -1

    V_reward.append(reward)
    V_pos.append(pos)
    V_time.append(t)
    
    if action == 0:
        print('subir')
    if action == 1:
        print('manter')
    if action == 2:
        print('descer')
    
    print('pos = ', pos)
    if t < 10:
        plt.plot(V_time,V_upperWall, color = 'black')
        plt.plot(V_time,V_bottonWall, color = 'black')
        plt.scatter(V_time[-1], pos, color = 'red')
        plt.plot(V_time,V_pos, color = 'red')
        plt.plot(V_time, V_upperRewardLim, color = 'green')
        plt.plot(V_time, V_bottonRewardLim, color = 'green')
        plt.plot(V_time, V_pos,color = 'red')
        plt.show()
        plt.clf()
    else:
        plt.plot(V_time[-100:], V_upperWall[-100:], color = 'black')
        plt.plot(V_time[-100:], V_bottonWall[-100:],color = 'black')
        plt.scatter(V_time[-1], pos, color = 'red')
        plt.plot(V_time[-100:], V_pos[-100:],color = 'red')
        plt.plot(V_time[-100:], V_upperRewardLim[-100:], color = 'green')
        plt.plot(V_time[-100:], V_bottonRewardLim[-100:], color = 'green')
        plt.plot(V_time[-100:], V_pos[-100:], color = 'red')
        plt.show()
        plt.clf()
