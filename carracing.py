import gym
import numpy as np
import random
import matplotlib.pyplot as plt

#you need to pip install Box2d for 
#THE BOX2D ENVS DON'T INSTALL CORRECTLY UNLESS YOU HAVE SWIG INSTALLED
#ONE YOU INSTALL SWIG YOU NEED TO REINSTALL GYM
#AFTER SWIG YOU NEED TO INSTALL MICROSOFT VISUAL STUDIO BUILD TOOLS WITH THE SDK(THE SDK IS OPTIONAL AT INSTALL BUT YOU NEED IT)
try:
    env = gym.make("CarRacing-v0")
except AttributeError:
    print("(this is a custom error message written by me) Attribute Error: try installing box2d with 'pip install Box2d'. If you get an error doing that open this file and read the comment above this try/except block")
    exit(1)

def makeDecision(state):
    action_random = np.random.rand(3)
    return action_random#action_greedy if random.random()>epsilon else action_random


episodes = 100
for episode in range(episodes):
    done=False
    state = env.reset()
    iteration=0
    while not done:
        action = makeDecision(state)
        next_state, reward, done, info = env.step(action)#reward is one if the goal is reached and zero if the goal isn't reached

        #train((state,action,next_state,reward,done))
        env.render()
#cart position and angle can be put into bins
#cart velocity and angular velocity typically aren't greater in magniture than 3.5, so make a function that is about equal to y=x for -3<x<3, then asymptotes at y=5 and y=-5, then put values into bins
