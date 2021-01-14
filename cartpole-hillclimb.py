import gym
import numpy as np
import matplotlib.pyplot as plt
# import gym_snake
# env = gym.make("snake-v0")
# print(env.observation_space)
# print(env.action_space)

import random
env = gym.make("CartPole-v1")
state = env.reset()

weights = np.random.rand(4,2)

maxSurvivalTime=0

delta = np.random.rand(4,2)
times = []
maxWeights=weights
multiplier=1
maxVel=0
for episode in range(100):#for each episode
    done=False
    survivalTime=0
    while not done:
        survivalTime+=1
        decision = np.argmax(np.dot(state,weights))
        action = decision
        state,reward,done,info = env.step(action)
        vel = state[1]
        if abs(vel)>maxVel:
            maxVel=abs(vel)
    env.reset()
    times.append(survivalTime)
    #print(f"episode {episode}: {survivalTime}")
    if survivalTime>=maxSurvivalTime:
        maxSurvivalTime=survivalTime
        maxWeights=weights
        multiplier/=2
    else:
        multiplier*=2
    delta = np.random.rand(4,2)
    weights=maxWeights+delta*multiplier
print(maxVel)
plt.plot(times)
plt.show()
    
