import gym
from gym.envs.registration import register
import numpy as np
import random
import os

try:
    register(
        id="FrozenLakeNoSlip-v0",
        entry_point="gym.envs.toy_text:FrozenLakeEnv",
        kwargs={'map_name': "4x4", "is_slippery": False},
        max_episode_steps=100,
        reward_threshold=0.78  # optimum = .8194
    )
except:
    pass
env = gym.make("FrozenLakeNoSlip-v0")
qtable = 1e-4*np.random.rand(16, 4)
qtable2 = np.copy(qtable)
discount_rate=0.9
learning_rate=0.1
epsilon = 1.0

def makeDecision(state):
    action_greedy = np.argmax(qtable[state])
    action_random = random.randint(0,3)
    return action_greedy if random.random()>epsilon else action_random
def makeDecision2(state):
    action_greedy = np.argmax(qtable2[state])
    action_random = random.randint(0,3)
    return action_greedy if random.random()>epsilon else action_random

def train(experience):
    global epsilon,discount_rate,learning_rate,qtable
    state,action,next_state,reward,done=experience
    q_next =  np.zeros(4) if done else qtable[next_state]#next_state is state, but after the agent made an action
    q_target = reward +  discount_rate*np.max(q_next)#for a win, q_target will be 1, and -0.5 for a loss
    q_update = q_target - qtable[state,action]#this table entry is the one that was just used to make a decision and generate the next state
    qtable[state,action] += q_update*learning_rate#

    if done:
        epsilon*=0.99

def train2(experience):
    global epsilon,discount_rate,learning_rate,qtable2
    state,action,next_state,reward,done=experience
    q_next =  np.zeros(4) if done else qtable2[next_state]#next_state is state, but after the agent made an action
    q_target = reward +  discount_rate*np.max(q_next)#for a win, q-next is 15 and the entry is a random low number
    q_update = q_target - qtable2[state,action]#this table entry is the one that was just used to make a decision and generate the next state
    qtable2[state,action] += q_update*learning_rate#

    if done:
        epsilon*=0.99

total_reward = 0
total_reward2 = 0

for episode in range(500):
    done = False
    state = env.reset()
    while not done:
        action = makeDecision(state)
        next_state, reward, done, info = env.step(action)#reward is one if the goal is reached and zero if the goal isn't reached
        train((state,action,next_state,reward,done))
        state=next_state
        total_reward+=reward


for episode in range(500):
    done = False
    state = env.reset()
    while not done:
        action = makeDecision2(state)
        next_state, reward, done, info = env.step(action)#reward is one if the goal is reached and zero if the goal isn't reached
        if done and reward==0:
            reward=-0.5
        train2((state,action,next_state,reward,done))
        state=next_state
        total_reward2+=reward
    #reward_history.append(total_reward/(episode+1))
# print(qtable-starting_table)
print(total_reward)
print(total_reward2)
#print(qtable)
#plt.plot(reward_history)
#plt.show()

