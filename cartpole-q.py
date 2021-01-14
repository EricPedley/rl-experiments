import gym
import numpy as np
import random
import matplotlib.pyplot as plt
env = gym.make("CartPole-v1")
bin_size = 10

qtable = 1e-4*np.random.rand(bin_size,bin_size,bin_size,bin_size,2)

discount_rate=1
learning_rate=0.5
epsilon = 1.0


def makeDecision(state):
    state_discrete = int((state[0]+4.8)//(9.6/bin_size)),int((state[1]+4)//(8/bin_size)),int((state[2]+0.418)//(0.836/bin_size)),int((state[3]+4)//(8/bin_size))
    action_greedy = np.argmax(qtable[state_discrete[0],state_discrete[1],state_discrete[2],state_discrete[3]])
    action_random = random.randint(0,1)
    return action_greedy if random.random()>epsilon else action_random
def train(experience):
    state,action,next_state,reward,done=experience
    state_discrete = int((state[0]+4.8)//(9.6/bin_size)),int((state[1]+4)//(8/bin_size)),int((state[2]+0.418)//(0.836/bin_size)),int((state[3]+4)//(8/bin_size))
    state_next_discrete = int((next_state[0]+4.8)//(9.6/bin_size)),int((next_state[1]+4)//(8/bin_size)),int((next_state[2]+0.418)//(0.836/bin_size)),int((next_state[3]+4)//(8/bin_size))
    global epsilon,discount_rate,learning_rate,qtable

    q_next =  np.zeros(1) if done else qtable[state_next_discrete[0],state_next_discrete[1],state_next_discrete[2],state_next_discrete[3]]#next_state is state, but after the agent made an action
    q_target = reward +  discount_rate*np.max(q_next)#for a win, q_target will be 1, and -0.5 for a loss
    q_update = q_target - qtable[state_discrete[0],state_discrete[1],state_discrete[2],state_discrete[3],action]#this table entry is the one that was just used to make a decision and generate the next state
    qtable[state_discrete[0],state_discrete[1],state_discrete[2],state_discrete[3],action] += q_update*learning_rate#
    if done:
        epsilon*=0.99

total_reward=0
reward_history = []
episodes = 2000
for episode in range(episodes):
    done=False
    state = env.reset()
    iteration=0
    while not done:
        action = makeDecision(state)
        next_state, reward, done, info = env.step(action)#reward is one if the goal is reached and zero if the goal isn't reached
        # if episode>1980:
        #     env.render()
        iteration+=1
        reward=iteration/50
        train((state,action,next_state,reward,done))
        state=next_state
        total_reward+=1
    reward_history.append(total_reward)
    total_reward=0
plt.plot(reward_history)
plt.show()
#cart position and angle can be put into bins
#cart velocity and angular velocity typically aren't greater in magniture than 3.5, so make a function that is about equal to y=x for -3<x<3, then asymptotes at y=5 and y=-5, then put values into bins
