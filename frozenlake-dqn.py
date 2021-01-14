#!/usr/bin/env python
# coding: utf-8

# In[223]:


import gym
from gym.envs.registration import register
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from IPython.display import clear_output


# In[224]:



try:
    register(
        id="FrozenLakeNoSlip-v0",
        entry_point="gym.envs.toy_text:FrozenLakeEnv",
        kwargs={'map_name': "4x4", "is_slippery": False},
        max_episode_steps=20,
        reward_threshold=0.78  # optimum = .8194
    )
except:
    pass
env = gym.make("FrozenLakeNoSlip-v0")
discount_rate=0.9
learning_rate=0.01
epsilon = 1.0


# In[225]:


class CustomSquaredError(keras.losses.Loss):
  def call(self, y_true, y_pred):
    reduced_pred = tf.math.reduce_max(y_pred)#q value for selected action
    print("got past reduction")
    ans = tf.math.squared_difference(reduced_pred,y_true[0])#y_true is q_target
    bruh = tf.reduce_mean(tf.math.squared_difference(y_true,y_pred))
    print("got past squared difference")
    print(f"bruh: {bruh}, ans {ans}")
    return ans


# The custom squared error function was raising this error:
# 
# ValueError: Index out of range using input dim 0; input has only 0 dims for '{{node CustomMeanSquaredError/strided_slice}} = StridedSlice[Index=DT_INT32, T=DT_FLOAT, begin_mask=0, ellipsis_mask=0, end_mask=0, new_axis_mask=0, shrink_axis_mask=1](Cast, CustomMeanSquaredError/strided_slice/stack, CustomMeanSquaredError/strided_slice/stack_1, CustomMeanSquaredError/strided_slice/stack_2)' with input shapes: [], [1], [1], [1] and with computed input tensors: input[3] = <1>.
# 
# The fix was, instead of passsing in q_target as the true value, pass in a tensor containing q_target and reference its first index.

# In[226]:


model = keras.Sequential([
    layers.Dense(units=4, input_shape=[16])])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=CustomSquaredError()
)
model.summary()


# In[227]:


def makeDecision(state):
    action_greedy = np.argmax(model.predict(tf.one_hot([state],depth=16)))
    action_random = random.randint(0,3)
    return action_greedy if random.random()>epsilon else action_random
    


# In[228]:


def train(experience):
    global epsilon
    state,action,next_state,reward,done=experience
    q_next = model.predict(tf.one_hot([next_state],depth=16))[0]#next_state is state, but after the agent made an action
    #q_next is now the predicted q value for the next state, but wrapped like [[value]]
    if done:
        q_next = np.zeros(1)
    q_target = reward +  discount_rate*np.max(q_next)#for a win, q_target will be 1, and -0.5 for a loss
    current_state = tf.one_hot([state],depth=16)
    model.train_on_batch(x=current_state,y=tf.constant([q_target]))#q_target*tf.ones(shape=(4,)))
    
    if done:
        epsilon*=0.99


# IF the below snippet raises "ValueError: Creating variables on a non-first call to a function decorated with tf.function.", run the model initialization cell again and it should work

# In[ ]:


total_reward = 0
episodes = 100
verbose=False
for episode in range(episodes):
    done = False
    state = env.reset()
    print(f"episde: {episode}, reward: {total_reward}")
    while not done:
        action = makeDecision(state)
        next_state, reward, done, info = env.step(action)#reward is one if the goal is reached and zero if the goal isn't reached
#         if done and next_state !=15:
#             reward=-0.5
#         elif not done:
#             reward=-0.01
        train((state,action,next_state,reward,done))
        state=next_state
        total_reward+=reward
        if verbose:
            print(f"Epsiode: {episode}, total_reward: {total_reward}")
            env.render()
            print(model.layers[0].get_weights()[0])
            clear_output(wait=True)


print(total_reward)


# In[ ]:





# In[ ]:




