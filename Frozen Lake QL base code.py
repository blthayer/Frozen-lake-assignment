#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import gym
import random

env = gym.make("FrozenLake-v0")

action_size = env.action_space.n
state_size = env.observation_space.n


# In[ ]:


# Feel free to play with these hyperparameters

total_episodes = 15000        # Total episodes
test_episodes = 10            # Test episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.005            # Exponential decay rate for exploration prob


# In[ ]:


# Initializations
qtable = np.zeros((state_size, action_size))
rewards = []

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
        # Choose an action a in the current state (greedy or explore)
        
        exp_exp_tradeoff = random.uniform(0, 1)  
        # exploitation (taking the max Q value for this state)
        if exp_exp_tradeoff > epsilon:
            # Enter code here
            ## Hint: Greedily choose an action according to Q value

        # exploration
        else:
            # Enter code here
            ## Hint: Randomly choose an action

        # Take this action and observe
        new_state, reward, done, info = env.step(action)

        # Do a Q update
        # Enter code here
        ## Hint: One line update equation convert to one line code, start with "qtable[state, action] = ..."
        
        total_rewards += reward
        
        state = new_state
        
        if done == True: 
            break
        
    # Decay epsilon to reduce exploration as time progresses
    
    # Enter code here to assign a decay value to "decay_parameter"
    
    ## Hint: 
    ## 1. Use inbuilt polynomial, exponential(, or whatever works) functions to decay epsilon
    ## 2. "decay_parameter" is a function of "decay_rate" and "episode"
    
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*decay_parameter
    rewards.append(total_rewards)

print("Score over time: " +  str(sum(rewards)/total_episodes))
print("Q values:")
print(qtable)


# Q1. In short, explain why fixed "epsilon" above isn't the best choice? (Hint: You can keep epsilon fixed and see whether your reasoning explains the behavior)

# In[ ]:


########################################################################
#################### Final policy animation ############################
########################################################################

print("We only print the last state in each episode, to see if our agent has reached the destination or fallen into a hole")
env.reset()

for episode in range(test_episodes):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        # Taking action with Q learning
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        if done:
            env.render()
            
            print("Number of steps", step)
            break
        state = new_state
env.close()


# Q2. In some episodes above, the policy isn't reaching the goal, why?
