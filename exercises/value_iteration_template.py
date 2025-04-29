import numpy as np
import matplotlib.pyplot as plt

# Entrega #4 - Dynamic Programming - TEMPLATE
#
# value iteration, Gambler's Problem
# slide 35

state_set = list(range(1,100))  # 1, 2, ..., 99

V = np.zeros( (len(state_set)+1) )

ph = 0.4    # probability of heads

# returns next_state and its probability
def next_states(s, a):
    return [
        (s+a, ph),  # head
        (s-a, 1-ph)  # tails
    ]

def expected_value(s, a):
    gamma = 1
    evalue = 0
    for snext, prob in next_states(s, a):
        # reward is 0 if action leads to state 0 or 100
        reward = 0 
        if snext == 100:
            reward = 1
            evalue += prob* reward
            # evalue += prob * (reward + gamma * V[snext-1])

        else:
            evalue += prob * (reward + gamma * V[snext])
    return evalue

Delta = 10
k = 0
theta = 1e-9
while Delta > theta:
    Delta = 0
    for s in state_set:
        # One step lookahead to find the best action for this state
        action_set = range(0, min(s, 100-s)+1)
        values_actions = [expected_value(s, a) for a in action_set]
        v = max(values_actions)
        Delta = max(Delta, abs(V[s] - v))
        V[s] = v
    k += 1

print(V)
plt.plot(V)
plt.grid(True)

def policy(s):
    action_set = range(0, min(s, 100-s)+1)
    # One step lookahead to find the best action for this state
    values_actions = [expected_value(s,a) for a in action_set]# <YOUR CODE HERE>
    return np.argmax(values_actions)  # 0,1,.... min(s,100-s)

final_policy = [policy(s) for s in state_set]
plt.figure()
plt.bar(state_set, final_policy, align='center', alpha=0.5)
plt.plot(state_set, final_policy,'.')
plt.grid(True)
plt.show()


