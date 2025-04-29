import numpy as np

# Entrega #4 - Dynamic Programming - TEMPLATE
#
# grid world, episodic task (terminal state)
# bellman equation, slide 29
# policy evaluation / prediction, for estimating V, page 27

actions = ['l', 'r', 'u', 'd']

# transition function: 
# maps current state s, and action a to next state ns and reward
# returns a tuple <next state>, <reward>
def act(s, a):
    ns = list(s)
    reward = -1

    # special case
    if s == [0,0] or s == [3,3]:
        return ns, # <YOUR CODE HERE>

    if a == 'r':
        ns[1] += 1
    elif a == 'l':
        ns[1] -= 1
    elif a == 'u':
        ns[0] -= 1
    elif a == 'd':
        ns[0] += 1

    # test for action taking outside the grid
    for i in [0,1]: # for both dimensions
        if ns[i] < 0:
            ns[i] = 0
        if ns[i] >= 4:
            ns[i] = 3
    return ns, reward

pi_as = 0.25  # pi(a/s), equiprobable random policy
gamma = 1 # discount factor

Delta = 10
k = 0
# mx, my = 4,4
value = np.zeros((4,4))

while Delta > 0.01:# <YOUR CODE HERE>:
    Delta = 0
    # for all states in state space (grid (i,j))
    for i in range(4):
        for j in range(4):
            s =[i,j]
            if s == [0,0] or s == [3,3]:
                continue
            v = value[i,j]
            # for all actions in action space
            v_new = 0
            for a in actions:
                snext, rw = act(s, a)
                v_new += pi_as * (rw + gamma * value[snext[0], snext[1]])
            value[i,j] = v_new
            Delta = max(Delta, abs(v - value[i,j]))
    k += 1

print("Value function after", k, "iterations:")
print(np.round(value, decimals=2))

