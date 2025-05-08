import numpy as np
import matplotlib.pyplot as plt

# Environment parameters
ROWS, COLS = 4, 12
START = (3, 0)
GOAL = (3, 11)
CLIFF = [(3, i) for i in range(1, 11)]

# Actions: 0=up, 1=right, 2=down, 3=left
ACTIONS = [0, 1, 2, 3]
ACTION_TO_DELTA = {
    0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)
}

# Hyperparameters
alpha = 0.5       # learning rate
gamma = 1.0       # discount factor
epsilon = 0.1     # exploration rate
episodes = 500

# Initialize Q-table
Q_q = np.zeros((ROWS, COLS, len(ACTIONS)))
Q_s = np.zeros((ROWS, COLS, len(ACTIONS)))

def step(state, action):
    """Apply action and return new state and reward."""
    delta = ACTION_TO_DELTA[action]
    new_state = (state[0] + delta[0], state[1] + delta[1])
    # stay in bounds
    new_state = (
        min(max(new_state[0], 0), ROWS - 1),
        min(max(new_state[1], 0), COLS - 1)
    )
    if new_state in CLIFF:
        return START, -100, True
    elif new_state == GOAL:
        return GOAL, 0, True
    else:
        return new_state, -1, False

def epsilon_greedy(state, Q):
    """Choose action using epsilon-greedy strategy."""
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)
    else:
        return np.argmax(Q[state[0], state[1]])

# Run Q-learning
episode_rewards_q = []
episode_rewards_s = []

for ep in range(episodes):
    state = START
    finish = False
    total_reward = 0

    while finish is False:
        action = epsilon_greedy(state, Q_q)
        next_state, reward, finish = step(state, action)

        # Q-learning update
        Q_q[state[0], state[1], action] += alpha * (
            reward + gamma * Q_q[next_state[0], next_state[1], np.argmax(Q_q[next_state[0], next_state[1]])]
            - Q_q[state[0], state[1], action]
        )
        state = next_state
        total_reward += reward

    episode_rewards_q.append(total_reward)
    state = START
    finish = False
    total_reward = 0
    action = epsilon_greedy(state, Q_s)

    while finish is False:
        next_state, reward, finish = step(state, action)
        next_action = epsilon_greedy(next_state, Q_s)

        Q_s[state[0], state[1], action] += alpha * (reward + gamma * Q_s[state[0], state[1], next_action] - Q_s[state[0], state[1], action])

        state = next_state
        action = next_action
        total_reward += reward
        
    episode_rewards_s.append(total_reward)


# Plot total reward per episode
plt.plot(episode_rewards_q, label='Q-learning')
plt.plot(episode_rewards_s, label='SARSA')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.title('Q-Learning on Cliff Walking')
plt.legend()
plt.grid(True)
plt.show()

# Plot the learned policy
def plot_policy(Q):
    direction_map = {
        0: '↑',  # up
        1: '→',  # right
        2: '↓',  # down
        3: '←'   # left
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.5, ROWS - 0.5)
    ax.set_xticks(np.arange(-0.5, COLS, 1))
    ax.set_yticks(np.arange(-0.5, ROWS, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    # Draw cliff cells
    for (r, c) in CLIFF:
        ax.add_patch(plt.Rectangle((c - 0.5, ROWS - r - 1 - 0.5), 1, 1, color='black'))

    # Draw start and goal
    sr, sc = START
    gr, gc = GOAL
    ax.text(sc, ROWS - sr - 1, 'S', ha='center', va='center', fontsize=14, color='blue')
    ax.text(gc, ROWS - gr - 1, 'G', ha='center', va='center', fontsize=14, color='green')

    # Draw arrows for best policy
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) == GOAL or (r, c) in CLIFF:
                continue
            best_action = np.argmax(Q[r, c])
            arrow = direction_map[best_action]
            ax.text(c, ROWS - r - 1, arrow, ha='center', va='center', fontsize=16, color='red')

    ax.set_title('Learned Policy (Q-learning)', fontsize=16)
    plt.show()

plot_policy(Q_q)
plot_policy(Q_s)