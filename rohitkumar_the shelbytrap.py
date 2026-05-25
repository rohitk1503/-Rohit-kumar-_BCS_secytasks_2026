import numpy as np
import random

def take_action(state, action):
    if state == 0:
        if action == 0:
            next_state = 1 if random.random() < 0.8 else 2
        else:
            next_state = 2 if random.random() < 0.8 else 1
        return next_state, 0
    elif state == 1:
        reward = 1 if random.random() < 0.8 else 0
        return None, reward
    elif state == 2:
        reward = 1 if random.random() < 0.2 else 0
        return None, reward

Q = np.zeros((3, 2))

alpha = 0.1
gamma = 0.9
epsilon = 1.0
episodes = 1000

for episode in range(episodes):
    state = 0
    while state is not None:
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(Q[state])
        next_state, reward = take_action(state, action)
        if next_state is None:
            Q[state][action] += alpha * (reward - Q[state][action])
        else:
            Q[state][action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state][action]
            )
        state = next_state
    epsilon = max(0.01, epsilon * 0.995)

Q_shelby = np.zeros((3, 2))
Q_shelby[0][0] += alpha * (0 + gamma * np.max(Q_shelby[2]) - Q_shelby[0][0])
Q_shelby[2][0] += alpha * (1 - Q_shelby[2][0])
Q_shelby[0][0] += alpha * (0 + gamma * np.max(Q_shelby[2]) - Q_shelby[0][0])

V_S1 = 0.8
V_S2 = 0.2
V_A1 = 0.8 * V_S1 + 0.2 * V_S2
V_A2 = 0.8 * V_S2 + 0.2 * V_S1

lambda_val = 0.5
Q_hybrid_A1 = (1 - lambda_val) * Q_shelby[0][0] + lambda_val * V_A1
Q_hybrid_A2 = (1 - lambda_val) * Q_shelby[0][1] + lambda_val * V_A2

print("Final Q-table:", Q)
print(f"Shelby - A1: {Q_shelby[0][0]:.4f}, A2: {Q_shelby[0][1]:.4f}")
print(f"Model-Based - A1: {V_A1:.2f}, A2: {V_A2:.2f}")
print(f"Hybrid - A1: {Q_hybrid_A1:.4f}, A2: {Q_hybrid_A2:.4f}")