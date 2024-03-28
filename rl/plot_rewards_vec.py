import json
import matplotlib.pyplot as plt
from helpers import rolling_averages

n_envs = 16
actions = 4

board_size = "4x4"

with open(f'rl/{board_size}_models/vec{n_envs}a2c{actions}_5_rewards.txt', 'r') as file:
    rewards = json.load(file)



interval = 1000

for i in range(n_envs):
    plt.plot(rolling_averages(rewards[i], interval), label=i)

plt.title('Rewards vs Iteration')
plt.xlabel('Iteration #')
plt.ylabel(f'Rewards (Averaged every {interval} iterations)')
plt.legend()

plt.show()