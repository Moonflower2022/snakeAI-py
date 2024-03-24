import json
import matplotlib.pyplot as plt

n_envs = 4
actions = 4

board_size = "6x6"

with open(f'rl/{board_size}_models/vec{n_envs}ppo{actions}_1_rewards.txt', 'r') as file:
    rewards = json.load(file)

def average_at_intervals(data, interval):
    n = len(data)
    averages = []
    for i in range(0, n, interval):
        chunk = data[i:min(i+interval, n)]  # Handle the last chunk if its length is less than the interval
        if chunk:  # Check if the chunk is not empty
            averages.append(sum(chunk) / len(chunk))
    return averages

interval = 1000

for i in range(n_envs):
    plt.plot(average_at_intervals(rewards[i], interval), label=i)

plt.title('Rewards vs Iteration')
plt.xlabel('Iteration #')
plt.ylabel(f'Rewards (Averaged every {interval} iterations)')
plt.legend()

plt.show()