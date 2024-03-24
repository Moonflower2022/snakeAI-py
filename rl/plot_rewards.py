import json
import matplotlib.pyplot as plt

with open(f'rl/6x6_models/ppo4_5_rewards.txt', 'r') as file:
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
averages = average_at_intervals(rewards, interval)

plt.plot(averages)

plt.title('Rewards vs Iteration')
plt.xlabel('Iteration #')
plt.ylabel(f'Rewards (Averaged every {interval} iterations)')

plt.show()