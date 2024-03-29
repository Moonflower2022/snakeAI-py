import json
import matplotlib.pyplot as plt
from helpers import rolling_averages

with open(f'rl/4x4_models/a2c4_19_rewards.txt', 'r') as file:
    rewards = json.load(file)


interval = 1000
averages = rolling_averages(rewards, interval)

plt.plot(averages)

plt.title('Rewards vs Iteration')
plt.xlabel('Iteration #')
plt.ylabel(f'Rewards (Averaged every {interval} iterations)')

plt.show()