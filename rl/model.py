from snakes.snake_4 import Snake4
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# CnnPolicy

good_trials = [
{'learning_rate': 0.0009712152710668071, 'ent_coef': 0.013574795187369957, 'gamma': 0.9757006706208164},
{'learning_rate': 0.0007534396829162255, 'ent_coef': 0.019011104842401535, 'gamma': 0.9790678425582531},
{'learning_rate': 0.0008754703052793183, 'ent_coef': 0.019832299628796565, 'gamma': 0.9793731979429715},
{'learning_rate': 0.0009579932237818799, 'ent_coef': 0.01934918506617971, 'gamma': 0.98027651226877},
]

model_name = "ppo4_3_unbugged"

board_size = "4x4"
starting_length = 4
env_type = "4action"
rewards_description = "-10 for dying, 10 for getting fruit, -0.00001 if neither"
gamma = 0.98
ent_coef = 0.005
learning_rate = 0.0008895296207610578

info = {
    f"{model_name}": {
        "board_size": f"{board_size}",
        "starting_length": starting_length,
        "env": f"{env_type}",
        "rewards": f"{rewards_description}",
        "gamma": gamma,
        "ent_coef": ent_coef,
        "learning_rate": learning_rate,
        "strength": "",
        "notes": ""
    }
}

# Parallel environments
env = Snake4(render_mode='train', width=int(board_size[0]), height=int(board_size[2]), snake_length=starting_length) 

model = PPO("MlpPolicy", env, verbose=1, gamma=gamma, ent_coef=ent_coef, learning_rate=learning_rate)
model.learn(total_timesteps=500000)
model.save(f"/{board_size}models/{model_name}")

import json

# Load the existing JSON file
with open(f'stable_baselines/{board_size}models/info.json', 'r') as file:
    data = json.load(file)

data.update(info)

# Write the updated JSON back to the file
with open(f'stable_baselines/{board_size}models/info.json', 'w') as file:
    json.dump(data, file, indent=4)
