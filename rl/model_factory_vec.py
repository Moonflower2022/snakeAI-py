from snakes_prevent.snake_4_starve import Snake4
from snakes.snake_3_2 import Snake3
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
import json

# CnnPolicy

good_trials = [
{'learning_rate': 0.0009712152710668071, 'ent_coef': 0.013574795187369957, 'gamma': 0.9757006706208164},
{'learning_rate': 0.0007534396829162255, 'ent_coef': 0.019011104842401535, 'gamma': 0.9790678425582531},
{'learning_rate': 0.0008754703052793183, 'ent_coef': 0.019832299628796565, 'gamma': 0.9793731979429715},
{'learning_rate': 0.0009579932237818799, 'ent_coef': 0.01934918506617971, 'gamma': 0.98027651226877},
]

n_envs = 16 # False if no vec

env_type = "4action"

model_name = f"vec{n_envs}a2c{env_type[0]}_3"

board_size = "6x6"
starting_length = 2
rewards_description = "1 for eating fruit, -1 for dying, 5 for winning, 1 for starving"
# "1 for eating fruit, -1 for dying, 100 for winning, -0.01 for nothing"
# "1 for eating fruit, -1 for dying, 5 for winning, 1 for starving"
# "1 for eating fruit, -5 for dying, 5 for winning, 0 for nothing"
gamma = 0.98
ent_coef = 0.2
exploration_initial_eps = 0.035
exploration_final_eps = 0.02
exploration_fraction = 1
learning_rate = 0.0008895296207610578
time_steps = 2_000_000

if model_name[:3] == "dqn":
    info = {
        f"{model_name}": {
            "board_size": f"{board_size}",
            "starting_length": starting_length,
            "env": f"{env_type}",
            "rewards": f"{rewards_description}",
            "gamma": gamma,
            "ent_coef": ent_coef,
            "learning_rate": learning_rate,
            "time_steps": time_steps,
            "strength": "",
            "notes": "backwards -> forwards"
        }
    }
else:
    info = {
        f"{model_name}": {
            "board_size": f"{board_size}",
            "starting_length": starting_length,
            "env": f"{env_type}",
            "rewards": f"{rewards_description}",
            "gamma": gamma,
            "exploration_fraction": exploration_fraction,
            "exploration_initial_eps": exploration_initial_eps, 
            "exploration_final_eps": exploration_final_eps,
            "learning_rate": learning_rate,
            "time_steps": time_steps,
            "strength": "",
            "notes": "backwards -> forwards"
        }
    }

# Snake4(render_mode='train', width=int(board_size[0]), height=int(board_size[2]), snake_length=starting_length) 
if env_type[0] == "4":
    env = make_vec_env(lambda: Snake4(width=int(board_size[0]), height=int(board_size[2]), snake_length=starting_length), n_envs)
else: # env_type[0] == "3"
    env = make_vec_env(lambda: Snake3(width=int(board_size[0]), height=int(board_size[2]), snake_length=starting_length), n_envs)
model = PPO("MlpPolicy", env, verbose=1, gamma=gamma, ent_coef=ent_coef, learning_rate=learning_rate)
'''
model = DQN("MlpPolicy", env, verbose=1, gamma=gamma, 
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps, 
            exploration_final_eps=exploration_final_eps,
            learning_rate=learning_rate)
            '''

model.learn(total_timesteps=time_steps)
model.save(f"rl/{board_size}_models/{model_name}")

envs = env.envs
rewards = [environment.get_episode_rewards() for environment in envs] 

with open(f'rl/{board_size}_models/{model_name}_rewards.txt', 'w') as file:
    json.dump(rewards, file, indent=4)

# Load the existing JSON file
with open(f'rl/info/{board_size}_models_info.json', 'r') as file:
    data = json.load(file)

data.update(info)

# Write the updated JSON back to the file
with open(f'rl/info/{board_size}_models_info.json', 'w') as file:
    json.dump(data, file, indent=4)
