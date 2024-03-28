from snakes.snake_env import SnakeEnv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from helpers import replace_key_with_multiple
import json

# CnnPolicy

good_trials = [
{'learning_rate': 0.0009712152710668071, 'ent_coef': 0.013574795187369957, 'gamma': 0.9757006706208164},
{'learning_rate': 0.0007534396829162255, 'ent_coef': 0.019011104842401535, 'gamma': 0.9790678425582531},
{'learning_rate': 0.0008754703052793183, 'ent_coef': 0.019832299628796565, 'gamma': 0.9793731979429715},
{'learning_rate': 0.0009579932237818799, 'ent_coef': 0.01934918506617971, 'gamma': 0.98027651226877},
]


model_type = "a2c"

model_name = f"{model_type}4_19"

width = 4
height = 4
starting_length = 4
rewards = {'fruit': 1, 'lose': -1, 'win': 1, 'nothing': -0.001}

# {'fruit': 1, 'lose': -1, 'win': 1, 'nothing': -0.001}
# {'fruit': 1, 'lose': -1, 'win': 100, 'nothing': -0.01}
# {'fruit': 1, 'lose': -width*height, 'win': width*height, 'nothing': 0}
# {'fruit': 1, 'lose': -1, 'win': 1, 'nothing': -0.001}

starve = True
no_backwards = True
step_limit = 2 ** ((width + height)/4) * 50
gamma = 0.95
ent_coef = 0.02
learning_rate = 0.0008
time_steps = 6_000_000

# for dqn only
exploration_fraction = 1
exploration_initial_eps = 0.45
exploration_final_eps = 0.03




env = Monitor(SnakeEnv(width=width, height=height, snake_length=starting_length, rewards=rewards, starve=starve, no_backwards=no_backwards, step_limit=step_limit))

model = A2C("MlpPolicy", env, verbose=1, gamma=gamma, ent_coef=ent_coef, learning_rate=learning_rate)
'''
model = DQN("MlpPolicy", env, verbose=1, gamma=gamma, 
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps, 
            exploration_final_eps=exploration_final_eps,
            learning_rate=learning_rate)
'''
model.learn(total_timesteps=time_steps)
model.save(f"rl/{width}x{height}_models/{model_name}")

rewards = env.get_episode_rewards()

with open(f'rl/{width}x{height}_models/{model_name}_rewards.txt', 'w') as file:
    json.dump(rewards, file, indent=4)

# Load the existing JSON file
with open(f'rl/info/{width}x{height}_models_info.json', 'r') as file:
    data = json.load(file)

eval_eposodes = 500

info = {
    f"{model_name}": {
        "board_size": f"{width}x{height}",
        "starting_length": starting_length,
        "rewards": rewards,
        "starve": starve, 
        "step_limit": step_limit,
        "gamma": gamma,
        "ent_coef": ent_coef, 
        "learning_rate": learning_rate,
        "time_steps": time_steps,
        f"ending_{eval_eposodes}_avg_rewards": sum(rewards[-eval_eposodes:])/eval_eposodes if len(rewards) >= eval_eposodes else "not long enough",
        "notes": ""
    }
}

if model_type == "dqn":
    info = replace_key_with_multiple(info, 'ent_coef', {
        "exploration_fraction": exploration_fraction,
        "exploration_initial_eps": exploration_initial_eps, 
        "exploration_final_eps": exploration_final_eps,
    })

data.update(info)

# Write the updated JSON back to the file
with open(f'rl/info/{width}x{height}_models_info.json', 'w') as file:
    json.dump(data, file, indent=4)
