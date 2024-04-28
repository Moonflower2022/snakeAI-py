from envs.snake_env import SnakeEnv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from helpers import replace_key_with_multiple
from model_evaluation import evaluate_model
import json

model_type = "ppo"
model_name = f"test2"

trials = 5000

width = 4
height = 4
starting_length = 4
rewards = {'fruit': 1, 'lose': -1, 'win': width*height, 'nothing': -0.001}

# {'fruit': 1, 'lose': -1, 'win': 1, 'nothing': -0.001}
# {'fruit': 1, 'lose': -1, 'win': 100, 'nothing': -0.01}
# {'fruit': 1, 'lose': -width*height, 'win': width*height, 'nothing': 0}
# {'fruit': 1, 'lose': -1, 'win': 1, 'nothing': -0.001}

policy = "MlpPolicy"
starve = False
no_backwards = True
fruit_limit = width * height * 4
step_limit = 2 ** ((width + height)/4) * 50
gamma = 0.98
ent_coef = 0.01
learning_rate = 0.0008
time_steps = 10_000

# for dqn only
exploration_fraction = 1
exploration_initial_eps = 0.45
exploration_final_eps = 0.03

env = Monitor(SnakeEnv(width=width, height=height, snake_length=starting_length, rewards=rewards, starve=starve, no_backwards=no_backwards, step_limit=step_limit, fruit_limit=fruit_limit))


if model_type == "ppo":
    model = PPO(policy, env, verbose=1, gamma=gamma, ent_coef=ent_coef, learning_rate=learning_rate)
elif model_type == "a2c":
    model = A2C(policy, env, verbose=1, gamma=gamma, ent_coef=ent_coef, learning_rate=learning_rate)
elif model_type == "dqn":
    model = DQN(policy, env, verbose=1, gamma=gamma, 
                exploration_fraction=exploration_fraction,
                exploration_initial_eps=exploration_initial_eps, 
                exploration_final_eps=exploration_final_eps,
                learning_rate=learning_rate)
else:
    raise Exception("bruh the model_type is not ppo, a2c, or dqn, wtf are you doing :'(")

model.learn(total_timesteps=time_steps)
model.save(f"rl/{width}x{height}_models/{model_name}")

training_rewards = env.get_episode_rewards()

with open(f'rl/{width}x{height}_models/{model_name}_rewards.txt', 'w') as file:
    json.dump(training_rewards, file, indent=4)

# Load the existing JSON file
with open(f'rl/info/{width}x{height}_models_info.json', 'r') as file:
    data = json.load(file)

eval_eposodes = 500

info = {
    f"{model_name}": {
        "board_size": f"{width}x{height}",
        "starting_length": starting_length,
        "rewards": rewards,
        "no_backwards": no_backwards,
        "starve": starve, 
        "step_limit": step_limit,
        "fruit_limit": fruit_limit,
        "policy": policy,
        "gamma": gamma,
        "ent_coef": ent_coef, 
        "learning_rate": learning_rate,
        "time_steps": time_steps,
        f"ending_{eval_eposodes}_avg_rewards": sum(training_rewards[-eval_eposodes:])/eval_eposodes if len(training_rewards) >= eval_eposodes else "not long enough",
    }
}

if model_type == "dqn":
    info = replace_key_with_multiple(info, 'ent_coef', {
        "exploration_fraction": exploration_fraction,
        "exploration_initial_eps": exploration_initial_eps, 
        "exploration_final_eps": exploration_final_eps,
    })

env_args = {
    "width": width,
    "height": height,
    "starting_length": starting_length,
    "no_backwards": no_backwards,
    "step_limit": step_limit,
    "starve": False,
    "fruit_limit": False
}

info[model_name].update(evaluate_model(f"rl/{width}x{height}_models/{model_name}", trials, env_args))

info[model_name].update({"notes": ""})

data.update(info)

# Write the updated JSON back to the file
with open(f'rl/info/{width}x{height}_models_info.json', 'w') as file:
    json.dump(data, file, indent=4)
