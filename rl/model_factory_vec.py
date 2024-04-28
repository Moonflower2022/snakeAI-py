from envs.snake_env_cnn import SnakeEnv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from helpers import replace_key_with_multiple
import json

def main():
    n_envs = 16
    model_type = "dqn"
    model_name = f"vec{n_envs}{model_type}4_5"

    width = 4
    height = 4
    starting_length = 4
    rewards = {'fruit': 1, 'lose': -1, 'win': 1, 'nothing': -0.001}
    # {'fruit': 1, 'lose': -1, 'win': 1, 'nothing': -0.001}
    # {'fruit': 1, 'lose': -1, 'win': 100, 'nothing': -0.01}
    # {'fruit': 1, 'lose': -1, 'win': 5, 'nothing': 0}
    # {'fruit': 1, 'lose': -1, 'win': 1, 'nothing': -0.001}

    policy = "CnnPolicy"

    starve = True
    no_backwards = True
    step_limit = 2 ** ((width + height) / 4) * 50
    gamma = 0.95
    ent_coef = 0.1
    learning_rate = 0.0008895296207610578
    time_steps = 5_000_000

    # for dqn only
    exploration_initial_eps = 0.6
    exploration_final_eps = 0.4
    exploration_fraction = 1

    env = SubprocVecEnv([lambda: Monitor(SnakeEnv(width=width, height=height, snake_length=starting_length, rewards=rewards, starve=starve, no_backwards=no_backwards, step_limit=step_limit)) for i in range(n_envs)])

    if model_type == "ppo":
        model = PPO(policy, env, verbose=1, n_steps=5, gamma=gamma, ent_coef=ent_coef, learning_rate=learning_rate)
    elif model_type == "a2c":
        model = A2C(policy, env, verbose=1, n_steps=5, gamma=gamma, ent_coef=ent_coef, learning_rate=learning_rate)
    elif model_type == "dqn":
        model = DQN(policy, env, verbose=1, gamma=gamma, 
                    exploration_fraction=exploration_fraction,
                    exploration_initial_eps=exploration_initial_eps, 
                    exploration_final_eps=exploration_final_eps,
                    learning_rate=learning_rate)
    else:
        raise Exception("bruh model isnt ppo a2c or dqn")

    model.learn(total_timesteps=time_steps)
    model.save(f"rl/{width}x{height}_models/{model_name}")

    rewards = [environment.get_episode_rewards() for environment in env] 

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
            "policy": policy,
            "gamma": gamma,
            "ent_coef": ent_coef, 
            "learning_rate": learning_rate,
            "time_steps": time_steps,
            f"ending_{eval_eposodes}_avg_rewards": sum(sum(reward[-eval_eposodes:]) for reward in rewards)/(eval_eposodes * n_envs) if all(len(reward) >= eval_eposodes for reward in rewards) else "not long enough",
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

if __name__ == '__main__':
    main()
