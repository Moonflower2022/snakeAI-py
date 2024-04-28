from envs.snake_game_custom_wrapper_mlp import SnakeEnv
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from helpers import replace_key_with_multiple
import json

def main():
    model_type = "ppo"
    model_name = f"{model_type}4_44"

    n_envs = 1

    board_size = 6
    starting_length = 3
    gamma = 0.98
    ent_coef = 0.01
    learning_rate = 0.0008
    time_steps = 4_000_000

    # for dqn only
    exploration_fraction = 1
    exploration_initial_eps = 0.45
    exploration_final_eps = 0.03

    # env = SubprocVecEnv([lambda: Monitor(SnakeEnv(board_size=board_size)) for i in range(n_envs)])
    env = Monitor(SnakeEnv(board_size=board_size))

    if model_type == "ppo":
        model = PPO("MlpPolicy", env, verbose=1, gamma=gamma, ent_coef=ent_coef, learning_rate=learning_rate)
    elif model_type == "a2c":
        model = A2C("MlpPolicy", env, verbose=1, gamma=gamma, ent_coef=ent_coef, learning_rate=learning_rate)
    elif model_type == "dqn":
        model = DQN("MlpPolicy", env, verbose=1, gamma=gamma, 
                    exploration_fraction=exploration_fraction,
                    exploration_initial_eps=exploration_initial_eps, 
                    exploration_final_eps=exploration_final_eps,
                    learning_rate=learning_rate)
    else:
        raise Exception("bruh the model_type is not ppo, a2c, or dqn, wtf are you doing :'(")

    model.learn(total_timesteps=time_steps)
    model.save(f"rl/other_models/{model_name}")

    training_rewards = env.get_episode_rewards()

    with open(f'rl/other_models/{model_name}_rewards.txt', 'w') as file:
        json.dump(training_rewards, file, indent=4)

    # Load the existing JSON file
    with open(f'rl/info/other_models_info.json', 'r') as file:
        data = json.load(file)

    eval_eposodes = 500

    info = {
        f"{model_name}": {
            "board_size": f"{board_size}x{board_size}",
            "starting_length": starting_length,
            "gamma": gamma,
            "ent_coef": ent_coef, 
            "learning_rate": learning_rate,
            "time_steps": time_steps,
            f"ending_{eval_eposodes}_avg_rewards": sum(training_rewards[-eval_eposodes:])/eval_eposodes if len(training_rewards) >= eval_eposodes else "not long enough",
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
    with open(f'rl/info/other_models_info.json', 'w') as file:
        json.dump(data, file, indent=4)
if __name__ == '__main__':
    main()