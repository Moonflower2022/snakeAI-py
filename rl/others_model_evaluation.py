from envs.snake_game_custom_wrapper_mlp import SnakeEnv
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import time

board_size = 4
starting_length = 3

model_name = "ppo4_42"

model = PPO.load(f"rl/other_models/{model_name}")
test_env = SnakeEnv(board_size=board_size, silent_mode=True)

trials = 5000
wins = 0
total_win_moves = 0
total_snake_length = 0

print("Starving evaluation...")

for i in range(trials):
    print(f"{i}/{trials}", end="\r")
    obs, info = test_env.reset()

    steps = 0
    
    while True:
        steps += 1
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = test_env.step(int(action))

        
        
        if terminated and info["snake_size"] == test_env.grid_size:
            wins += 1
            total_win_moves += steps

        if terminated: 
            total_snake_length += info["snake_size"]
            break

test_env.close()

print("model:", model_name)

print("\"win ratio\": ", wins/trials, ",")
if wins != 0:
    print("\"avg moves to win\": ", total_win_moves/wins, ",")
else: 
    print("no wins")
print("\"avg ending snake length\": ", total_snake_length/trials, ",")
