from envs.snake_env import SnakeEnv
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

width = 4
height = 4
starting_length = 4

model = PPO.load(f"rl/{width}x{height}_models/ppo4_41")

test_env = SnakeEnv(render_mode='human', width=width, height=height, snake_length=starting_length, no_backwards=True)

'''
total_rewards = 0
total_moves = 0

obs, info = test_env.reset()

    while not test_env.quit:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = test_env.step(int(action))
        total_rewards += rewards
        total_moves += 1
        test_env.render()
        if terminated:
            print("total_rewards: ", total_rewards)
            print("total_moves: ", total_moves)
            obs, info = test_env.reset()
            total_rewards = 0
            total_moves = 0
'''



for i in range(10):
    obs, info = test_env.reset()
    
    while not test_env.quit:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = test_env.step(int(action))
        test_env.render()
        if terminated or truncated:
            break

test_env.close()