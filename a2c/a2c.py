from snake_env import SnakeEnv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env(SnakeEnv, n_envs=4)

model = A2C("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("a2c/a2c_snake")

'''
del model # remove to demonstrate saving and loading

model = PPO.load("a2c/a2c_snake")


env = SnakeEnv(render_mode='non_human', display_width=400, display_height=400, width=4, height=4, snake_length=4, FPS=5)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("a2c/a2c_snake")
'''