from snake_env_3action import SnakeEnv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = SnakeEnv(render_mode='non', display_width=400, display_height=400, width=4, height=4, snake_length=4, FPS=5)

# CnnPolicy

good_trials = [
{'learning_rate': 0.0009712152710668071, 'ent_coef': 0.013574795187369957, 'gamma': 0.9757006706208164},
{'learning_rate': 0.0007534396829162255, 'ent_coef': 0.019011104842401535, 'gamma': 0.9790678425582531},
{'learning_rate': 0.0008754703052793183, 'ent_coef': 0.019832299628796565, 'gamma': 0.9793731979429715},
{'learning_rate': 0.0009579932237818799, 'ent_coef': 0.01934918506617971, 'gamma': 0.98027651226877},
]

model = PPO("MlpPolicy", vec_env, verbose=1, gamma=0.98, ent_coef=0.01, learning_rate=0.0008895296207610578)
model.learn(total_timesteps=2000000)
model.save("a2c/models/a2c_snake6")

'''
del model # remove to demonstrate saving and loading

model = PPO.load("a2c/a2c_snake")


env = SnakeEnv(render_mode='non_human', display_width=400, display_height=400, width=4, height=4, snake_length=4, FPS=5)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("a2c/a2c_snake")
'''