from snake_env import SnakeEnv
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

'''
vec_env = make_vec_env(SnakeEnv, n_envs=4)

model = A2C.load("a2c/a2c_snake")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated = vec_env.step(action)
    vec_env.render("human")



'''
model = A2C.load("a2c/a2c_snake")

test_env = SnakeEnv(render_mode='human', display_width=400, display_height=400, width=4, height=4, snake_length=4, FPS=5)

obs, info = test_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = test_env.step(int(action))
    test_env.render()
    if terminated:
        obs, info = test_env.reset()

test_env.close()
