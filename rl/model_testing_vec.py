from snakes.snake_4_2 import Snake4
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

width = 4
height = 4
starting_length = 4

model = DQN.load(f"rl/{width}x{height}_models/vec16dqn4_1")

test_env = DummyVecEnv([lambda: Snake4(render_mode='human')])

obs = test_env.reset()

# up down left right
action_map = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}

action_map2 = {
    0: "counterclock",
    1: "none",
    2: "clock"
}

total_rewards = 0
total_moves = 0

while True:
    action = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncate = test_env.step(action)
    test_env.render()
    if terminated:
        obs = test_env.reset()


test_env.close()