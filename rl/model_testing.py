from snakes.snake_4 import Snake4
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

width = 4
height = 4

model = A2C.load(f"rl/{width}x{height}_models/a2c4_6")

test_env = Snake4(render_mode='human', width=width, height=height, snake_length=4)

obs, info = test_env.reset()

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

test_env.close()