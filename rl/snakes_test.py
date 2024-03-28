from snakes.snake_env import SnakeEnv
from stable_baselines3.common.env_checker import check_env

print(check_env(SnakeEnv()))