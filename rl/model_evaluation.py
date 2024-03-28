from snakes_prevent.snake_4 import Snake4
from snakes.snake_3 import Snake3
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN

width = 4
height = 4
starting_length = 8

model_name = "ppo4_40"

model = PPO.load(f"rl/{width}x{height}_models/{model_name}")
# model = PPO.load(f"rl/strong_models/ppo4_33")

test_env = Snake4(render_mode='train', width=width, height=height, snake_length=starting_length)

trials = 5000
wins = 0
total_win_moves = 0
total_snake_length = 0
stuck = 0

print("Starting evaluation...")

for i in range(trials):
    print(f"{i}/{trials}", end="\r")
    obs, info = test_env.reset()
    moves = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = test_env.step(int(action))
        if terminated or truncated:
            total_snake_length += len(test_env.snake)
        if terminated:
            if info["won"] == True:
                wins += 1
                total_win_moves += moves
            break
        moves += 1
        if truncated:
            stuck += 1
            break
            
            

test_env.close()

print("model:", model_name)

print("\"win ratio\": ", wins/trials, ",")
if wins != 0:
    print("\"avg moves to win\": ", total_win_moves/wins, ",")
else: 
    print("no wins")
print("\"avg ending snake length\": ", total_snake_length/trials, ",")
print("\"stuck ratio\": ", stuck/trials)