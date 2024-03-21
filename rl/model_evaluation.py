from snakes.snake_4 import Snake4
from snakes.snake_3 import Snake3
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN

width = 4
height = 4
starting_length = 3

model = PPO.load(f"rl/{width}x{height}_models/ppo4_24")
# model = DQN.load(f"rl/strong_models/ppo4_1")

test_env = Snake4(render_mode='train', width=width, height=height, snake_length=starting_length)

trials = 10000
wins = 0
total_win_moves = 0
total_snake_length = 0
stuck = 0

for i in range(trials):
    obs, info = test_env.reset()
    moves = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = test_env.step(int(action))
        if terminated:
            total_snake_length += len(test_env.snake)
            if info["won"]:
                wins += 1
                total_win_moves += moves
            break
        moves += 1
        if moves > 1000:
            stuck += 1
            break
            
            

test_env.close()

print("\"win %\": ", wins/trials, ",")
if wins != 0:
    print("\"avg moves to win\": ", total_win_moves/wins, ",")
else: 
    print("no wins")
print("\"avg ending snake length\": ", total_snake_length/trials, ",")
print("\"stuck %\": ", stuck/trials)