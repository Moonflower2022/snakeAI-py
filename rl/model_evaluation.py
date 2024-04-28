from envs.snake_env import SnakeEnv
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN

def evaluate_model(model_path, trials, env_args):
    model = PPO.load(model_path)
    # model = PPO.load(f"rl/strong_models/ppo4_33")

    test_env = SnakeEnv(render_mode='train', width=env_args['width'], height=env_args['height'], snake_length=env_args['starting_length'], no_backwards=env_args['no_backwards'], starve=env_args['starve'], step_limit=env_args['step_limit'], fruit_limit=env_args['fruit_limit'])

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
            action, _states = model.predict(obs, deterministic=False)
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
                
    return {
        "win ratio": wins/trials,
        "avg moves to win": "no wins" if wins == 0 else total_win_moves/wins,
        "avg ending snake length": total_snake_length/trials,
        "stuck ratio": stuck/trials
    }