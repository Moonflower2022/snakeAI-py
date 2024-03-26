import optuna
from snakes.snake_4_2 import Snake4
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

env = Snake4

def objective(trial):
    # Define hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
    exploration_initial_eps = trial.suggest_float('ent_coef', 1e-4, 2e-2)
    gamma = trial.suggest_float('gamma', 0.97, 0.99)
    
    # Create A2C model with sampled hyperparameters
    model = DQN("MlpPolicy", env(), verbose=0, learning_rate=learning_rate, exploration_initial_eps=exploration_initial_eps, exploration_final_eps=0.01, gamma=gamma)
    
    # Train the model
    model.learn(total_timesteps=250000)
    
    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env(), n_eval_episodes=100)
    
    return mean_reward

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    print("Best trial:")
    print(study.best_trial.params)
    print("Best trial value:", study.best_value)