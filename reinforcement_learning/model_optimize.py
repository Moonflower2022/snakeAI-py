import optuna
from snakes.snake_4 import Snake4
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

vec_env = make_vec_env(Snake4, n_envs=4)

def objective(trial):
    # Define hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    ent_coef = trial.suggest_uniform('ent_coef', 1e-4, 2e-2)
    gamma = trial.suggest_uniform('gamma', 0.97, 0.99)
    
    # Create A2C model with sampled hyperparameters
    model = PPO("MlpPolicy", vec_env, verbose=0, learning_rate=learning_rate, ent_coef=ent_coef, gamma=gamma, n_steps=5)
    
    # Train the model
    model.learn(total_timesteps=50000)
    
    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=10)
    
    return mean_reward

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    print("Best trial:")
    print(study.best_trial.params)
    print("Best trial value:", study.best_value)