from snake_env_3action import SnakeEnv
import time
'''
env = SnakeEnv(render_mode='human', display_width=400, display_height=400,
               width=4, height=4, snake_length=4)

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        break

env.close()
'''

env = SnakeEnv(render_mode='human', display_width=400, display_height=400,
               width=4, height=4, snake_length=4, FPS=5)

while env.running:
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        env.reset()

env.close()