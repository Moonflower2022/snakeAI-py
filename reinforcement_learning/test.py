from snakes.snake_3 import Snake3
import numpy as np

env = Snake3(render_mode='human')

m = []

for i in range(100):
    env.reset()
    m.append(np.count_nonzero(env._get_state() == 1))

print(m)