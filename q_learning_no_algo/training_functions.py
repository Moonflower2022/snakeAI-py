import numpy as np
import random
from hash_functions import hash_state

def random_move_function(game, *args):
    move_magnitudes = [1, -1]
    
    options = [np.array([i, 0]) for i in move_magnitudes] + [np.array([0, i]) for i in move_magnitudes]

    options = [option for option in options if not np.array_equal(option, -game.velocity)]
    
    if args:
        options = [option for option in options if not tuple(option) in args]

    if len(options) == 0:
        return game.velocity
    return np.array(options[np.random.choice(len(options))])

def q_table_basic(game):
    if not hash_state(game.snake, game.fruit) in game.q_table:
        return random_move_function(game)
    else:
        return np.array(max(game.q_table[hash_state(game.snake, game.fruit)], key=game.q_table[hash_state(game.snake, game.fruit)].get))

def q_table_gathering_data(game):
    if not hash_state(game.snake, game.fruit) in game.q_table:
        return random_move_function(game)
    positive_keys = [key for key, value in game.q_table[hash_state(game.snake, game.fruit)].items() if value > 0]
    if positive_keys:
        return np.array(random.choice(positive_keys))
    
    negative_keys = [key for key, value in game.q_table[hash_state(game.snake, game.fruit)].items() if value < 0]
    return random_move_function(game, *negative_keys)