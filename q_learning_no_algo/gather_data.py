import numpy as np
from classes.medium_data_gathering import Game
import pickle
import os
import random
from hash_functions import hash_state
from training_functions import q_table_gathering_data

def decay_function(x):
    if x > 0:
        return max(0, x - 1)
    elif x < 0:
        return min(0, x + 1)
    else:
        return 0
    
def calculate_positive_percentage(inner_dict):
    total_values = sum(len(inner) for inner in inner_dict.values())
    positive_values = sum(1 for inner in inner_dict.values() for value in inner.values() if value > 0)
    percentage_positive = (positive_values / total_values) * 100 if total_values > 0 else 0
    return percentage_positive

import random

def generate_random_snake(snake_length, grid_length):
    """
    Generate a random snake of a given length on a grid.

    Parameters:
        length (int): The length of the snake.
        grid_width (int): The width of the grid.
        grid_height (int): The height of the grid.

    Returns:
        list: A list of coordinates representing the snake.
    """
    # Initialize the snake with a random starting position
    snake = [(random.randint(0, grid_length - 1), random.randint(0, grid_length - 1))]
    
    # Generate the rest of the snake's body
    while len(snake) < snake_length:
        last_position = snake[-1]

        options = [
            np.array((last_position[0], last_position[1] - 1)), 
            np.array((last_position[0], last_position[1] + 1)), 
            np.array((last_position[0] - 1, last_position[1])), 
            np.array((last_position[0] + 1, last_position[1]))
        ]

        viable_options = [option for option in options if not any(np.array_equal(snake_segment, option) for snake_segment in snake) and 0 <= option[0] < grid_length and 0 <= option[1] < grid_length]

        if len(viable_options) == 0:
            return generate_random_snake(snake_length, grid_length)
        snake.append(random.choice(viable_options))

    return np.array(snake)

def update_q_table(game, win_reward, lose_penalty):
    value = win_reward if game.won else lose_penalty

    for i in range(game.states):
        index = game.states - i - 1
        if not game.state_history[index] in q_table:
            q_table[game.state_history[index]] = {}
        if game.action_history[index] in q_table[game.state_history[index]]:
            q_table[game.state_history[index]][game.action_history[index]] = (1 - learning_rate) * q_table[game.state_history[index]][game.action_history[index]] + learning_rate * value
        else: 
            q_table[game.state_history[index]][game.action_history[index]] = value
        value = decay_function(value)

def update_q_table_split(game, win_reward, lose_penalty, is_first_half):
    value = win_reward if game.won else lose_penalty

    first_fruit_index = game.one_fruit_index + 1

    if is_first_half:
        state_history = game.state_history[:first_fruit_index]
        action_history = game.action_history[:first_fruit_index]
    else:
        state_history = game.state_history[first_fruit_index:]
        action_history = game.action_history[first_fruit_index:]
    
    #assert len(state_history) == len(action_history)
    
    for i in range(len(state_history)):
        if not state_history[-i - 1] in q_table:
            q_table[state_history[-i - 1]] = {}
        if action_history[-i - 1] in q_table[state_history[-i - 1]]:
            q_table[state_history[-i - 1]][action_history[-i - 1]] = (1 - learning_rate) * q_table[state_history[-i - 1]][action_history[-i - 1]] + learning_rate * value
        else: 
            q_table[state_history[-i - 1]][action_history[-i - 1]] = value
        value = decay_function(value)

# settings
starting_length = 5
board_length = 4
win_reward = 6
lose_penalty = -2
learning_rate = 0.3 # 0 < _ <= 1

dirname = os.path.dirname(__file__)
file_path = os.path.join(dirname, "Q_Tables/table14.pkl")
first = False

if first:
    q_table = {}
else:
    with open(file_path, 'rb') as file:
        q_table = pickle.load(file)

velocity = random.choice([np.array((1, 0)), np.array((-1, 0)), np.array((0, -1)), np.array((0, 1))])

for i in range(10000):
    snake = generate_random_snake(starting_length, board_length)
    game = Game(q_table_gathering_data, board_length, snake, velocity, q_table)

    while game.update():
        pass
    
    if game.won:
        update_q_table_split(game, win_reward, lose_penalty, True)
        #update_q_table_split(game, win_reward + 1, lose_penalty, False)
    else:
        update_q_table(game, win_reward, lose_penalty)

print(f"len: {len(q_table)}")

print(calculate_positive_percentage(q_table))

with open(file_path, 'wb') as file:
    pickle.dump(q_table, file)
