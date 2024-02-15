import numpy as np
from classes.simplified_data_gathering import Game
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

        viable_options = [option for option in options if 0 <= option[0] < grid_length and 0 <= option[1] < grid_length]

        if len(viable_options) == 0:
            return generate_random_snake(snake_length, grid_length)
        snake.append(random.choice(viable_options))

    return np.array(snake)


# settings
snake_length = 7
board_length = 4
use_distance = False
win_reward = 6
lose_penalty = -3
learning_rate = 0.3 # 0 < _ <= 1
score_threshold = 3 # how many fruits to end the game

dirname = os.path.dirname(__file__)
file_path = os.path.join(dirname, "Q_Tables/table12.pkl")
first = False

if first:
    q_table = {}
else:
    with open(file_path, 'rb') as file:
        q_table = pickle.load(file)


velocity = random.choice([np.array((1, 0)), np.array((-1, 0)), np.array((0, -1)), np.array((0, 1))])

for i in range(100000):
    snake = generate_random_snake(snake_length, board_length)
    game = Game(q_table_gathering_data, board_length, snake, velocity, q_table, score_threshold)

    while game.update():
        pass
    
    if game.won:
        value = game.distance if use_distance else win_reward
    else: 
        value = lose_penalty

    for i in range(game.states):
        index = game.states - i - 1
        if not game.state_history[index] in q_table:
            q_table[game.state_history[index]] = {}
        if game.action_history[index] in q_table[game.state_history[index]]:
            q_table[game.state_history[index]][game.action_history[index]] = (1 - learning_rate) * q_table[game.state_history[index]][game.action_history[index]] + learning_rate * value
        else: 
            q_table[game.state_history[index]][game.action_history[index]] = value
        value = decay_function(value)

print(f"len: {len(q_table)}")

print(calculate_positive_percentage(q_table))
'''
# Create a new dictionary from the shuffled items
items_list = list(q_table.items())

# Shuffle the list
random.shuffle(items_list)

# Create a new dictionary from the shuffled list
shuffled_dict = dict(items_list)

# Set the limit for the number of values to print
limit = 10

# Counter to keep track of the number of values printed
count = 0

# Iterate through the dictionary items and print values up to the limit
for key, value in q_table.items():
    print(f'{key}: {value}')
    count += 1

    # Check if the limit is reached
    if count == limit:
        break
'''
with open(file_path, 'wb') as file:
    pickle.dump(q_table, file)
