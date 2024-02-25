from p5 import *
import numpy as np
import os
from classes.simplified_game import Game
from hash_functions import hash_state
from joblib import load
import sklearn

def funny_move_function(game):
   # Get the current position of the snake
   snake_position = game.snake[0]

   # Get the position of the food
   food_position = game.fruit

   # Calculate the direction to the food
   direction = food_position - snake_position

   # Normalize the direction vector
   magnitude = np.linalg.norm(direction)
   direction = direction / magnitude

   # Round the direction vector to the nearest allowed direction
   direction = np.round(direction)

   # Return the direction
   return direction

def random_move_function(game):
    magnitudes = [1, -1]
    
    options = [np.array([i, 0]) for i in magnitudes] + [np.array([0, i]) for i in magnitudes]

    options = [option for option in options if not np.array_equal(option, -game.velocity)]
    
    return np.array(options[np.random.choice(len(options))])

def q_table_move_function(game):
    if hash_state(game.snake, game.fruit) in game.q_table:
        print(game.q_table[hash_state(game.snake, game.fruit)])
        print(np.array(max(game.q_table[hash_state(game.snake, game.fruit)], key=game.q_table[hash_state(game.snake, game.fruit)].get)))
        return np.array(max(game.q_table[hash_state(game.snake, game.fruit)], key=game.q_table[hash_state(game.snake, game.fruit)].get))
    else:
        print("confused")
        return random_move_function(game)
    
def minimize_taxi(game):
    def distance(tuple1, tuple2):
        return abs(tuple1[0] - tuple2[0]) + abs(tuple1[1] - tuple2[1])
            
    magnitudes = [1, -1]
    
    options = [np.array([i, 0]) for i in magnitudes] + [np.array([0, i]) for i in magnitudes]

    options = [option for option in options if not np.array_equal(option, -game.velocity)]

    distances = [distance(game.snake[len(game.snake) - 1] + option, game.fruit) for option in options]

    return options[np.argmin(distances)]

def model_move(game):
    def convert_state_to_2d(state, length):
        snake, fruit = state
        snake_2d = np.zeros(length**2)
        for segment in snake:
            snake_2d[segment[0] + length*segment[1]] = 1
        fruit_2d = np.zeros(length**2)
        fruit_2d[fruit[0] + length*fruit[1]] = 1
        return np.concatenate((snake_2d, fruit_2d))

    def convert_action_to_2d(action):
        action_2d = np.zeros(4)
        action_2d[{(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}[action]] = 1
        return action_2d
    
    options = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    model_input = convert_state_to_2d(hash_state(game.snake, game.fruit), game.board_length)

    values = [game.model.predict(np.concatenate((model_input, convert_action_to_2d(action)), axis=0).reshape(1, -1)) for action in options]
    return np.array(options[np.argmax(values)])
    

dirname = os.path.dirname(__file__)
file_path = os.path.join(dirname, "Q_Tables/table13.pkl")

with open(file_path, 'rb') as file:
    q_table = pickle.load(file)
    
model = None #load('model.joblib')

tiles = 4
tile_size = 50
background_color = Color(24)
gameover_overlay_color = Color(88, 127)

snake_color = Color(161, 181, 108)
head_color = Color(102, 204, 0)
food_color = Color(171, 70, 66)

start_pos = (4, 4)
snake = np.array([np.array([start_pos[0] - 4, start_pos[1] - 4])] + [np.array([start_pos[0] - 4, start_pos[1] - 3])] + [np.array([start_pos[0] - 4, start_pos[1] - 2])] + [np.array([start_pos[0] + i - 4, start_pos[1] - 1]) for i in range(4)])
#snake = np.array([np.array([start_pos[0] - 4, start_pos[1] - 3])] + [np.array([start_pos[0] - 4, start_pos[1] - 2])] + [np.array([start_pos[0] + i - 4, start_pos[1] - 1]) for i in range(4)])
#snake = np.array([np.array([start_pos[0] - 4, start_pos[1] - 2])] + [np.array([start_pos[0] + i - 4, start_pos[1] - 1]) for i in range(4)])
#snake = np.array([np.array([start_pos[0] + i - 4, start_pos[1] - 1]) for i in range(4)])

#start_pos = (4, 1)
#snake = np.array([np.array([start_pos[0] - 4, start_pos[1] +1 ])] + [np.array([start_pos[0] - 4, start_pos[1]])] + [np.array([start_pos[0] + i - 4, start_pos[1] - 1]) for i in range(4)])

game = Game(q_table_move_function, tiles, snake, q_table, model)

proxy_score = 0

def draw_game(game):
    background(background_color)

    fill(snake_color)
    for i in range(len(game.snake)):
        if i == len(game.snake) - 1:
            fill(head_color)
        square(Vector(game.snake[i][0], game.snake[i][1]) * tile_size, 0.8 * tile_size)

    fill(food_color)
    square(Vector(game.fruit[0], game.fruit[1]) * tile_size, 0.8 * tile_size)

def setup():
    size(tile_size * tiles, tile_size * tiles)
    title("p5 Snake")

    no_stroke()
    draw_game(game)

def draw():
    global game
    global proxy_score
    game.update()
    '''
    if proxy_score != game.score:
        print(game.score)
        proxy_score += 1
        start_pos = (4, 4)
        snake_length = 4
        game.snake = np.array([np.array([start_pos[0] + i - snake_length, start_pos[1]]) for i in range(snake_length)])
        game.velocity = np.array([1, 0])
    '''
    '''
    if game.score != 0 or game.game_over:
        game = Game(q_table_move_function, tiles, (4, 4), 4, q_table)
    '''

    draw_game(game)

if __name__ == '__main__':
    run(frame_rate=4)