from p5 import *
import numpy as np
import os
import random
from classes.game import Game

def random_move_function(game):
    magnitudes = [1, -1]
    
    options = [np.array([i, 0]) for i in magnitudes] + [np.array([0, i]) for i in magnitudes]

    options = [option for option in options if not np.array_equal(option, -game.velocity)]
    
    return np.array(options[np.random.choice(len(options))])
    
def minimize_taxi(game):
    def distance(tuple1, tuple2):
        return abs(tuple1[0] - tuple2[0]) + abs(tuple1[1] - tuple2[1])
            
    magnitudes = [1, -1]
    
    options = [np.array([i, 0]) for i in magnitudes] + [np.array([0, i]) for i in magnitudes]

    options = [option for option in options if not np.array_equal(option, -game.velocity)]

    distances = [distance(game.snake[len(game.snake) - 1] + option, game.fruit) for option in options]

    return options[np.argmin(distances)]

def hamiltonian_cycle_from_point(grid, start_row, start_col):
    def is_valid_move(grid, path, row, col):
        # Check if the move is within the grid and the cell is not already visited
        if (row >= 0 and row < len(grid[0])) and (col >= 0 and col < len(grid)):
            if (row, col) not in path:
                return True
        return False
    def dfs(grid, path, row, col):
        path.append((row, col))

        if len(path) == len(grid) * len(grid[0]):
            # Check if the last cell is adjacent to the start cell
            last_row, last_col = path[-1]
            if abs(last_row - start_row) + abs(last_col - start_col) == 1:
                return True
        
        # Possible moves
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(moves)

        for move in moves:
            new_row = row + move[0]
            new_col = col + move[1]
            if is_valid_move(grid, path, new_row, new_col):
                if dfs(grid, path, new_row, new_col):
                    return True
        
        path.pop() # Backtrack
        return False

    # Start from the specified point and try to find a Hamiltonian cycle
    path = []
    if dfs(grid, path, start_row, start_col):
        return path
    raise Exception("Oopsie the rows and columns are both odd or smth idk but it no work sad fact :(")

def grid_generator(width, height):
    grid = []
    for i in range(height):
        grid.append(list(np.arange(width) + 1 + i*width))
    return grid

def cycle_to_directions(cycle):
    directions = {}
    for i in range(len(cycle)):
        current_point = cycle[i]
        next_point = cycle[(i + 1) % len(cycle)]  # Wrap around to the first point if we're at the last point
        direction = (next_point[0] - current_point[0], next_point[1] - current_point[1])
        directions[current_point] = direction
    return directions

def direction_move(game):
    return np.array(game.directions[tuple(game.snake[len(game.snake) - 1])])

width = 6
height = 5
tile_size = 50
background_color = Color(24)
gameover_overlay_color = Color(88, 127)

snake_color = Color(161, 181, 108)
head_color = Color(102, 204, 0)
food_color = Color(171, 70, 66)

start_pos = (4, 4)
#7
#snake = np.array([np.array([start_pos[0] - 4, start_pos[1] - 4])] + [np.array([start_pos[0] - 4, start_pos[1] - 3])] + [np.array([start_pos[0] - 4, start_pos[1] - 2])] + [np.array([start_pos[0] + i - 4, start_pos[1] - 1]) for i in range(4)])
#6
#snake = np.array([np.array([start_pos[0] - 4, start_pos[1] - 3])] + [np.array([start_pos[0] - 4, start_pos[1] - 2])] + [np.array([start_pos[0] + i - 4, start_pos[1] - 1]) for i in range(4)])
#5
#snake = np.array([np.array([start_pos[0] - 4, start_pos[1] - 2])] + [np.array([start_pos[0] + i - 4, start_pos[1] - 1]) for i in range(4)])
#4
#snake = np.array([np.array([start_pos[0] + i - 4, start_pos[1] - 1]) for i in range(4)])

#1
snake = np.array([np.array([0, 0])])

#start_pos = (4, 1)
#snake = np.array([np.array([start_pos[0] - 4, start_pos[1] +1 ])] + [np.array([start_pos[0] - 4, start_pos[1]])] + [np.array([start_pos[0] + i - 4, start_pos[1] - 1]) for i in range(4)])

grid = grid_generator(width, height)

cycle = hamiltonian_cycle_from_point(grid, snake[len(snake) - 1][1], snake[len(snake) - 1][0])

directions = cycle_to_directions(cycle)

game = Game(direction_move, width, height, snake, cycle, directions)

def draw_game(game):
    if game.game_over:
        background(gameover_overlay_color)
    else:
        background(background_color)

    # Draw the path of the Hamiltonian cycle
    stroke(255, 0, 0)  # Red color for the path
    stroke_weight(2)   # Set the thickness of the line
    for i in range(len(game.cycle)):
        current_point = game.cycle[i]
        next_point = game.cycle[(i + 1) % len(game.cycle)]  # Wrap around to the first point if we're at the last point
        line(Vector(current_point[0], current_point[1]) * int(tile_size) + Vector(0.5 * int(tile_size), 0.5 * int(tile_size)),
             Vector(next_point[0], next_point[1]) * int(tile_size) + Vector(0.5 * int(tile_size), 0.5 * int(tile_size)))
    no_stroke()
    # Draw the snake and food
    fill(snake_color)
    for i in range(len(game.snake)):
        if i == len(game.snake) - 1:
            fill(head_color)
        square(Vector(game.snake[i][0], game.snake[i][1]) * int(tile_size), 0.8 * int(tile_size))

    fill(food_color)
    square(Vector(game.fruit[0], game.fruit[1]) * int(tile_size), 0.8 * int(tile_size))


def setup():
    size(tile_size * width, tile_size * height)
    title("p5 Snake")
    draw_game(game)

def draw():
    global game
    global proxy_score
    draw_game(game)
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

    

if __name__ == '__main__':
    run(frame_rate=15)