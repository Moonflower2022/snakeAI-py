from p5 import *
import numpy as np
from game import Game
from cycle_finder import hamiltonian_cycle_from_point, grid_generator
from cycle_finder2 import generate_hamiltonian_circuit

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

width = 10
height = 10
tile_size = 50
background_color = Color(24)
gameover_overlay_color = Color(88, 127)

snake_color = Color(161, 181, 108)
head_color = Color(102, 204, 0)
food_color = Color(171, 70, 66)

square_size_factor = 0.8

snake = np.array([np.array([np.random.randint(0, width), np.random.randint(0, height)])])

#grid = grid_generator(width, height)
#cycle = hamiltonian_cycle_from_point(grid, snake[0][1], snake[0][0])

q = 1
n, cycle = generate_hamiltonian_circuit(q, width - 1, height - 1)

print(cycle)

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
        square(Vector(game.snake[i][0], game.snake[i][1]) * int(tile_size) + Vector((1 - square_size_factor) * tile_size / 2, (1 - square_size_factor) * tile_size / 2), square_size_factor * int(tile_size))

    fill(food_color)
    square(Vector(game.fruit[0], game.fruit[1]) * int(tile_size) + Vector((1 - square_size_factor) * tile_size / 2, (1 - square_size_factor) * tile_size / 2), square_size_factor * int(tile_size))


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
    run(frame_rate=60)