from p5 import *
import numpy as np

def hash_2d(array_2d):
    return tuple(tuple(inner_array) for inner_array in array_2d)

def hash_state(snake, fruit):
    return (hash_2d(snake), tuple(fruit))

class Game:
    def __init__(self, move_function, board_length, snake, q_table, model):
        # snake starts left from start_pos so start_pos[0] cant be less than snake_length
        # move_function should take the game object and return a velocity in np.array ex: np.array([1, 0])
        self.q_table = q_table
        self.model = model
        self.board_length = board_length
        self.snake = snake
        self.fruit = self.add_fruit()
        self.score = 0
        self.game_over = False
        self.velocity = np.array([1, 0])
        self.velocity = move_function(self) # Initial velocity
        self.move_function = move_function

    def move(self):
        new_position = self.snake[-1] + self.velocity
        if np.any([np.array_equal(new_position, segment) for segment in np.delete(self.snake, 0, axis=0)]) or new_position[0] < 0 or new_position[0] >= self.board_length or new_position[1] < 0 or new_position[1] >= self.board_length:
            self.game_over = True
            return
        else:
            self.snake = np.append(self.snake, [new_position], axis=0)
            if np.array_equal(new_position, self.fruit):
                self.score += 1
                self.fruit = self.add_fruit()
            self.snake = np.delete(self.snake, 0, axis=0)

    def update(self):
        if self.game_over:
            return False
        self.move()
        self.velocity = self.move_function(self)
        return True

    def add_fruit(self):
        empty_spaces = [cell for cell in np.argwhere(np.zeros([self.board_length] * 2) == 0) if not any(np.array_equal(cell, arr) for arr in self.snake)]
        if not empty_spaces:
            return False
        return empty_spaces[np.random.choice(len(empty_spaces))]
