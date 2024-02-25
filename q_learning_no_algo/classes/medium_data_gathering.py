import numpy as np
import random

def hash_2d(array_2d):
    return tuple(tuple(inner_array) for inner_array in array_2d)

def hash_state(snake, fruit):
    return (hash_2d(snake), tuple(fruit))

class Game:
    def __init__(self, move_function, board_length, starting_snake, starting_velocity, q_table):
        # move_function should take the game object and return a velocity in np.array ex: np.array([1, 0])
        self.move_function = move_function
        self.q_table = q_table
        self.board_length = board_length
        self.snake = starting_snake
        self.fruit = self.add_fruit()
        self.score = 0
        self.won = False
        self.game_over = False
        self.states = 1
        self.one_fruit_index = None
        self.velocity = starting_velocity
        self.velocity = move_function(self) # Initial velocity
        self.state_history = [hash_state(self.snake, self.fruit)]
        self.action_history = [tuple(self.velocity)]
        

    def __str__(self):
        return f"<class Game snake: {self.snake} fruit: {self.fruit} states: {self.states}>"

    def move(self):
        new_position = self.snake[-1] + self.velocity
        if np.any([np.array_equal(new_position, segment) for segment in np.delete(self.snake, 0, axis=0)]) or new_position[0] < 0 or new_position[0] >= self.board_length or new_position[1] < 0 or new_position[1] >= self.board_length:
            self.game_over = True
            return
        else:
            self.snake = np.append(self.snake, [new_position], axis=0)
            if np.array_equal(new_position, self.fruit):
                self.score += 1
                if self.score == 1:
                    self.one_fruit_index = self.states
                else: # self.score == 2
                    self.won = True
                    self.game_over = True
                self.fruit = self.add_fruit()
                return
            self.snake = np.delete(self.snake, 0, axis=0)

    def update(self):
        self.move()
        if self.game_over:
            return False
        
        self.velocity = self.move_function(self)
        self.state_history.append(hash_state(self.snake, self.fruit))
        self.action_history.append(tuple(self.velocity))
        self.states += 1

        if self.states > 100000:
            print(self)
            return False
            #raise Exception("Too many iterations :(")
        
        return True

    def add_fruit(self):
        empty_spaces = [cell for cell in np.argwhere(np.zeros([self.board_length] * 2) == 0) if not any(np.array_equal(cell, arr) for arr in self.snake)]
        if not empty_spaces:
            return False
        return random.choice(empty_spaces) 
