from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random

# rewards:
# 1 for eating fruit 
# -1 for dying
# 100 for winning

class SnakeEnv(Env):
    def __init__(self, width, height, snake_length, random_seed=None) -> None:
        self.random_seed = random_seed

        # game variables

        self.width = width
        self.height = height
        self.snake_length = snake_length
        self.snake = self.generate_snake(width, height, snake_length) # last index is head
        self.fruit = self.generate_fruit()

        # env variables

        self.action_space = Discrete(3) 
        # turn counter-clockwise, do nothing, turn clockwise
        # maybe it should be 4 for up down left right
        self.observation_space = Box(low=0, high=3, shape=(width, height), dtype=np.int32)
        # 0: empty space, 1: snake body, 2: snake head, 3: fruit

    def generate_snake(self, width, height, snake_length): 
        """
        Generate a random snake of a given length on a grid.

        Parameters:
            snake_length (int): The length of the snake.
            width (int): The width of the grid.
            height (int): The height of the grid.

        Returns:
            (list): A list of coordinates representing the snake.
        """
        # Initialize the snake with a random starting position
        if self.random_seed: 
            random.seed(self.random_seed)
        snake = [(random.randint(0, width - 1), random.randint(0, height - 1))]
        
        # Generate the rest of the snake's body
        while len(snake) < snake_length:
            last_position = snake[-1]

            options = [
                np.array((last_position[0], last_position[1] - 1)), 
                np.array((last_position[0], last_position[1] + 1)), 
                np.array((last_position[0] - 1, last_position[1])), 
                np.array((last_position[0] + 1, last_position[1]))
            ]

            viable_options = [option for option in options if not any(np.array_equal(snake_segment, option) for snake_segment in snake) and 0 <= option[0] < self.width and 0 <= option[1] < self.height]

            if len(viable_options) == 0:
                return self.generate_snake(snake_length, width, height)
            if self.random_seed: 
                random.seed(self.random_seed)
            snake.append(random.choice(viable_options))

        return np.array(snake)
    
    def generate_fruit(self):
        empty_spaces = [cell for cell in np.argwhere(np.zeros([self.width, self.height]) == 0) if not any(np.array_equal(cell, arr) for arr in self.snake)]
        if not empty_spaces:
            return False
        if self.random_seed: 
            random.seed(self.random_seed)
        return random.choice(empty_spaces)
    
    def state_to_grid(self):
        '''
        Parameters:
            snake (np.array, shape: (l, 2)): 2d positions (x, y) where x, y in Z, x in [0, width-1], y in [0, height-1]
            fruit (np.array, shape: (2, )): fruit's position as a position (x, y) where x, y in Z, x in [0, width-1], y in [0, height-1]

        Returns:
            (np.array, shape: (w, h)): values in Z and in [0, 3], where 0: empty space, 1: snake body, 2: snake head, 3: fruit
        '''

        grid = np.zeros((self.width, self.height), dtype=np.int32)

        for i in range(len(self.snake)):
            grid[self.snake[i][1]][self.snake[i][0]] = 2 if i == 0 else 1

        grid[self.fruit[1]][self.fruit[0]] = 3

        return grid
    
    def collision(self, new):
        if np.any([np.array_equal(new, segment) for segment in np.delete(self.snake, 0, axis=0)]) or 0 > new[0] >= self.width or 0 > new[1] >= self.height:
            return True
        return False        

    def step(self, action):
        # step(action) -> ObseravtionType, Float, Bool, Bool

        new = self.snake[-1] + action

        if self.collision(new):
            return self.state_to_grid(), -1, True, False
        
        self.snake = np.append(self.snake, [new], axis=0)

        if np.array_equal(new, self.fruit):
            if len(self.snake) == self.width * self.height:
                return self.state_to_grid(), 100, True, False
            self.generate_fruit()
            return self.state_to_grid(), 1, False, False
        
        self.snake = np.delete(self.snake, 0, axis=0)
        return self.state_to_grid(), 0, False, False

    def reset(self) -> None:
        self.snake = self.generate_snake(self.width, self.height, self.snake_length) # last index is head
        self.fruit = self.generate_fruit()
        return self.state_to_grid()
    def render(self):
        pass
    def close(self):
        pass

env = SnakeEnv(width=4, height=4, snake_length=4, random_seed=100)