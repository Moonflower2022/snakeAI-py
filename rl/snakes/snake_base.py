from gymnasium import Env
import numpy as np
import random
import pygame

class SnakeEnv(Env):
    render_modes = ['human', 'train']
    display_width = 600
    display_height = 600

    body_color = (161, 181, 108)
    head_color = (102, 204, 0)
    fruit_color = (171, 70, 66)
    square_size_factor=0.8
    FPS=5

    colors = {
        1: pygame.Color(*body_color),
        2: pygame.Color(*head_color),
        3: pygame.Color(*fruit_color)
    }

    def __init__(self, render_mode='train', width=4, height=4, snake_length=4, random_seed=None) -> None:
        assert snake_length > 0
        assert width > 0
        assert height > 0
        assert width * height > snake_length
        assert render_mode in self.render_modes

        self.random_seed = random_seed

        if render_mode == 'human':
            pygame.init()
            pygame.display.set_caption('Snake Display')
            self.screen = None
            self.clock = pygame.time.Clock()
            self.quit = False
        
        self.render_mode = render_mode

        # game variables

        self.width = width
        self.height = height
        self.snake_length = snake_length
        self.snake = None
        self.fruit = None
        self.over = False

    def _collision(self, snake, new, in_game):
        actual_snake = np.delete(snake, 0, axis=0) if in_game else snake

        if np.any([np.array_equal(new, segment) for segment in actual_snake]) or 0 > new[0] or new[0] >= self.width or 0 > new[1] or new[1] >= self.height:
            return True
        return False

    def _generate_snake(self, width, height, snake_length): 
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

            viable_options = [option for option in options if not self._collision(snake, option, False)]

            if len(viable_options) == 0:
                return self.generate_snake(snake_length, width, height)
            if self.random_seed: 
                random.seed(self.random_seed)
            snake.append(random.choice(viable_options))

        return np.array(snake)
    
    def _generate_fruit(self):        
        empty_spaces = [cell for cell in np.argwhere(np.zeros([self.width, self.height]) == 0) if not any(np.array_equal(cell, arr) for arr in self.snake)]
               
        if not empty_spaces:
            return False
        if self.random_seed: 
            random.seed(self.random_seed)
        return random.choice(empty_spaces)
    
    def _get_state(self):
        '''
        Parameters:
            snake (np.array, shape: (l, 2)): 2d positions (x, y) where x, y in Z, x in [0, width-1], y in [0, height-1]
            fruit (np.array, shape: (2, )): fruit's position as a position (x, y) where x, y in Z, x in [0, width-1], y in [0, height-1]

        Returns:
            (np.array, shape: (w, h)): values in Z and in [0, 3], where 0: empty space, 1: snake body, 2: snake head, 3: fruit
        '''

        grid = np.zeros((self.width, self.height), dtype=np.int32)

        for i in range(len(self.snake)):
            grid[self.snake[i][1]][self.snake[i][0]] = 2 if (i == len(self.snake) - 1) else 1

        grid[self.fruit[1]][self.fruit[0]] = 3

        return grid