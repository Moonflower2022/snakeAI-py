from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
import pygame

# rewards:
# 1 for eating fruit 
# -1 for dying
# 100 for winning

class SnakeEnv(Env):
    def __init__(self, render_mode='rgb_array', width=4, height=4, snake_length=4, display_width=400, display_height=400, body_color=(161, 181, 108), head_color=(102, 204, 0), fruit_color=(171, 70, 66), square_size_factor=0.8, FPS=30, random_seed=None) -> None:
        assert display_width/width == display_height/height
        assert snake_length > 0
        assert width > 0
        assert height > 0
        assert width * height > snake_length

        self.display_width = display_width
        self.display_height = display_height
        self.random_seed = random_seed
        if render_mode == 'human':
            pygame.init()
            pygame.display.set_caption('Snake Display')
            self.screen = pygame.display.set_mode((display_width, display_height))
            self.clock = pygame.time.Clock()
            self.running = True
        self.colors = {
            1: pygame.Color(*body_color),
            2: pygame.Color(*head_color),
            3: pygame.Color(*fruit_color)
        }
        self.square_size_factor = square_size_factor
        self.render_mode = render_mode
        self.FPS = FPS

        # game variables

        self.width = width
        self.height = height
        self.snake_length = snake_length
        self.snake = self.generate_snake(width, height, snake_length) # last index is head
        self.fruit = self.generate_fruit()

        # env variables

        self.action_space = Discrete(4) 
        # up down left right
        self.action_map = {
            0: np.array([0, -1]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0]),
        }
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

            viable_options = [option for option in options if not self.collision(snake, option)]

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
            grid[self.snake[i][1]][self.snake[i][0]] = 2 if i == len(self.snake) - 1 else 1

        grid[self.fruit[1]][self.fruit[0]] = 3

        return grid
    
    def collision(self, snake, new):
        if np.any([np.array_equal(new, segment) for segment in np.delete(snake, 0, axis=0)]) or 0 > new[0] or new[0] >= self.width or 0 > new[1] or new[1] >= self.height:
            return True
        return False    
    
    def step(self, action):
        # step(action) -> ObseravtionType, Float, Bool, Bool
        new = self.snake[-1] + self.action_map[action]

        if self.collision(self.snake, new):
            return self.state_to_grid(), -1, True, False, {'snake': self.snake}
        
        self.snake = np.append(self.snake, [new], axis=0)

        if np.array_equal(new, self.fruit):
            if len(self.snake) == self.width * self.height:
                return self.state_to_grid(), 100, True, False, {'snake': self.snake}
            self.fruit = self.generate_fruit()
            return self.state_to_grid(), 1, False, False, {'snake': self.snake}

        self.snake = np.delete(self.snake, 0, axis=0)
        
        return self.state_to_grid(), 0, False, False, {'snake': self.snake}

    def reset(self, seed=None) -> None:
        if seed:
            self.random_seed = seed
        self.snake = self.generate_snake(self.width, self.height, self.snake_length) # last index is head
        self.fruit = self.generate_fruit()
        if self.render_mode == 'human':
            pygame.init()
            pygame.display.set_caption('Snake Display')
            self.display = pygame.display.set_mode((self.display_width, self.display_height))
            self.clock = pygame.time.Clock()
            self.render()
        return self.state_to_grid(), {'snake': self.snake}
    
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.close()
        self.screen.fill("black")

        grid = self.state_to_grid()
        factor_x = self.display_width/self.width
        factor_y = self.display_height/self.height
        assert factor_x == factor_y
        for y in range(len(grid)):
            for x in range(len(grid[y])):
                if grid[y][x] != 0:
                    pygame.draw.rect(self.screen, self.colors[grid[y][x]], pygame.Rect((x + 0.5 * (1-self.square_size_factor)) * factor_x, (y + 0.5 * (1-self.square_size_factor)) * factor_y, factor_x*self.square_size_factor, factor_y*self.square_size_factor))
        pygame.display.flip()
        pygame.display.update()
        self.clock.tick(self.FPS)
    def close(self):
        pygame.quit()



test_env = SnakeEnv(render_mode='human', display_width=400, display_height=400, width=4, height=4, snake_length=4, FPS=5)

obs, rewards, terminated, truncated, info = test_env.step(3)