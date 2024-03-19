from snakes.snake_base import SnakeEnv
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
import pygame

# rewards:
# 1 for eating fruit, -1 for dying, 100 for winning, 0 for nothing


class Snake4(SnakeEnv):
    def __init__(self, render_mode='train', width=4, height=4, snake_length=4, random_seed=None) -> None:
        super().__init__(render_mode=render_mode, width=width, height=height, snake_length=snake_length, random_seed=random_seed)

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
    
    def step(self, action):
        # step(action) -> ObseravtionType, Float, Bool, Bool
        new = self.snake[-1] + self.action_map[action]

        if self._collision(self.snake, new, True):
            return self._get_state(), -1, True, False, {'snake': self.snake}
        
        self.snake = np.append(self.snake, [new], axis=0)

        if np.array_equal(new, self.fruit):
            if len(self.snake) == self.width * self.height:
                return self._get_state(), 100, True, False, {'snake': self.snake}
            self.fruit = self._generate_fruit()
            return self._get_state(), 1, False, False, {'snake': self.snake}

        self.snake = np.delete(self.snake, 0, axis=0)
        
        return self._get_state(), 0, False, False, {'snake': self.snake}

    def reset(self, seed=None) -> None:
        if seed:
            self.random_seed = seed
        self.over = False
        self.quit = False
        self.steps = self.num_food = self.last_meal = 0

        self.snake = self._generate_snake(self.width, self.height, self.snake_length) # last index is head
        self.fruit = self._generate_fruit()

        if self.render_mode == 'human':
            self.clock = pygame.time.Clock()
            self.render()

        return self._get_state(), {'snake': self.snake}
    
    def render(self):
        if not self.screen:
            pygame.display.set_caption('Snake RL')
            self.screen = pygame.display.set_mode((self.display_width, self.display_height))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                self.quit = True
                return 

        self.screen.fill("black")

        grid = self._get_state()
        factor_x = self.display_width/self.width
        factor_y = self.display_height/self.height
        assert factor_x == factor_y

        for y in range(len(grid)):
            for x in range(len(grid[y])):
                if grid[y][x] != 0:
                    pygame.draw.rect(self.screen, self.colors[grid[y][x]], pygame.Rect((x + 0.5 * (1-self.square_size_factor)) * factor_x, (y + 0.5 * (1-self.square_size_factor)) * factor_y, factor_x*self.square_size_factor, factor_y*self.square_size_factor))
        
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def close(self):
        pygame.quit()