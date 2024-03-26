from snakes.snake_base import SnakeEnv
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
import pygame

# rewards:
# 10 for eating fruit 
# -10 for dying
# 0 for winning (you still get 10 for fruit)
# -0.01 for nothing

class Snake4(SnakeEnv):

    reward_unit = 0.1
    max_reward_abs = 100 * reward_unit

    def __init__(self, render_mode='train', width=4, height=4, snake_length=4, random_seed=None) -> None:
        super().__init__(render_mode=render_mode, width=width, height=height, snake_length=snake_length, random_seed=random_seed)

        self.steps = self.last_meal = 0

        # env variables

        self.reward_range = (-self.max_reward_abs, self.max_reward_abs)

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
    
    def _hunger(self):
        """Steps since last meal."""
        return self.steps - self.last_meal

    def _stamina(self):
        """How long snake can live without food."""
        world_area = self.width * self.height
        stamina = world_area + len(self.snake) + 1
        stamina = min(world_area * 2, stamina)
        return stamina

    def _is_starved_to_death(self):
        """Return true if the snake hasn't eaten for too long."""
        return self._hunger() > self._stamina()
    
    def step(self, action):
        # step(action) -> ObseravtionType, Float, Bool, Bool

        self.steps += 1

        if self.steps > (self.width*self.height) ** 2:
            return self._get_state(), -100, False, True, {'snake': self.snake}
        
        new = self.snake[-1] - self.action_map[action]

        won = None # none means nothing, true means won, false means lost

        reward = 0
        win_reward = 50 * self.reward_unit
        loss_reward = -win_reward

        collision = self._collision(self.snake, new, True)
        starved = self._is_starved_to_death()

        if collision or starved:
            won = False
            reward += loss_reward

        else:
        
            self.snake = np.append(self.snake, [new], axis=0)

            if np.array_equal(new, self.fruit):
                if len(self.snake) == self.width * self.height:
                    reward += win_reward
                    won = True
                else:
                    self.last_meal = self.steps
                    self.fruit = self._generate_fruit()
                    reward += 10 * self.reward_unit
            else:
                self.snake = np.delete(self.snake, 0, axis=0)

        reward = np.clip(reward, *self.reward_range)
        
        return self._get_state(), reward, not won == None, False, {'won': won}

    def reset(self, seed=None) -> None:
        if seed:
            self.random_seed = seed
        self.over = False
        self.quit = False
        self.steps = self.last_meal = 0

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