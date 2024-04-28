from envs.snake_base import Snake
from gymnasium.spaces import Discrete, Box
import numpy as np
import pygame
import math

# rewards dict needs 'lose', 'win', 'fruit', and 'nothing'

class SnakeEnvCustomRewards(Snake):
    reward_unit = 1

    def __init__(self, render_mode='train', width=4, height=4, snake_length=4, no_backwards=True, fruit_limit=False, random_seed=None) -> None:
        super().__init__(render_mode=render_mode, width=width, height=height,
                         snake_length=snake_length, random_seed=random_seed)

        self.steps = self.last_meal = 0
        self.no_backwards = no_backwards
        self.fruit_limit = fruit_limit

        # env variables

        self.action_space = Discrete(4)
        # up down left right
        self.action_map = {
            0: np.array([0, -1]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0]),
        }
        # self.observation_space = Box(low=0, high=3, shape=(width, height), dtype=np.int32)
        # 0: empty space, 1: snake body, 2: snake head, 3: fruit

        self.observation_space = Box(low=-1, high=1, shape=(width, height), dtype=np.float32)
        # empty space: 0, snake body: spread from 0.5 to 1, snake head: 1, fruit: -1

    def _hunger(self):
        """Steps since last meal."""
        return self.steps - self.last_meal
    
    def _get_state(self):
        '''
        Parameters:
            snake (np.array, shape: (l, 2)): 2d positions (x, y) where x, y in Z, x in [0, width-1], y in [0, height-1]
            fruit (np.array, shape: (2, )): fruit's position as a position (x, y) where x, y in Z, x in [0, width-1], y in [0, height-1]

        Returns:
            (np.array, shape: (w, h)): values in Z and in [0, 3], where 0: empty space, 1: snake body, 2: snake head, 3: fruit
        '''

        observation = np.zeros((self.width, self.height), dtype=np.float32)

        observation[tuple(np.transpose(self.snake))] = np.linspace(0.5, 1, len(self.snake), dtype=np.float32)
        observation[tuple(self.fruit)] = -1

        return observation

    def step(self, action):
        # step(action) -> ObseravtionType, Float, Bool, Bool

        self.steps += 1

        if self.no_backwards and not len(self.snake) == 1 and np.array_equal(self.snake[-2] - self.snake[-1], self.action_map[action]):
            new = self.snake[-1] - self.action_map[action]
        else:
            new = self.snake[-1] + self.action_map[action]

        won = False
        terminated = False 
        truncated = False

        reward = 0

        if self._collision(self.snake, new, True) or (self.fruit_limit != None and self._hunger() > self.fruit_limit):
            terminated = True
            reward = len(self.snake) - self.width * self.height
        else:
            self.snake = np.append(self.snake, [new], axis=0)

            if np.array_equal(new, self.fruit):
                if len(self.snake) == self.width * self.height:
                    won = True
                    terminated = True
                else:
                    self.last_meal = self.steps
                    self.fruit = self._generate_fruit()
                reward = math.exp(((self.width * self.height) - (self.steps - self.last_meal)) / (self.width * self.height))
            else:
                self.snake = np.delete(self.snake, 0, axis=0)
                reward = -0.001

        return self._get_state(), self.reward_unit * reward, terminated, truncated, {'won': won, 'snake': self.snake}

    def reset(self, seed=None) -> None:
        if seed:
            self.random_seed = seed
        self.over = False
        self.quit = False
        self.steps = self.last_meal = 0

        self.snake = self._generate_snake(
            self.width, self.height, self.snake_length)  # last index is head
        self.fruit = self._generate_fruit()

        if self.render_mode == 'human':
            self.clock = pygame.time.Clock()
            self.render()

        return self._get_state(), {'snake': self.snake}
    
    def _draw_square(self, surface, color, x, y):

        pygame.draw.rect(surface, color, pygame.Rect((x + 0.5 * (1-self.square_size_factor)) * self.factor_x, (y + 0.5 * (
                1-self.square_size_factor)) * self.factor_y, self.factor_x*self.square_size_factor, self.factor_y*self.square_size_factor))

    def render(self):

        if not self.screen:
            pygame.display.set_caption('Snake RL')
            self.screen = pygame.display.set_mode(
                (self.display_width, self.display_height))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                self.quit = True
                return

        self.screen.fill("black")

        self.factor_x = self.display_width/self.width
        self.factor_y = self.display_height/self.height
        assert self.factor_x == self.factor_y

        for i, position in enumerate(self.snake):
            x = position[0]
            y = position[1]

            segment_type = "head" if i == len(self.snake) - 1 else "body"

            self._draw_square (self.screen, self.colors[segment_type], x, y)

        self._draw_square (self.screen, self.colors["fruit"], self.fruit[0], self.fruit[1])

        pygame.display.flip()
        self.clock.tick(self.FPS)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = SnakeEnvCustomRewards()
    env.reset()
    print(env._get_state())