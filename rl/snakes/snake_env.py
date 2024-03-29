from snakes.snake_base import Snake
from gymnasium.spaces import Discrete, Box
import numpy as np
import pygame

# rewards dict needs 'lose', 'win', 'fruit', and 'nothing'


class SnakeEnv(Snake):
    reward_unit = 0.1

    def __init__(self, render_mode='train', width=4, height=4, snake_length=4, rewards={'win': 1, 'fruit': 1, 'lose': -1, 'nothing': -0.0001}, 
                starve=False, no_backwards=True, step_limit=None, random_seed=None) -> None:
        super().__init__(render_mode=render_mode, width=width, height=height,
                         snake_length=snake_length, random_seed=random_seed)

        self.steps = self.last_meal = 0
        self.starve = starve
        self.rewards = rewards
        self.no_backwards = no_backwards
        self.step_limit = step_limit

        # env variables

        self.action_space = Discrete(4)
        # up down left right
        self.action_map = {
            0: np.array([0, -1]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0]),
        }
        self.observation_space = Box(
            low=0, high=3, shape=(width, height), dtype=np.int32)
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

        if self.no_backwards and (not len(self.snake) == 1 and np.array_equal(self.snake[-2] - self.snake[-1], self.action_map[action])):
            new = self.snake[-1] - self.action_map[action]
        else:
            new = self.snake[-1] + self.action_map[action]

        won = None  # none means nothing, true means won, false means lost

        reward = 0
        win_reward = self.rewards['win'] * self.reward_unit
        loss_reward = self.rewards['lose'] * self.reward_unit

        if self.step_limit != None and self.steps > self.step_limit:
            won = False
            reward = loss_reward

        if self._collision(self.snake, new, True) or (self.starve and self._is_starved_to_death()):
            won = False
            reward = loss_reward
        else:
            self.snake = np.append(self.snake, [new], axis=0)

            if np.array_equal(new, self.fruit):
                if len(self.snake) == self.width * self.height:
                    reward = win_reward
                    won = True
                else:
                    self.last_meal = self.steps
                    self.fruit = self._generate_fruit()
                    reward = self.rewards['fruit'] * self.reward_unit
            else:
                self.snake = np.delete(self.snake, 0, axis=0)
                reward = self.rewards['nothing'] * self.reward_unit

        return self._get_state(), reward, not won == None, False, {'won': won, 'snake': self.snake}

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

        grid = self._get_state()
        factor_x = self.display_width/self.width
        factor_y = self.display_height/self.height
        assert factor_x == factor_y

        for y in range(len(grid)):
            for x in range(len(grid[y])):
                if grid[y][x] != 0:
                    pygame.draw.rect(self.screen, self.colors[grid[y][x]], pygame.Rect((x + 0.5 * (1-self.square_size_factor)) * factor_x, (y + 0.5 * (
                        1-self.square_size_factor)) * factor_y, factor_x*self.square_size_factor, factor_y*self.square_size_factor))

        pygame.display.flip()
        self.clock.tick(self.FPS)

    def close(self):
        pygame.quit()
