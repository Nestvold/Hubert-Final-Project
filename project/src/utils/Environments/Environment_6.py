# Utils module
from .extra import GridValues

# Reward functions
# from .extra import reward_func_base as reward_function
# from .extra import reward_func_height_focus as reward_function
from .extra import reward_func_height_and_exploration as reward_function
# from .extra import reward_func_no_negative as reward_function
# from .extra import reward_to_good_to_be_true as reward_function


# Other modules
from matplotlib.pyplot import title, savefig, figure, close, show
from numpy import ndarray, zeros, float32
from random import random, choice, randint
from gym.spaces import Discrete, Box
from imageio.v2 import imread
from imageio import mimsave
from seaborn import heatmap
from typing import Optional
from tqdm import tqdm
from abc import ABC
from gym import Env
from copy import deepcopy
import os


class Environment_6(Env, ABC):
    def __init__(self, name: str, grid: ndarray, MM: list = None, fans: list = None, pMM: float = 1.0, start_coords: tuple = (46, 1),
                 project_path: str = 'project/src/level_6.py'):
        self.name = name
        self.project_path = project_path

        self.MM = MM
        self.pMM = pMM
        self.fans = fans

        # Gym
        self.action_space = Discrete(3)  # 0 = go left, 1 = jump, 2 = go right

        # y, x, 9 x 9 grid -> 0: nothing, 1: Air, 2: Solid, 3: Semisolid, 4: MM, 5: fan
        self.observation_space = Box(low=-1, high=6, shape=(9, 9), dtype=float32)

        # State representation
        self.start_coords = start_coords
        self.y, self.x = self.start_coords
        self.best_y = self.y
        self.energy = 150
        self.energy_cons = 0
        self.surroundings = zeros(shape=9 * 9)

        # Action mapping
        self.action_mapping = {0: (0, -1), 1: (-1, 0), 2: (0, 1)}

        # Grid
        self.grid = grid
        self.solids = set([2, 3])
        self.grid_y = GridValues(grid)
        self.grid_x = GridValues(grid[0])

        # Start state
        self.scan_surroundings()
        self.peak = self.best_y / (self.grid.shape[0] - 3)
        self.prev_states = set([hash(str(list(self.surroundings)))])

    def __str__(self):
        return f'Name: {self.name}, project path: {self.project_path}, MM: {self.MM}, fans: {self.fans}, start pos: {self.start_coords} == {self.y, self.x}, '

    def step(self, action: int) -> tuple[ndarray, float, bool]:

        if action not in self.action_mapping:
            raise ValueError(f'Invalid action: {action}.')

        energy = 1.0
        old_pos = self.y, self.x
        dy, dx = self.action_mapping[action]

        trajectory = []

        if action in [0, 2]:
            if self.can_go(self.y, self.x + dx):
                self.x += dx
                trajectory.append((self.y, self.x, deepcopy(self.MM), deepcopy(self.fans), energy))

                if not self.on_solid_grounds():
                    self.y += 1
                    trajectory.append((self.y, self.x, deepcopy(self.MM), deepcopy(self.fans), energy))

                    if self.can_go(self.y, self.x + dx):
                        self.x += dx
                        trajectory.append((self.y, self.x, deepcopy(self.MM), deepcopy(self.fans), energy))

                        if not self.on_solid_grounds():
                            self.y += 1
                            trajectory.append((self.y, self.x, deepcopy(self.MM), deepcopy(self.fans), energy))
            else:
                if not self.on_solid_grounds():
                    self.y += 1
                    trajectory.append((self.y, self.x, deepcopy(self.MM), deepcopy(self.fans), energy))

                    if not self.on_solid_grounds():
                        self.y += 1
                    trajectory.append((self.y, self.x, deepcopy(self.MM), deepcopy(self.fans), energy))
        else:
            energy += 4
            if self.on_solid_grounds():
                if self.can_go(self.y + dy, self.x):
                    self.y += dy
                    trajectory.append((self.y, self.x, deepcopy(self.MM), deepcopy(self.fans), energy))

                    if self.can_go(self.y + dy, self.x):
                        self.y += dy
                        trajectory.append((self.y, self.x, deepcopy(self.MM), deepcopy(self.fans), energy))

            elif random() < 1 / 3 and self.can_go(self.y + dy, self.x):
                energy -= 2
                self.y += dy
                trajectory.append((self.y, self.x, deepcopy(self.MM), deepcopy(self.fans), energy))

            else:
                self.y += 1
                trajectory.append((self.y, self.x, deepcopy(self.MM), deepcopy(self.fans), energy))

                if not self.on_solid_grounds():
                    self.y += 1
                    trajectory.append((self.y, self.x, deepcopy(self.MM), deepcopy(self.fans), energy))

        self.energy -= energy
        self.energy_cons += energy

        # Move enemies
        self.update_enemies(old_pos)

        trajectory.append((self.y, self.x, deepcopy(self.MM), deepcopy(self.fans), energy))
        reward, done = self.reward_function(old_pos)

        self.scan_surroundings(action)

        return self.surroundings, reward, done, {'energy': energy, 'energy_cons': self.energy_cons, 'peak': self.peak, 'trajectory': trajectory}

    def reward_function(self, prev_pos):
        reward, done = 0.0, False

        # If encountering MM
        if self.busted():
            return -5.0, True

        # If reached the ceiling
        if self.in_end_state():
            return 10.0, True

        # Penalize getting seen by fans
        self.energy -= self.seen()

        if self.energy < 0:
            return -5.0, True

        # Encourage height
        if self.y < self.best_y:
            change = self.best_y - self.y
            self.energy += change * 15
            self.best_y = self.y
            self.peak = self.best_y / (self.grid.shape[0] - 2) if self.best_y > 1 else 0
            reward += 1.0
        else:
            reward -= 0.1

        # Encourage exploration
        if (surroundings := hash(str(list(self.surroundings.flatten())))) not in self.prev_states:
            self.prev_states.add(surroundings)
            reward += 0.1
        else:
            reward -= 0.1

        if self.x == prev_pos[1] and self.y == prev_pos[0]:
            reward -= 0.1

        return reward, done

    def busted(self):
        return [self.y, self.x] in self.MM and random() < self.pMM

    def seen(self):
        cost = 0
        for fan in self.fans:
            if [self.y, self.x] == fan:
                cost = randint(20, 100)
        self.energy_cons += cost
        return cost

    def can_move_enemy(self, entity, move):
        return self.grid[entity[0], entity[1] + move] != 2 and self.grid[entity[0] + 1, entity[1] + move] != 1

    def update_enemies(self, pre_pos):
        # Update MM randomly
        for i in range(len(self.MM)):
            move = choice([-1, 1])
            if self.can_move_enemy(self.MM[i], move):
                self.MM[i] = [self.MM[i][0], self.MM[i][1] + move]

        # Update fans based on fan vision
        for i in range(len(self.fans)):
            if self.fans[i][0] == pre_pos[0]:
                move = 1 if self.fans[i][1] < pre_pos[1] else -1
            else:
                move = choice([-1, 1])

            if self.can_move_enemy(self.fans[i], move):
                self.fans[i] = [self.fans[i][0], self.fans[i][1] + move]

    def scan_surroundings(self, action: int = None) -> None:
        area = zeros(shape=(9, 9))

        for y, y_v in enumerate(range(self.y - 4, self.y + 5)):
            for x, x_v in enumerate(range(self.x - 4, self.x + 5)):
                if self.on_grid(y_v, x_v):
                    if [y_v, x_v] in self.MM:
                        area[y, x] = 4
                    elif [y_v, x_v] in self.fans:
                        area[y, x] = 5
                    else:
                        area[y, x] = self.grid[y_v, x_v]

        action_tune = {0: 0.1, 1: 0.2, 2: 0.3}.get(action, 0.0)
        area[4, 4] += action_tune

        self.surroundings = area

    def can_go(self, y: int, x: int) -> bool:
        return self.grid[y, x] != 2.0

    def on_grid(self, y: int, x: int) -> bool:
        return self.grid_y.min <= y <= self.grid_y.max and self.grid_x.min <= x <= self.grid_x.max

    def on_solid_grounds(self):
        return self.grid[self.y + 1, self.x] in self.solids

    def in_end_state(self) -> bool:
        return self.y == 1

    def reset(self, seed: Optional[int] = None) -> tuple[ndarray, dict]:
        self.y, self.x = self.start_coords
        self.scan_surroundings()
        self.prev_states = set([hash(str(list(self.surroundings)))])
        self.best_y = self.y
        self.peak = self.best_y / (self.grid.shape[0] - 2)
        self.energy_cons = 0
        self.energy = 150

        return self.surroundings

    def plot(self, agent: dict = None, color_bar: bool = True, save: bool = False):
        """
        Plot a heatmap of the environment with optional agent routes.
        :param agent: A dictionary containing the agent names as keys and their corresponding routes as values.
        :param color_bar: A boolean indicating whether to display a color bar for the heatmap.
        :param save: A boolean indicating whether to save the plots to a file.
        :returns: None
        """
        if agent is not None:  # 2a7fff
            #              Green      White      Black      Blue       Pink       Orange
            color_map = ['#00d400', '#FFFFFF', '#000000', '#2a7fff', '#f77979', '#FFA500']
        else:
            color_map = ['#00d400', '#FFFFFF', '#000000', '#2a7fff', '#f77979']

        figure(figsize=(10, 10))
        map = heatmap(data=self.grid, cmap=color_map, cbar=color_bar)

        for _, spine in map.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)

        title(self.name.capitalize(), size=30)

        if save:
            savefig(f'{self.project_path}/images/{self.name}.png')
        show()