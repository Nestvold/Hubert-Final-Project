# Utils module
from .extra import GridValues

# Reward functions
# from .extra import reward_func_base as reward_function
# from .extra import reward_func_height_focus as reward_function
from .extra import reward_func_height_and_exploration as reward_function
# from .extra import reward_func_no_negative as reward_function
# from .extra import reward_to_good_to_be_true as reward_function


# Other modules
from matplotlib.pyplot import title, savefig, figure, close
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
import os


class Environment_6(Env, ABC):
    def __init__(self, name: str, grid: ndarray, MM: dict, fans: dict, pMM: float = 1.0, start_coords: tuple = (46, 1),
                 project_path: str = 'project/src/level_6.py'):
        self.name = name
        self.project_path = project_path

        self.MM = MM
        self.pMM = pMM
        self.fans = fans

        # Gym
        self.action_space = Discrete(3)  # 0 = go left, 1 = jump, 2 = go right

        # y, x, 9 x 9 grid -> 0: nothing, 1: Air, 2: Solid, 3: Semisolid, 4: MM, 5: fan
        self.observation_space = Box(low=-1, high=5, shape=(9, 9), dtype=float32)

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

    def step(self, action: int, track: bool = False, t: int = None, trajectory: list = None) -> tuple[
        ndarray, float, bool]:

        if action not in self.action_mapping:
            raise ValueError(f'Invalid action: {action}.')

        energy = 1.0

        old_pos = self.y, self.x

        dy, dx = self.action_mapping[action]

        if action in [0, 2]:
            if self.can_go(self.y, self.x + dx):
                self.x += dx
                if track: trajectory.append((self.y, self.x, self.MM, self.fans, energy, t))

                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, self.MM, self.fans, energy, t))

                    if self.can_go(self.y, self.x + dx):
                        self.x += dx
                        if track: trajectory.append((self.y, self.x, self.MM, self.fans, energy, t))

                        if not self.on_solid_grounds():
                            self.y += 1
                            if track: trajectory.append((self.y, self.x, self.MM, self.fans, energy, t))
            else:
                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, self.MM, self.fans, energy, t))

                    if not self.on_solid_grounds():
                        self.y += 1
                    if track: trajectory.append((self.y, self.x, self.MM, self.fans, energy, t))
        else:
            energy += 4
            if self.on_solid_grounds():
                if self.can_go(self.y + dy, self.x):
                    self.y += dy
                    if track: trajectory.append((self.y, self.x, self.MM, self.fans, energy, t))

                    if self.can_go(self.y + dy, self.x):
                        self.y += dy
                        if track: trajectory.append((self.y, self.x, self.MM, self.fans, energy, t))

            elif random() < 1 / 3 and self.can_go(self.y + dy, self.x):
                energy -= 2
                self.y += dy
                if track: trajectory.append((self.y, self.x, self.MM, self.fans, energy, t))

            else:
                self.y += 1
                if track: trajectory.append((self.y, self.x, self.MM, self.fans, energy, t))

                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, self.MM, self.fans, energy, t))

        self.energy -= energy
        self.energy_cons += energy

        # Move enemies
        self.update_enemies(old_pos)
        if track: trajectory.append((self.y, self.x, self.MM, self.fans, energy, t))

        reward, done = self.reward_function(old_pos)

        self.scan_surroundings()

        return self.surroundings, reward, done, {'energy': energy, 'energy_cons': self.energy_cons, 'peak': self.peak}


    def reward_function(self, prev_pos):
        reward, done = 0, False

        # If encountering MM
        if self.busted():
            return -1.0, True

        # If reached the ceiling
        if self.in_end_state():
            return 1.0, True

        # Penalize getting seen by fans
        self.energy -= self.seen()

        if self.energy < 0:
            return -1.0, True

        # Encourage height
        if self.y < self.best_y:
            change = self.best_y - self.y
            self.energy += change * 15
            self.best_y = self.y
            self.peak = self.best_y / (self.grid.shape[0] - 2) if self.best_y > 1 else 0
            reward += 1

        # Encourage exploration
        if (surroundings := hash(str(list(self.surroundings.flatten())))) not in self.prev_states:
            self.prev_states.add(surroundings)
            reward += 0.1

        if self.x == prev_pos[1] and self.y >= prev_pos[0]:
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
        return self.grid[entity[0], entity[1] + move] != 2 and self.grid[entity[0] - 1, entity[1] + move] != 1

    def update_enemies(self, pre_pos):
        # Update MM randomly
        for MM in self.MM:
            move = choice([-1, 1])
            if self.can_move_enemy(MM, move):
                MM[1] += move

        # Update fans based on fan vision
        for fan in self.fans:
            if fan[0] == pre_pos[0]:
                move = 1 if fan[1] < pre_pos[1] else -1
            else:
                move = choice([-1, 1])

            if self.can_move_enemy(fan, move):
                fan[1] += move

    def scan_surroundings(self) -> None:
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

    def create_gif(self, agent: list[list, list], color_bar: bool = False):
        colors = ['#00d400', '#FFFFFF', '#000000', '#2a7fff', '#f77979', '#FFA500']

        def draw_frame(time_step, t, y, x, e, fans):
            template = self.grid.copy()
            template[y, x] = 5

            for s, x_s in fans.items():
                for j in x_s:
                    template[s, j] = 4

            figure(figsize=(10, 10))
            fig = heatmap(template, cmap=colors, cbar=color_bar)
            title(f'{self.name} \nTime: {t + 1} - energy: {e}', size=30)

            for _, spine in fig.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(1)

            savefig(f'gif/img_{time_step}.png')
            close()

        y_values, x_values, e_values, fans, t_values = zip(*agent)

        for t in tqdm(range(len(x_values)), desc='Creating plots'):
            draw_frame(t, t_values[t], y_values[t], x_values[t], e_values[t], fans[t])

        frames = []

        for t in tqdm(range(len(x_values)), desc='Creating GIP'):
            image = imread(f'gif/img_{t}.png')
            frames.append(image)

        mimsave(uri=f'Levels/gifs/{self.name}.gif',
                ims=frames,
                fps=10
                )

        # Delete plots
        folder_path = "gif"

        # Get all the file names in the folder
        files = os.listdir(folder_path)

        # Loop through the files and delete them
        for file_name in tqdm(files, desc='Deleting files'):
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        print('[GIF CREATED]')
