# Utils module
from .extra import GridValues

# Other modules
from matplotlib.pyplot import title, savefig, figure, close, show
from numpy import float32, ndarray, zeros, concatenate
from gym.spaces import Discrete, Box
from imageio.v2 import imread
from collections import deque
from imageio import mimsave
from seaborn import heatmap
from random import random
from tqdm import tqdm
from abc import ABC
from gym import Env
import os


class Environment_5(Env, ABC):
    def __init__(self, name: str, n_actions, grid: ndarray, project_path: str = ''):
        self.name = name
        self.project_path = project_path

        # Gym
        self.action_space = Discrete(3)  # 0 = go left, 1 = jump, 2 = go right

        # y, x, 9 x 9 grid -> 0: Goal/Start, 1: Air, 2: Solid, 3: Semisolid
        self.observation_space = Box(low=-1, high=4, shape=(9, 9), dtype=float32)

        # State representation
        self.y, self.x = 46, 1
        self.surroundings = zeros(shape=9 * 9)

        # Extra
        self.visited_states = set([(self.y, self.x)])
        self.prev_actions = deque(maxlen=n_actions)

        # Action mapping
        self.action_mapping = {0: (0, -1), 1: (-1, 0), 2: (0, 1)}

        # Grid
        self.grid = grid
        self.solids = set([2, 3])
        self.grid_y = GridValues(grid)
        self.grid_x = GridValues(grid[0])

        # Start state
        self.scan_surroundings(-1)

    def step(self, action: int, track: bool = False, t: int = None, trajectory: list = None) -> tuple[ndarray, float, bool]:

        if action not in self.action_mapping:
            raise ValueError(f'Invalid action: {action}.')

        done = False
        reward = -1.0
        dy, dx = self.action_mapping[action]

        if action in [0, 2]:
            if self.can_go(self.y, self.x + dx):
                self.x += dx
                if track: trajectory.append((self.y, self.x, reward, t))

                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, reward, t))

                    if self.can_go(self.y, self.x + dx):
                        self.x += dx
                        if track: trajectory.append((self.y, self.x, reward, t))

                        if not self.on_solid_grounds():
                            self.y += 1
                            if track: trajectory.append((self.y, self.x, reward, t))
            else:
                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, reward, t))

                    if not self.on_solid_grounds():
                        self.y += 1
                    if track: trajectory.append((self.y, self.x, reward, t))
        else:
            reward -= 4
            if self.on_solid_grounds():
                if self.can_go(self.y + dy, self.x):
                    self.y += dy
                    if track: trajectory.append((self.y, self.x, reward, t))

                    if self.can_go(self.y + dy, self.x):
                        self.y += dy
                        if track: trajectory.append((self.y, self.x, reward, t))

            elif random() < 1 / 3 and self.can_go(self.y + dy, self.x):
                reward += 2
                self.y += dy
                if track: trajectory.append((self.y, self.x, reward, t))

            else:
                self.y += 1
                if track: trajectory.append((self.y, self.x, reward, t))

                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, reward, t))

        if self.in_end_state():
            done = True
            reward = 0

        if (self.y, self.x) in self.visited_states:
            reward -= 1

        self.scan_surroundings(action)
        self.prev_actions.append(action)
        self.visited_states.add((self.y, self.x))

        return self.surroundings, reward, done, False, {}

    def scan_surroundings(self, action) -> None:
        area = zeros(shape=(9, 9))

        for y, y_v in enumerate(range(self.y - 4, self.y + 5)):
            for x, x_v in enumerate(range(self.x - 4, self.x + 5)):
                if self.on_grid(y_v, x_v):
                    if (y_v, x_v) in self.visited_states and self.grid[y_v, x_v] == 1.0:
                        area[y, x] = 0
                    else:
                        area[y, x] = self.grid[y_v, x_v]
                else:
                    area[y, x] = -1

        self.surroundings = area.flatten()  # concatenate((self.prev_actions, area.flatten()))

    def can_go(self, y: int, x: int) -> bool:
        return self.grid[y, x] != 2.0

    def on_grid(self, y: int, x: int) -> bool:
        return self.grid_y.min <= y <= self.grid_y.max and self.grid_x.min <= x <= self.grid_x.max

    def on_solid_grounds(self):
        return self.grid[self.y + 1, self.x] in self.solids

    def in_end_state(self) -> bool:
        return self.y == 1 and self.x == 46

    def reset(self) -> ndarray:
        self.y, self.x = 46, 1
        self.prev_actions.clear()
        self.scan_surroundings(-1)
        return self.surroundings, {}

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


    def create_gif(self, agent: list[list, list], color_bar: bool = False):
        colors = ['#00d400', '#FFFFFF', '#000000', '#2a7fff', '#f77979', '#FFA500']

        def draw_frame(time_step, t, y, x, e):
            template = self.grid.copy()
            template[y, x] = 5

            figure(figsize=(10, 10))
            fig = heatmap(template, cmap=colors, cbar=color_bar)
            title(f'{self.name} \nTime: {t + 1} - energy: {e}', size=30)

            for _, spine in fig.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(1)

            savefig(f'gif/img_{time_step}.png')
            close()

        y_values, x_values, e_values, t_values = zip(*agent)

        for t in tqdm(range(len(x_values)), desc='Creating plots'):
            draw_frame(t, t_values[t], y_values[t], x_values[t], e_values[t])

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
