# Utils module
from .utils import GridValues

# Other modules
from numpy import array, ndarray, zeros, int32, concatenate
from matplotlib.pyplot import title, savefig, figure, close
from gym.spaces import Discrete, Box
from imageio.v2 import imread
from imageio import mimsave
from seaborn import heatmap
from random import random
from tqdm import tqdm
from abc import ABC
from gym import Env
import os


class OpenAIEnv(Env, ABC):
    def __init__(self, name: str, grid: ndarray, project_path: str = ''):
        self.name = name
        self.project_path = project_path

        # Gym
        self.action_space = Discrete(3)  # 0 = go left, 1 = jump, 2 = go right

        # y, x, 9 x 9 grid -> 0: nothing, 1: Air, 2: Solid, 3: Semisolid, 4: MM, 5: fan
        self.observation_space = Box(
            low=array([0] * 81),  # Low Bound
            high=array([3] * 81),  # High Bound
            dtype=int32  # Type: Integer
        )

        # State representation
        self.y, self.x = 46, 1
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

    def step(self, action: int, track: bool = False, t: int = None, trajectory: list = None) -> tuple[ndarray, float, bool]:
        if action not in self.action_mapping:
            raise ValueError(f'Invalid action: {action}.')

        done = False
        reward = -1.0
        dy, dx = self.action_mapping[action]

        if action in [0, 2]:
            if self.can_go(self.y, self.x + dx):
                self.x += dx
                if track: trajectory.append((self.y, self.x, t))

                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, t))

                    if self.can_go(self.y, self.x + dx):
                        self.x += dx
                        if track: trajectory.append((self.y, self.x, t))

                        if not self.on_solid_grounds():
                            self.y += 1
                            if track: trajectory.append((self.y, self.x, t))
            else:
                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, t))

                    if not self.on_solid_grounds():
                        self.y += 1
                    if track: trajectory.append((self.y, self.x, t))
        else:
            reward -= 4
            if self.on_solid_grounds():
                if self.can_go(self.y + dy, self.x):
                    self.y += dy
                    if track: trajectory.append((self.y, self.x, t))

                    if self.can_go(self.y + dy, self.x):
                        self.y += dy
                        if track: trajectory.append((self.y, self.x, t))

            elif random() < 1 / 3 and self.can_go(self.y + dy, self.x):
                reward += 2
                self.y += dy
                if track: trajectory.append((self.y, self.x, t))

            else:
                self.y += 1
                if track: trajectory.append((self.y, self.x, t))

                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, t))

        if self.in_end_state():
            done = True
            reward = 0

        self.scan_surroundings()

        return self.surroundings, reward, done, False,  {}

    def scan_surroundings(self) -> None:
        area = zeros(shape=(9, 9))
        for y, y_v in enumerate(range(self.y - 4, self.y + 5)):
            for x, x_v in enumerate(range(self.x - 4, self.x + 5)):
                if self.on_grid(y_v, x_v):
                    area[y, x] = self.grid[y, x]
        self.surroundings = area.flatten()

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
        self.scan_surroundings()
        return self.surroundings, {}

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
