from matplotlib.pyplot import title, savefig, figure, close
from numpy.random import uniform, choice, randint
from numpy import array, zeros, ndarray, save
from collections import defaultdict
from imageio.v2 import imread
from imageio import mimsave
from seaborn import heatmap
from .utils import BaseEnv
from tqdm import tqdm
import os


class Environment_4(BaseEnv):
    def __init__(self, name: str, grid: ndarray, project_path: str = '', data: dict = None):
        super().__init__(name, grid, project_path, data)

        # State
        self.y: int = 46
        self.x: int = 1
        self.scan = None
        self.scan_surroundings()

    def step(self, action: int, track: bool = False, t: int = None, trajectory: list = None) -> tuple[tuple[int, int, int], float, bool]:
        done = False
        reward = -1
        dy, dx = self.action_mapping[action]

        if action in [0, 2]:
            if self.can_go(self.y, self.x + dx):
                self.x += dx
                if track: trajectory.append((self.y, self.x, self.walking_fans, t))

                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, self.walking_fans, t))

                    if self.can_go(self.y, self.x + dx):
                        self.x += dx
                        if track: trajectory.append((self.y, self.x, self.walking_fans, t))

                        if not self.on_solid_grounds():
                            self.y += 1
                            if track: trajectory.append((self.y, self.x, self.walking_fans, t))
            else:
                if not self.on_solid_grounds():
                    # Added rocket engine
                    if uniform() < 1/3 and self.can_go(self.y - 1, self.x):
                        reward -= 2
                        self.y -= 1
                        if track: trajectory.append((self.y, self.x, self.walking_fans, t))
                    else:
                        self.y += 1
                        if track: trajectory.append((self.y, self.x, self.walking_fans, t))

                        if not self.on_solid_grounds():
                            self.y += 1
                        if track: trajectory.append((self.y, self.x, self.walking_fans, t))
        else:
            reward -= 4
            if self.on_solid_grounds():
                if self.can_go(self.y + dy, self.x):
                    self.y += dy
                    if track: trajectory.append((self.y, self.x, self.walking_fans, t))

                    if self.can_go(self.y + dy, self.x):
                        self.y += dy
                        if track: trajectory.append((self.y, self.x, self.walking_fans, t))

            else:
                self.y += 1
                if track: trajectory.append((self.y, self.x, self.walking_fans, t))

                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, self.walking_fans, t))

        if self.in_end_state():
            done = True
            reward = 0

        self.scan_surroundings()

        return (self.y, self.x, self.scan), reward, done

    def scan_surroundings(self) -> None:
        area = zeros(shape=(9, 9))

        for y, y_v in enumerate(range(self.y - 4, self.y + 5)):
            for x, x_v in enumerate(range(self.x - 4, self.x + 5)):
                if self.on_grid(y_v, x_v):
                    area[y, x] = self.grid[y, x]

        grid = tuple(map(tuple, area))
        self.scan = hash(grid)

    def on_solid_grounds(self):
        return self.grid[self.y + 1, self.x] in self.solids

    def encountered_enemy(self, y: int, x: int) -> bool:
        return (y, x) in self.enemies and self.pMM > uniform()

    def in_end_state(self) -> bool:
        return self.y == 1 and self.x == 30

    def reset(self) -> tuple[int, int, int]:
        self.y = 30
        self.x = 1
        self.scan_surroundings()
        return self.y, self.x, self.scan

    def create_gif(self, agent: list[list, list], color_bar: bool = False):
        colors = ['#00d400', '#FFFFFF', '#000000', '#2a7fff', '#f77979', '#FFA500']

        def draw_frame(time_step, t, y, x, fans):
            template = self.grid.copy()
            template[y, x] = 5

            for s, x_s in fans.items():
                for j in x_s:
                    template[s, j] = 4

            figure(figsize=(10, 10))
            fig = heatmap(template, cmap=colors, cbar=color_bar)
            title(f'{self.name} \ncost: {t + 1}', size=30)

            for _, spine in fig.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(1)

            savefig(f'gif/img_{time_step}.png')
            close()

        y_values, x_values, fans, t_values = zip(*agent)

        for t in tqdm(range(len(x_values)), desc='Creating GIF'):
            draw_frame(t, t_values[t], y_values[t], x_values[t], fans[t])

        frames = []

        for t in range(len(x_values)):
            image = imread(f'gif/img_{t}.png')
            frames.append(image)

        mimsave(uri=f'Levels/gifs/{self.name}.gif',
                ims=frames,
                fps=10
                )

        #Delete plots
        folder_path = "gif"

        # Get all the file names in the folder
        files = os.listdir(folder_path)

        # Loop through the files and delete them
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        print('[GIF CREATED]')
