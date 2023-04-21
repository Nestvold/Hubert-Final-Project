from matplotlib.pyplot import title, savefig, figure, close
from numpy.random import uniform, choice
from numpy import array, zeros, ndarray
from imageio.v2 import imread
from imageio import mimsave
from seaborn import heatmap
from .utils import BaseEnv
from tqdm import tqdm
import os


class Environment_3(BaseEnv):
    def __init__(self, name: str, grid: ndarray, enemies: set, pMM: float = 1.0, project_path: str = '', data: dict = None):
        super().__init__(name, grid, project_path, data)
        self.enemies = {key: list(value) for key, value in enemies.items()}
        self.pMM = pMM

        # State
        self.y: int = 30
        self.x: int = 1
        self.walking_fans = {(key, value[0]) for key, value in enemies.items()}

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

        scan = self.scan_surroundings()

        # Check if MM sees us in the new position
        if self.seen():
            self.y, self.x = 30, 1
            if track: trajectory.append((self.y, self.x, self.walking_fans, t))
            return (self.y, self.x, scan), reward, done

        # Move MM
        self.update_enemies()

        if track: trajectory.append((self.y, self.x, self.walking_fans, t))

        # Check if MM sees us
        if self.seen():
            self.y, self.x = 30, 1
            if track: trajectory.append((self.y, self.x, self.walking_fans, t))
            return (self.y, self.x, scan), reward, done

        if self.in_end_state():
            done = True
            reward = 0

        return (self.y, self.x, scan), reward, done

    def scan_surroundings(self):
        area = zeros(shape=(9, 9))
        for y, y_v in enumerate(range(self.y - 4, self.y + 5)):
            for x, x_v in enumerate(range(self.x - 4, self.x + 5)):
                if self.on_grid(y_v, x_v) and (y, x) in self.enemies:
                    area[y, x] = 1

        grid = tuple(map(tuple, area))
        return hash(grid)

    def update_enemies(self):
        for fan in self.walking_fans:
            y, x = fan

            if self.y == y:
                if self.x < x < self.enemies[y][0]:
                    x -= 1
                elif self.x > x > self.enemies[y][1]:
                    x += 1
            else:
                if x == self.enemies[y][0]:
                    x += 1
                elif x == self.enemies[y][1]:
                    x -= 1
                else:
                    x += choice([-1, 1])
            fan = y, x

    def seen(self):
        return (self.y, self.x) in self.enemies and uniform() < self.pMM

    def on_solid_grounds(self):
        return self.grid[self.y + 1, self.x] in self.solids

    def encountered_enemy(self, y: int, x: int) -> bool:
        return (y, x) in self.enemies and self.pMM > uniform()

    def in_end_state(self) -> bool:
        return self.y == 1 and self.x == 30

    def reset(self) -> tuple[int, int, int]:
        self.y = 30
        self.x = 1
        return self.y, self.x

    def create_gif(self, agent: list[list, list], color_bar: bool = False):
        colors = ['#00d400', '#FFFFFF', '#000000', '#2a7fff', '#f77979', '#FFA500']

        def draw_frame(time_step, t, y, x, fans):
            old_val_1 = self.grid[y, x]
            self.grid[y, x] = 5
            prev_fan_vals = {}

            for fan in fans:
                prev_fan_vals[fan] = self.grid[fan]
                self.grid[fan] = 4

            figure(figsize=(10, 10))
            fig = heatmap(self.grid, cmap=colors, cbar=color_bar)
            title(f'{self.name} \ncost: {t + 1}', size=30)

            for _, spine in fig.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(1)

            self.grid[y, x] = old_val_1

            for fan in prev_fan_vals:
                self.grid[fan] = o_vals[fan]

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
