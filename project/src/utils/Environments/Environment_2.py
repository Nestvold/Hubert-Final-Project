from matplotlib.pyplot import title, savefig, figure, close
from numpy.random import uniform, choice
from numpy import array, zeros, ndarray
from imageio.v2 import imread
from imageio import mimsave
from seaborn import heatmap
from .utils import BaseEnv
from tqdm import tqdm
import os


class Environment_2(BaseEnv):
    def __init__(self, name: str, grid: ndarray, enemies: set, pMM: float = 1.0, project_path: str = '', data: dict = None):
        super().__init__(name, grid, project_path, data)
        self.enemies = enemies
        self.pMM = pMM

        # State
        self.y: int = 30
        self.x: int = 1
        self.enemy = next(iter(enemies))[1]

    def search(self, y: int, x: int):
        grid = zeros(shape=len(self.action_mapping.keys()), dtype=float)
        for action in self.action_mapping.keys():
            dy, dx = self.action_mapping[action]
            new_y, new_x = y + dy, x + dx

            if self.on_grid(new_y, new_x) and self.can_go(new_y, new_x):
                grid[action] = 0
            else:
                grid[action] = float('-inf')

        return array(grid)

    def update_enemies(self):
        if self.enemy == 2: move = 1
        elif self.enemy == 29: move = -1
        else: move = choice([-1, 1])

        if move == -1:
            self.enemies = {(20, max(2, self.enemy + move))}
            self.enemy += move
        elif move == 1:
            self.enemies = {(20, min(29, self.enemy + move))}
            self.enemy += move

    def step(self, action: int, track: bool = False, t: int = None, trajectory: list = None) -> tuple[tuple[int, int, int], float, bool]:
        done = False
        reward = -1
        dy, dx = self.action_mapping[action]

        if action in [0, 2]:
            if self.can_go(self.y, self.x + dx):
                self.x += dx
                if track: trajectory.append((self.y, self.x, self.enemy, t))

                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, self.enemy, t))

                    if self.can_go(self.y, self.x + dx):
                        self.x += dx
                        if track: trajectory.append((self.y, self.x, self.enemy, t))

                        if not self.on_solid_grounds():
                            self.y += 1
                            if track: trajectory.append((self.y, self.x, self.enemy, t))
            else:
                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, self.enemy, t))

                    if not self.on_solid_grounds():
                        self.y += 1
                        if track: trajectory.append((self.y, self.x, self.enemy, t))
        else:
            reward -= 4
            if self.on_solid_grounds():
                if self.can_go(self.y + dy, self.x):
                    self.y += dy
                    if track: trajectory.append((self.y, self.x, self.enemy, t))

                    if self.can_go(self.y + dy, self.x):
                        self.y += dy
                        if track: trajectory.append((self.y, self.x, self.enemy, t))

            else:
                self.y += 1
                if track: trajectory.append((self.y, self.x, self.enemy, t))

                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, self.enemy, t))

        # Check if MM sees us in the new position
        if self.seen():
            self.y, self.x = 30, 1
            if track: trajectory.append((self.y, self.x, self.enemy, t))
            return (self.y, self.x), reward, done

        # Move MM
        self.update_enemies()

        if track: trajectory.append((self.y, self.x, self.enemy, t))

        # Check if MM sees us
        if self.seen():
            self.y, self.x = 30, 1
            if track: trajectory.append((self.y, self.x, self.enemy, t))
            return (self.y, self.x), reward, done

        if self.in_end_state():
            done = True
            reward = 0

        if 17 < self.y < 22:
            return (self.y, self.x, self.enemy), reward, done

        return (self.y, self.x), reward, done

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

        def draw_frame(time_step, t, y, x, MM):
            template = self.grid.copy()
            template[y, x] = 5
            template[20, MM] = 4

            figure(figsize=(10, 10))
            fig = heatmap(template, cmap=colors, cbar=color_bar)
            title(f'{self.name} \ncost: {t + 1}', size=30)

            for _, spine in fig.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(1)

            savefig(f'gif/img_{time_step}.png')
            close()

        y_values, x_values, MM_value, t_values = zip(*agent)

        for t in tqdm(range(len(x_values)), desc='Creating GIF'):
            draw_frame(t, t_values[t], y_values[t], x_values[t], MM_value[t])

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
