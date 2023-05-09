from matplotlib.pyplot import title, savefig, figure, close
from numpy.random import uniform, choice, randint
from numpy import array, zeros, ndarray, save
from collections import defaultdict
from imageio.v2 import imread
from imageio import mimsave
from seaborn import heatmap
from .extra import BaseEnv
from tqdm import tqdm
import os


class Environment_4(BaseEnv):
    def __init__(self, name: str, grid: ndarray, enemies: set, project_path: str = '', data: dict = None):
        super().__init__(name, grid, project_path, data)
        self.enemies = {key: list(value) for key, value in enemies.items()}
        self.walking_fans = {key: [randint(value[0], value[1])] for key, value in self.enemies.items()}
        self.walking_fans[3].append(4)

        # State
        self.y: int = 30
        self.x: int = 1
        self.scan = None

    def step(self, action: int, track: bool = False, t: int = None, trajectory: list = None) -> tuple[tuple[int, int, int], float, bool]:
        done = False
        reward = -1
        dy, dx = self.action_mapping[action]
        last_seen = self.y

        if action in [0, 2]:
            if self.can_go(self.y, self.x + dx):
                self.x += dx
                if track: trajectory.append((self.y, self.x, self.walking_fans, reward, t))

                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, self.walking_fans, reward, t))

                    if self.can_go(self.y, self.x + dx):
                        self.x += dx
                        if track: trajectory.append((self.y, self.x, self.walking_fans, reward, t))

                        if not self.on_solid_grounds():
                            self.y += 1
                            if track: trajectory.append((self.y, self.x, self.walking_fans, reward, t))
            else:
                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, self.walking_fans, reward, t))

                    if not self.on_solid_grounds():
                        self.y += 1
                        if track: trajectory.append((self.y, self.x, self.walking_fans, reward, t))
        else:
            reward -= 4
            if self.on_solid_grounds():
                if self.can_go(self.y + dy, self.x):
                    self.y += dy
                    if track: trajectory.append((self.y, self.x, self.walking_fans, reward, t))

                    if self.can_go(self.y + dy, self.x):
                        self.y += dy
                        if track: trajectory.append((self.y, self.x, self.walking_fans, reward, t))

            else:
                self.y += 1
                if track: trajectory.append((self.y, self.x, self.walking_fans, reward, t))

                if not self.on_solid_grounds():
                    self.y += 1
                    if track: trajectory.append((self.y, self.x, self.walking_fans, reward, t))

        # Move fans
        self.update_enemies(last_seen)
        if track: trajectory.append((self.y, self.x, self.walking_fans, reward, t))

        # Check if fan sees us in the new position
        reward += self.seen()

        if self.in_end_state():
            done = True
            reward = 0

        self.scan_surroundings()

        return (self.y, self.x, self.scan), reward, done

    def scan_surroundings(self) -> None:
        area = zeros(shape=(9, 9))
        enemies = set((key, value) for key, values in self.walking_fans.items() for value in values)

        for y, y_v in enumerate(range(self.y - 4, self.y + 5)):
            for x, x_v in enumerate(range(self.x - 4, self.x + 5)):
                if self.on_grid(y_v, x_v) and (y_v, x_v) in enemies:
                    area[y, x] = 1

        grid = tuple(map(tuple, area))
        self.scan = hash(grid)

    def update_enemies(self, last_y):
        new_positions = {}

        for y, fans in self.walking_fans.items():
            new_positions[y] = []
            for fan in fans:
                if last_y == y:
                    if self.x < fan > self.enemies[y][0]:
                        fan -= 1
                    if self.x > fan < self.enemies[y][1]:
                        fan += 1
                else:
                    if fan == self.enemies[y][0]:
                        fan += 1
                    elif fan == self.enemies[y][1]:
                        fan -= 1
                    else:
                        fan += choice([-1, 1])
                new_positions[y].append(fan)

        self.walking_fans = new_positions

    def seen(self):
        reward = 0
        enemies = set((key, value) for key, values in self.walking_fans.items() for value in values)
        if (self.y, self.x) in enemies:
            reward -= randint(19, 99)
            if self.y == 4 and len(enemies) == 3:
                reward -= randint(20, 100)
        return reward

    def on_solid_grounds(self):
        return self.grid[self.y + 1, self.x] in self.solids

    def in_end_state(self) -> bool:
        return self.y == 1 and self.x == 30

    def reset(self) -> tuple[int, int, int]:
        self.y = 30
        self.x = 1
        self.scan_surroundings()
        return self.y, self.x, self.scan

    def create_gif(self, agent: list[list, list], color_bar: bool = False):
        colors = ['#00d400', '#FFFFFF', '#000000', '#2a7fff', '#f77979', '#FFA500']
        r = 0
        i = 0

        def draw_frame(time_step, t, y, x, fans, reward):
            nonlocal r
            nonlocal i

            if t == i:
                r += abs(reward)
                i += 1

            template = self.grid.copy()
            template[y, x] = 5

            for s, x_s in fans.items():
                for j in x_s:
                    template[s, j] = 4

            figure(figsize=(10, 10))
            fig = heatmap(template, cmap=colors, cbar=color_bar)
            title(f'{self.name} \nTime: {t + 1} | Energy: {r}', size=25)

            for _, spine in fig.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(1)

            savefig(f'gif/img_{time_step}.png')
            close()

        y_values, x_values, fans, rewards, t_values = zip(*agent)

        for t in tqdm(range(len(x_values)), desc='[CREATING PLOTS]'):
            draw_frame(t, t_values[t], y_values[t], x_values[t], fans[t], rewards[t])

        frames = []

        for t in tqdm(range(len(x_values)), desc='[CREATING  GIF ]'):
            image = imread(f'gif/img_{t}.png')
            frames.append(image)

        mimsave(uri=f'Levels/gifs/{self.name}_{r}_test.gif',
                ims=frames,
                fps=8
                )

        #Delete plots
        folder_path = "gif"

        # Get all the file names in the folder
        files = os.listdir(folder_path)

        # Loop through the files and delete them
        for file_name in tqdm(files, desc='[DELETING PLOTS]'):
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        print('[DONE]')
