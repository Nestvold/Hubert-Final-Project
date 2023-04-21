# Other modules
from matplotlib.pyplot import title, savefig, show, figure, close
from seaborn import heatmap
from imageio.v2 import imread
from imageio import mimsave
from numpy import ndarray
import os


class GridValues:
    def __init__(self, grid: ndarray):
        self.min = 0
        self.max = len(grid) - 1


class BaseEnv:
    def __init__(self, name: str, grid: ndarray, project_path: str, data: dict):
        self.name = name
        self.grid = grid
        self.project_path = project_path
        self.data = data
        self.grid_y = GridValues(grid)
        self.grid_x = GridValues(grid[0])

        self.solids = set([2, 3])
        self.action_mapping = {0: (0, -1), 1: (-1, 0), 2: (0, 1)}

    def on_grid(self, y: int, x: int) -> bool:
        return self.grid_y.min <= y <= self.grid_y.max and self.grid_x.min <= x <= self.grid_x.max

    def can_go(self, y: int, x: int) -> bool:
        return self.grid[y, x] != 2.0

    def in_end_state(self) -> bool:
        raise NotImplementedError

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        raise NotImplementedError

    def reset(self) -> tuple[int, int]:
        raise NotImplementedError

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
            savefig(f'{self.project_path}/assets/{self.name}.png')
        show()

    def create_gif(self, agent: list[list, list], color_bar: bool = False):
        colors = ['#00d400', '#FFFFFF', '#000000', '#2a7fff', '#f77979', '#FFA500']

        def draw_frame(time_step, t, y, x):
            old_val = self.grid[y, x]
            self.grid[y, x] = 5
            figure(figsize=(10, 10))
            fig = heatmap(self.grid, cmap=colors, cbar=color_bar)
            title(f'{self.name} \ncost: {t + 1}', size=30)

            for _, spine in fig.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(1)

            self.grid[y, x] = old_val

            savefig(f'gif/img_{time_step}.png')
            close()

        y_values, x_values, t_values = zip(*agent)

        for t in range(len(x_values)):
            draw_frame(t, t_values[t], y_values[t], x_values[t])

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