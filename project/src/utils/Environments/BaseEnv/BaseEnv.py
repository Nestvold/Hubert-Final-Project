# State

# Other modules
from matplotlib.pyplot import title, savefig, show, figure, close
from seaborn import heatmap
from imageio.v2 import imread
from imageio import mimsave
from numpy import ndarray


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
        self.y = GridValues(grid)
        self.x = GridValues(grid[0])

    def on_grid(self, y: int, x: int) -> bool:
        return self.y.min <= y <= self.y.max and self.x.min <= x <= self.x.max

    def can_go(self, y: int, x: int) -> bool:
        return self.grid[y, x] != 0.0

    def in_end_state(self, y: int, x: int) -> bool:
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
        if agent is not None:
            color_map = ['#90EE90', '#FFFFFF', '#000000', '#4CC9F0', '#F72585', '#FFA500']
        else:
            color_map = ['#90EE90', '#FFFFFF', '#000000', '#4CC9F0', '#F72585']

        figure(figsize=(10, 10))
        map = heatmap(data=self.grid, cmap=color_map, cbar=color_bar)

        for _, spine in map.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)

        title(self.name.capitalize(), size=30)

        if save:
            savefig(f'{self.project_path}/assets/{self.name}.png')
        show()

    def create_gif(self, agent: list[list, list], data: dict, color_bar: bool = False):
        colors = ['black', 'white', 'red', 'blue', 'green']
        stations = {key for key in data['stations'].keys()}

        self.grid[data['start']] = 2
        self.grid[data['finish']] = 2

        for station in stations:
            self.grid[station] = 3

        def draw_frame(time_step, y, x, b):
            old_val = self.grid[y, x]
            self.grid[y, x] = 4
            figure(figsize=(10, 10))
            fig = heatmap(self.grid, cmap=colors, cbar=color_bar)
            title(f'Battery: {b}', size=50)

            for _, spine in fig.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(1)

            self.grid[y, x] = old_val

            savefig(f'gif/img_{time_step}.png')
            close()

        # Unwrap state (y, x, b)
        y_values, x_values, battery_lvl = zip(*agent)

        for t in range(len(x_values)):
            draw_frame(t, y_values[t], x_values[t], battery_lvl[t])

        frames = []

        for t in range(len(x_values)):
            image = imread(f'gif/img_{t}.png')
            frames.append(image)

        mimsave(uri=f'project/Episodes/gifs/{self.name}.gif',
                ims=frames,
                fps=10
                )
