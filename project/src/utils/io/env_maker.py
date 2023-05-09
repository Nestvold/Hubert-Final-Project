# Environment module
from project.src.utils.Environments import Environment_6 as Environment

# Other modules
from numpy import ndarray, zeros, array
from os import listdir, path


class ENVIRONMENTS:
    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path
        self.folders = [name for name in listdir(folder_path) if path.isdir(path.join(folder_path, name))]
        self.list_of_envs = self.__create_envs()

    def __create_envs(self):
        envs = []

        for folder in self.folders:
            files = listdir(f'{self.folder_path}/{folder}')
            enemy_files = []
            grid = None

            for file in files:

                file = f'{folder}/{file}'
                if path.getsize(f'{self.folder_path}/{file}') > 0:
                    if 'enemies' in file or 'fans' in file:
                        enemy_files.append(file)
                    else:
                        env_name = file.split('/')[-1].replace('.dat', '').replace('_', ' ').capitalize()
                        grid = self.read_environment(file)

            MM, fans = self.read_enemy(enemy_files)

            env = Environment(name=env_name, grid=grid, MM=MM, fans=fans, start_coords=(grid.shape[0] - 2, 1))
            envs.append(env)
        return envs

    def read_enemy(self, files: list):
        MM, fans = [], []

        for file in files:
            with open(f'{self.folder_path}/{file}', mode='r') as file:
                enemies = [[int(x) for x in line.split() if int(x)] for line in file if line.split()]
                for enemy in enemies:
                    if enemy[-1] == 1:
                        MM.append([x - 1 for x in enemy[:2]])
                    elif enemy[-1] == 2:
                        fans.append([x - 1 for x in enemy[:2]])
        return MM, fans

    def read_environment(self, file_path: str) -> ndarray:
        """
        Reads a TSV file and puts the value in an array to represent the environment.
        :param enemies:
        :param filename:
        :return: An array representing the values from file
        """

        with open(f'{self.folder_path}/{file_path}', mode='r') as file:
            content = file.readlines()
            y, x = len(content), len(content[0].split('\t'))
            grid = zeros((y, x), dtype=float)

            for i, line in enumerate(content):
                line = line.split('\t')
                values = array([float(x) for x in line])
                grid[i] = values

        return grid
