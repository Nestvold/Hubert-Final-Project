from project.src.utils.Environments import Environment_1 as Environment
from project.src.utils import SARSA, read_tsv_file, read_MM

from random import randint

MM = read_MM('../project_data/enemies_1.dat')
grid = read_tsv_file('../project_data/map_1.dat', enemies=MM, start=(30, 1), end=(1, 30))

env = Environment(name="Level 2 - pMM=1.0", grid=grid, enemies=MM, pMM=1.0, project_path='Levels')
# env.plot(color_bar=False, save=True)

sarsa = SARSA(environment=env)
sarsa.train()

trajectory, time = sarsa.get_optimal_trajectory()

print(f'Trajectory - {trajectory}')
print(f'Time - {time}')
env.create_gif(agent=trajectory)