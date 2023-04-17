from project.src.utils import read_tsv_file, read_MM
from project.src.utils.Environments import Environment

MM = read_MM('enemies_1.dat')
grid = read_tsv_file('map_1.dat', enemies=MM, start=(30, 1), end=(1, 30))

env = Environment(name="Level 1", grid=grid, enemies=MM, pMM=0.5, project_path='')
env.plot(color_bar=False)