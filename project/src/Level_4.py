from project.src.utils.Environments import Environment_3 as Environment
from project.src.utils import SARSA, read_tsv_file, read_fans

from random import randint

enemies = read_fans('../resources/enemies_3.dat')
fans = {(key, randint(value[0], value[1])) for key, value in enemies.items()}
grid = read_tsv_file('../resources/map_3.dat', enemies={}, start=(30, 1), end=(1, 30))

env = Environment(name="Level 4", grid=grid, enemies=enemies, project_path='Levels')
# env.plot(color_bar=False, save=True)

sarsa = SARSA(environment=env, epsilon=0.0, gamma=1.0)
sarsa.train(num_episodes=100_000)


trajectory, time = sarsa.get_optimal_trajectory()

print(f'Energy cost: {time}')
print(trajectory)
env.create_gif(agent=trajectory)
