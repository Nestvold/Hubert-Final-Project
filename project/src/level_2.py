from project.src.utils.Environments import Environment_1 as Environment
from project.src.utils import SARSA, read_tsv_file, read_MM


MM = read_MM('../resources/enemies_1.dat')
grid = read_tsv_file('../resources/map_1.dat', enemies=MM, start=(30, 1), end=(1, 30))

env = Environment(name="Level 2", grid=grid, enemies=MM, pMM=0.00, project_path='Levels')
# env.plot(color_bar=False, save=True)

sarsa = SARSA(environment=env, epsilon=0.0)
sarsa.train(num_episodes=100_000)

trajectory, time = sarsa.get_optimal_trajectory()

print(f'Trajectory - {trajectory}')
print(f'Time - {time}')
env.create_gif(agent=trajectory)
