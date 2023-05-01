from project.src.utils.Environments import Environment_3 as Environment
from project.src.utils import SARSA, read_tsv_file, read_walking_MM


MM = read_walking_MM('../resources/enemies_2.dat')
grid = read_tsv_file('../resources/map_2.dat', enemies={}, start=(30, 1), end=(1, 30))

env = Environment(name="Level 3", grid=grid, enemies=MM, pMM=1.0, project_path='Levels')
# env.plot(color_bar=False, save=True)

sarsa = SARSA(environment=env, epsilon=0.0, gamma=1.0)
sarsa.train(num_episodes=100_000)

trajectory, reward, time = sarsa.get_optimal_trajectory()

print(f'Energy cost: {reward}')
print(f'Total time: {time}')

# env.create_gif(agent=trajectory)
