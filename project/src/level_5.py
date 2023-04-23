from project.src.utils.Environments import Environment_4 as Environment
from project.src.utils import SARSA, read_tsv_file


grid = read_tsv_file('../resources/map_4.dat', enemies={}, start=(46, 1), end=(1, 46))

env = Environment(name="Level 5", grid=grid, project_path='Levels')
# env.plot(color_bar=False, save=True)

sarsa = SARSA(environment=env, epsilon=0.0, gamma=1.0)
sarsa.train(num_episodes=100_000)


trajectory, time = sarsa.get_optimal_trajectory()

print(f'Energy cost: {time}')
env.create_gif(agent=trajectory)
