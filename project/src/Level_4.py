from project.src.utils.Environments import Environment_3 as Environment
from project.src.utils import SARSA, read_tsv_file, read_fans


enemies = read_fans('../resources/enemies_3.dat')
fans = {(key, value[0]) for key, value in enemies.items()}
grid = read_tsv_file('../resources/map_3.dat', enemies=fans, start=(30, 1), end=(1, 30))

env = Environment(name="Level 4", grid=grid, enemies=enemies, pMM=1.0, project_path='Levels')
# env.plot(color_bar=False, save=True)

sarsa = SARSA(environment=env, epsilon=0.0, gamma=1.0)
# sarsa.train(num_episodes=1)
print(sarsa.Q[(30, 1)])
trajectory, time = sarsa.get_optimal_trajectory()

print(f'Trajectory:  {trajectory}')
print(f'Energy cost: {time}')
env.create_gif(agent=trajectory)
