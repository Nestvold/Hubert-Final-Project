from project.src.utils.Environments import Environment_2 as Environment
from project.src.utils import SARSA, read_tsv_file, read_MM

for pMM in [.0, .10, .20, .25, .30, .40, .50, .60, .66, .70, .75, .80, .90, 1.00]:

    MM = read_MM('../resources/enemies_1.dat')
    grid = read_tsv_file('../resources/map_1.dat', enemies=MM, start=(30, 1), end=(1, 30))

    env = Environment(name=f'Level 2, pMM={pMM}', grid=grid, enemies=MM, pMM=pMM, project_path='Levels')
    # env.plot(color_bar=False, save=True)

    sarsa = SARSA(environment=env, epsilon=0.0)
    sarsa.train(num_episodes=100_000)

    trajectory, reward, time = sarsa.get_optimal_trajectory()

    print(f'Energy cost: {reward}')
    print(f'Total time: {time}')

    env.create_gif(agent=trajectory)
