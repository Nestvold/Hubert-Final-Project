from project.src.utils.Environments import Environment_5 as Environment
# from project.src.utils import SARSA_N_STEP as SARSA
from project.src.utils import SARSA, read_tsv_file

from gym.wrappers import FrameStack, LazyFrames

grid = read_tsv_file('map_4.dat', enemies={})

env = Environment(name="Level 5", n_actions=10, grid=grid, project_path='Levels')
# env.plot(color_bar=False, save=True)
# env = FrameStack(env=env, num_stack=3)

sarsa = SARSA(environment=env, epsilon=0.0)
trajectory = sarsa.train(num_episodes=100_000)


trajectory, reward, time = sarsa.get_optimal_trajectory()

# print(f'Energy cost: {reward}')
# print(f'Total time: {time}')

env.create_gif(agent=trajectory)
