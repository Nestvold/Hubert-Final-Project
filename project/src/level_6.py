from project.src.utils.Environments import OpenAIEnv as Environment
from project.src.utils import SARSA, read_tsv_file

from gym.wrappers import FrameStack, LazyFrames

grid = read_tsv_file('../resources/map_4.dat', enemies={}, start=(46, 1), end=(1, 46))

env = Environment(name="Level 5", grid=grid, project_path='Levels')
env = FrameStack(env=env, num_stack=2)


sarsa = SARSA(environment=env, epsilon=0.1, gamma=1.0)
sarsa.train(num_episodes=100_000)


trajectory, reward, time = sarsa.get_optimal_trajectory()

print(f'Energy cost: {reward}')
print(f'Total time: {time}')

env.create_gif(agent=trajectory)
