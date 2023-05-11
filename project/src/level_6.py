# Utils module
from project.src.utils import Agent
from utils import ENVIRONMENTS

environments = ENVIRONMENTS(folder_path='resources/training_maps/').list_of_envs

n_stacks = 1
cuda_nr = 0

agent = Agent(n_stacks=n_stacks, environments=environments, epsilon=0.5, learning_rate=0.001)

agent.train()
