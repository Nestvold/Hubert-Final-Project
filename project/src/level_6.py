# Utils module
from gym.wrappers import FrameStack
from train_agent import Agent
from utils import ENVIRONMENTS
from numpy import array
from random import choice
import torch

n_stacks = 2

envs = ENVIRONMENTS(folder_path='resources/training_maps/').list_of_envs
env = envs[13]
env = FrameStack(env=env, num_stack=n_stacks)

version = 'base_plus'
print('Loading model ...', end=' ')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = torch.load(f'C:\\Users\\Fredrik\\Documents\\GitHub\\Hubert-Final-Project\\project\\src\\models\\{version}\\agent_{n_stacks}.pt').to(device)
agent.training = False
print('done!')

trajectory = []
total_reward = 0

print('Testing agent...', end=' ')
# Reset env
state = env.reset()

while True:
    # Convert the state to a PyTorch tensor
    state_array = array(state)
    state_tensor = torch.from_numpy(state_array).float()
    state_tensor = state_tensor.to(device)

    # Pass the state through the agent to obtain the action
    action, _, _, _ = agent.get_action_and_value(state_tensor, testing=True)
    action = action.item()  # Convert the action tensor to a scalar value

    # Take the action in the environment
    next_state, reward, done, info = env.step(action=action, track=True, t=i, trajectory=trajectory)
    energy = info['energy']

    if done:
        break

    # Update the current state
    state = next_state

print('done!')
print(info['peak'])
