from project.src.utils.Environments import Environment_6 as Environment
from project.src.utils import read_tsv_file, Agent

from gym.wrappers import FrameStack
import torch


def main():
    # Loading grid from file and environment
    grid = read_tsv_file('../resources/map_4.dat', enemies={}, start=(46, 1), end=(1, 46))
    env = Environment(name="Level 5", grid=grid, project_path='Levels')
    env = FrameStack(env=env, num_stack=3)

    # Setting device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initiating agent (Hubert)
    agent = Agent(num_actions=env.action_space.n, device=device)

    # Batch size
    batch_size = 32

    if device.type == 'cpu':
        episodes = 100
    else:
        episodes = 1_000

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, *_ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size)
            state = next_state

        agent.decrease_epsilon()


if __name__ == '__main__':
    main()
