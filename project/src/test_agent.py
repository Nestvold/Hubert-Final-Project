from matplotlib.pyplot import title, savefig, figure, close
from gym.wrappers import FrameStack
from numpy import zeros, mean, std
from utils import ENVIRONMENTS
from train_agent import Agent
from imageio.v2 import imread
from imageio import mimsave
from seaborn import heatmap
from tqdm import tqdm
from numpy import array
import torch
from time import sleep
import os


def welcome_prompt():
    print('Welcome to Hubert\'s Consciousness')
    path = input("Path to the folder ../resources/")
    n_episodes = max(int(input("Number of episodes: ")), 1)
    mr_agent = int(input("But last but not least which agent do you want: "
                         "\n [0] Hubert"
                         "\n [1] Audun"
                         "\n [2] Morten"
                         "\n [3] TROELS"
                         "\nI choose agent (index): "))
    try:
        cuda_nr = int(input("If you have multiple cudas, which one do you want us to access (0 is default):"))
    except ValueError:
        cuda_nr = 0
    x = 2
    if mr_agent == 1:
        x = 5
    elif mr_agent == 2:
        x = 10
    elif mr_agent == 3:
        x = 20

    return path, n_episodes, x, cuda_nr


def create_gif(env, agent: list[list, list], color_bar: bool = False):
    colors = ['#FFA500', '#FFFFFF', '#000000', '#2a7fff']
    max_dimension = max(env.grid.shape[0], env.grid.shape[1])
    scale_factor = 10 / max_dimension
    width = env.grid.shape[1] * scale_factor
    height = env.grid.shape[0] * scale_factor

    r = 0
    i = 0

    def draw_frame(time_step, t, y, x, e, MM_pos, fan_pos):
        nonlocal r
        nonlocal i

        if t == i:
            i += 1
            r += e

        template = env.grid.copy()
        template[y, x] = 0

        if fan_pos:
            for fan in fan_pos:
                pos = tuple(fan)
                template[pos] = 4

        if MM_pos:
            for M in MM_pos:
                pos = tuple(M)
                template[pos] = 5

        figure(figsize=(width, height))
        fig = heatmap(template, cmap=colors, cbar=color_bar)
        title(f'{env.name} \nTime: {t + 1} - energy: {r}', size=30)

        for _, spine in fig.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)

        savefig(f'gif/img_{time_step}.png')
        close()

    y_values, x_values, MM, fans, energy, times = zip(*agent)

    if fans[0]:
        colors.append('#79e2f7')

    if MM[0]:
        colors.append('#7630e6')

    for t in tqdm(range(len(x_values)), desc='Creating plots'):
        draw_frame(t, times[t], y_values[t], x_values[t], energy[t], MM[t], fans[t])

    frames = []

    for t in tqdm(range(len(x_values)), desc='Creating GIF'):
        image = imread(f'gif/img_{t}.png')
        frames.append(image)

    mimsave(uri=f'Levels/gifs/{env.name}_{r}.gif',
            ims=frames,
            fps=10
            )

    # Delete plots
    folder_path = "gif"

    # Get all the file names in the folder
    files = os.listdir(folder_path)

    # Loop through the files and delete them
    for file_name in tqdm(files, desc='Deleting files'):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    print('[GIF CREATED]')


if __name__ == '__main__':
    path, n_episodes, n_stacks, cuda_nr = welcome_prompt()

    # Load agent
    device = torch.device(f"cuda:{cuda_nr}" if torch.cuda.is_available() else "cpu")
    print(f'Loading model on: {device} ...', end=' ')
    agent = torch.load(
        f'C:\\Users\\Fredrik\\Documents\\GitHub\\Hubert-Final-Project\\project\\src\\models\\base_plus\\agent_{n_stacks}.pt').to(
        device)
    agent.training = False
    print('done.')

    run = {}

    envs = ENVIRONMENTS(folder_path=f'resources/{path}/').list_of_envs
    environments = [FrameStack(env=env, num_stack=n_stacks) for env in envs]

    for environment in environments:
        env = environment
        run[f'{env.name}'] = {}


        best_trajectory = []
        best_reward = 0
        best_time = 0
        peaks = zeros(shape=n_episodes)

        for episode in tqdm(range(n_episodes), desc='Episodes'):
            energy = 0
            state = env.reset()
            trajectory = []
            time = 0
            best_peak = 1000

            while True:
                # Convert the state to a PyTorch tensor
                state_array = array(state)
                state_tensor = torch.from_numpy(state_array).float()
                state_tensor = state_tensor.to(device)

                # Pass the state through the agent to obtain the action
                action, _, _, _ = agent.get_action_and_value(state_tensor, testing=True)
                action = action.item()  # Convert the action tensor to a scalar value

                # Take the action in the environment
                next_state, reward, done, info = env.step(action=action)

                path = info['trajectory']
                trajectory.extend([(*item, time) for item in path])

                if done:
                    break

                # Update the current state
                time += 1
                state = next_state

            peaks[episode] = info['peak']

            if not best_trajectory or info['peak'] < best_peak:
                best_trajectory = trajectory
                best_peak = info['peak']

        print(f'Environment: {env.name:<22} - Shape: {env.grid.shape} - MM: {len(env.MM) if env.MM else 0:>1} - Fans: {len(env.fans)  if env.MM else 0:>1} - Avg: {round(mean(peaks), 4)} - Std: {std(peaks)}')

        run[f'{env.name}']['trajectory'] = best_trajectory
        run[f'{env.name}']['time'] = best_time

        sleep(0.5)

    print('Done!')
    print('Would you like to create a gif of the following environments:')

    for i, environment in enumerate(environments):
        print(f'{i:>2}: {environment.name}')

    answer = input('Y (yes) / N (no): ')

    if answer.strip().lower() == 'n':
        exit()

    print('Write the index (or indexes) of the environments you would like. (separate by space)')

    indexes = input('Index(es): ').split(' ')
    indexes = [int(i) for i in indexes]

    print('Starting to create GIF\'s This can take some time, take a coffee, or take down a shell-shit-company like Fredriksens')

    for index in indexes:
        env = environments[index]
        print(f'[CREATING {env.name.upper().replace("DATA", "")}]')
        sleep(0.1)
        create_gif(env, agent=run[f'{env.name}']['trajectory'])
        sleep(2)
