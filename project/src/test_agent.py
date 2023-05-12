from utils import ENVIRONMENTS

from matplotlib.pyplot import title, savefig, figure, close
from numpy import zeros, mean
from imageio.v2 import imread
from imageio import mimsave
from seaborn import heatmap
from tqdm import tqdm
import os


def welcome_prompt():
    print('Welcome to Hubert\'s Consciousness')
    path = input("Path to the folder ../resources/")
    n_episodes = int(input("Number of episodes: "))
    return path, n_episodes


def create_gif(self, agent: list[list, list], color_bar: bool = False):
    colors = ['#00d400', '#FFFFFF', '#000000', '#2a7fff', '#f77979', '#FFA500']

    def draw_frame(time_step, t, y, x, e, fans):
        template = self.grid.copy()
        template[y, x] = 5

        for s, x_s in fans.items():
            for j in x_s:
                template[s, j] = 4

        figure(figsize=(10, 10))
        fig = heatmap(template, cmap=colors, cbar=color_bar)
        title(f'{self.name} \nTime: {t + 1} - energy: {e}', size=30)

        for _, spine in fig.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)

        savefig(f'gif/img_{time_step}.png')
        close()

    y_values, x_values, e_values, fans, t_values = zip(*agent)

    for t in tqdm(range(len(x_values)), desc='Creating plots'):
        draw_frame(t, t_values[t], y_values[t], x_values[t], e_values[t], fans[t])

    frames = []

    for t in tqdm(range(len(x_values)), desc='Creating GIP'):
        image = imread(f'gif/img_{t}.png')
        frames.append(image)

    mimsave(uri=f'Levels/gifs/{self.name}.gif',
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

from random import randint
def agent(x): return randint(0, 2)

if __name__ == '__main__':
    path, n_episodes = welcome_prompt()
    run = {}

    environments = ENVIRONMENTS(folder_path=f'resources/{path}/').list_of_envs

    for environment in environments:
        env = environment
        run[f'{env.name}'] = {}

        print(f'Environment: {env.name:<18} Shape: {env.grid.shape} MM: {env.MM is not None:>2} Fans: {env.fans is not None:>2}', end=' ')

        best_trajectory = []
        best_reward = 0
        best_time = 0
        peaks = zeros(shape=n_episodes)

        for episode in range(n_episodes):
            energy = 0
            state = env.reset()

            trajectory = []
            total_reward = 0

            i = 0
            while energy >= 0:
                action = agent(state)
                next_state, reward, done, info = env.step(action=action, track=True, t=i, trajectory=trajectory)
                energy = info['energy']  # TODO: Change to energy and not energy consumption
                state = next_state
                total_reward += reward

                if done:
                    break

            peaks[episode] = info['peak']

            if not best_trajectory:
                best_trajectory = trajectory
                best_reward = total_reward
                best_time = i
            elif total_reward > best_reward:
                best_trajectory = trajectory
                best_reward = total_reward
                best_time = i

        print(f'Peak (avg): {mean(peaks[episode]):>2}')

        run[f'{env.name}']['trajectory'] = best_trajectory
        run[f'{env.name}']['reward'] = best_reward
        run[f'{env.name}']['time'] = best_time

    print('Done!')
    print('Would you like to create a gif of the following environments:')

    for i, environment in enumerate(environments):
        print(f'{i:>2}: {environment.name}')

    answer = input('Y (yes) / N (no): ')
    print('Write the index (or indexes) of the environments you would like. (separate by space)')
    indexes = input('Index(es): ').split(' ')
    indexes = [int(i) for i in indexes]
    print('Starting to create GIF\'s This can take some time, take a coffee, or take down a shell-company like Fredriksens')
    for index in indexes:
        env = environments[index]
        create_gif(agent=run[f'{env.name}']['trajectory'])
