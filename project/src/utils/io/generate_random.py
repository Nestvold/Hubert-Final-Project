import numpy as np
import os


def create_environment(n):
    env = np.ones((n, n), dtype=np.int32)  # Initialize the array with all cells set to 1 (air)
    env[[0, -1], :] = 2  # Set the top and bottom rows to 2 (solid)
    env[:, [0, -1]] = 2  # Set the left and right columns to 2 (solid)
    env[2:-2, 1:-1] = np.random.choice([1, 3], size=(n-4, n-2))  # Randomize the inner cells
    return env


def save_environment_as_tsv(environment, folder_path, file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    np.savetxt(file_path, environment, delimiter='\t', fmt='%d')

# Example usage
n = 48
environment = create_environment(n)

folder_path = 'project/resources/training_maps'
for i in range(22, 100):
    save_environment_as_tsv(environment, f'{folder_path}/map_{i}', f'random_env.dat')
