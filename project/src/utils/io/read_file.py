from numpy import ndarray, array, zeros


def read_tsv_file(filename: str, enemies: set, start: tuple, end: tuple) -> ndarray:
    """
    Reads a TSV file and puts the value in an array to represent the track/hill.
    :param end:
    :param start:
    :param enemies:
    :param filename:
    :return: An array representing the values from file
    """

    with open(f'resources/{filename}') as file:
        content = file.readlines()
        y, x = len(content), len(content[0].split('\t'))
        grid = zeros((y, x), dtype=float)

        for i, line in enumerate(content):
            line = line.split('\t')
            values = array([float(x) for x in line])
            grid[i] = values

    for enemy in enemies:
        y, x = enemy
        grid[y, x] = 4

    grid[start] = 0
    grid[end] = 0

    return grid


def read_MM(filename: str) -> set:
    """
    Reads a TSV file and puts the value in an array to represent the Middle Managers.
    :param filename:
    :return: An dict representing the values from file
    """
    MM = set()
    with open(f'project_data/{filename}') as file:
        content = file.readlines()
        for line in content:
            line = line.split('\t')
            y, x = (int(z) for z in line[:2])
            MM.add((y - 1, x - 1))
    return MM


def read_walking_MM(filename: str) -> set:
    """
    Reads a TSV file and puts the value in an array to represent the walking Middle Managers.
    :param filename:
    :return: An dict representing the values from file
    """
    MM = set()
    with open(f'resources/{filename}') as file:
        content = file.readlines()
        for line in content:
            line = line.split('\t')
            y, x = (int(z) for z in line[:2])

            MM.add((y - 1, x - 1))
    return MM


def read_fans(filename: str) -> set:
    """
    Reads a TSV file and puts the value in an array to represent the fans.
    :param filename:
    :return: An dict representing the values from file
    """
    fans = {}
    with open(f'resources/{filename}') as file:
        content = file.readlines()
        for line in content:
            line = line.split('\t')
            y, x_1, x_2 = (int(z) for z in line)
            fans[y-1] = (x_1 - 1, x_2 - 1)
    return fans
