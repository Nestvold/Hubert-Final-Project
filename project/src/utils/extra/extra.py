from numpy.random import choice as rand_choice
from random import choice, uniform, randint
from numpy import where, amax, zeros


def Reward(SA_1, Sa_2, R, alpha=0.3, gamma=1.0):
    return SA_1 + alpha * (R + (gamma * max(Sa_2) - SA_1))


def Walking_MM(enemy):
    pos = enemy
    i, j = pos[0], pos[1]

    if j == 2:
        j += 1
    elif j == 29:
        j -= 1
    else:
        j += choice([-1, 1])
    return i, j


def Action(state, epsilon=0.2):
    if uniform(0, 1) < epsilon:
        action = randint(0, 2)
    else:
        if not state.any():
            action = randint(0, 2)
        else:
            options = where(state == amax(state))[0]
            action = rand_choice(options)
    return action


def Scan_Fans(grid, pos, fan_base=set()):
    scan = zeros((9, 9))
    limit = grid.shape[0]

    for y, i_1 in enumerate(range(pos[0] - 4, pos[0] + 5)):
        for x, i_2, in enumerate(range(pos[1] - 4, pos[1] + 5)):
            if 0 <= i_1 < limit and 0 <= i_2 < limit and (y, x) in fan_base:
                scan[y, x] = 4

    return tuple(scan.flatten())


def Scan(grid, pos):
    scan = zeros((9, 9))
    limit = grid.shape[0]

    for y, i_1 in enumerate(range(pos[0] - 4, pos[0] + 5)):
        for x, i_2, in enumerate(range(pos[1] - 4, pos[1] + 5)):
            if 0 <= i_1 < limit and 0 <= i_2 < limit:
                scan[y, x] = grid[i_1, i_2]

    return tuple(scan.flatten())
