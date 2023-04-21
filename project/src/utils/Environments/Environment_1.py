from .utils import BaseEnv

from numpy import array, zeros, ndarray
from numpy.random import uniform


class Environment_1(BaseEnv):
    def __init__(self, name: str, grid: ndarray, enemies: set, pMM: float, project_path: str, data: dict = None):
        super().__init__(name, grid, project_path, data)
        self.enemies = enemies
        self.pMM = pMM

        # State
        self.y: int = 30
        self.x: int = 1

    def search(self, y: int, x: int):
        grid = zeros(shape=len(self.action_mapping.keys()), dtype=float)
        for action in self.action_mapping.keys():
            dy, dx = self.action_mapping[action]
            new_y, new_x = y + dy, x + dx

            if self.on_grid(new_y, new_x) and self.can_go(new_y, new_x):
                grid[action] = 0
            else:
                grid[action] = float('-inf')

        return array(grid)

    def step(self, action: int, track: bool = False, t: int = None, trajectory: list = None) -> tuple[tuple[int, int], float, bool]:
        done = False
        reward = -1
        dy, dx = self.action_mapping[action]

        if action in [0, 2]:
            if self.can_go(self.y, self.x + dx):
                self.x += dx

                if track:
                    trajectory.append((self.y, self.x, t))

                if self.seen():
                    self.y, self.x = 30, 1
                    return (self.y, self.x), reward, done

                if not self.on_solid_grounds():
                    self.y += 1

                    if track:
                        trajectory.append((self.y, self.x, t))

                    if self.seen():
                        self.y, self.x = 30, 1
                        return (self.y, self.x), reward, done

                    if self.can_go(self.y, self.x + dx):
                        self.x += dx

                        if track:
                            trajectory.append((self.y, self.x, t))

                        if self.seen():
                            self.y, self.x = 30, 1
                            return (self.y, self.x), reward, done

                        if not self.on_solid_grounds():
                            self.y += 1

                            if track:
                                trajectory.append((self.y, self.x, t))

                            if self.seen():
                                self.y, self.x = 30, 1
                                return (self.y, self.x), reward, done
        else:
            reward -= 4
            if self.on_solid_grounds():
                if self.can_go(self.y + dy, self.x):
                    self.y += dy

                    if track:
                        trajectory.append((self.y, self.x, t))

                    if self.seen():
                        self.y, self.x = 30, 1
                        return (self.y, self.x), reward, done

                    if self.can_go(self.y + dy, self.x):
                        self.y += dy

                        if track:
                            trajectory.append((self.y, self.x, t))

                        if self.seen():
                            self.y, self.x = 30, 1
                            return (self.y, self.x), reward, done
            else:
                self.y += 1

                if track:
                    trajectory.append((self.y, self.x, t))

                if self.seen():
                    self.y, self.x = 30, 1
                    return (self.y, self.x), reward, done

                if not self.on_solid_grounds():
                    self.y += 1

                    if track:
                        trajectory.append((self.y, self.x, t))

                    if self.seen():
                        self.y, self.x = 30, 1
                        return (self.y, self.x), reward, done

        if self.in_end_state():
            done = True
            reward = 0

        return (self.y, self.x), reward, done

    def seen(self):
        return (self.y, self.x) in self.enemies and uniform() < self.pMM

    def on_solid_grounds(self):
        return self.grid[self.y + 1, self.x] in self.solids

    def encountered_enemy(self, y: int, x: int) -> bool:
        return (y, x) in self.enemies and self.pMM > uniform()

    def in_end_state(self) -> bool:
        return self.y == 1 and self.x == 30

    def reset(self) -> tuple[int, int]:
        self.y = 30
        self.x = 1
        return self.y, self.x
