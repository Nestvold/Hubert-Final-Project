from .utils import BaseEnv

from numpy import array, zeros, ndarray
from numpy.random import uniform, choice


class Environment_2(BaseEnv):
    def __init__(self, name: str, grid: ndarray, enemies: set, pMM: float = 1.0, project_path: str='', data: dict = None):
        super().__init__(name, grid, project_path, data)
        self.enemies = enemies
        self.enemy = next(iter(enemies))
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

    def update_enemies(self):
        move = choice([-1, 0, 1])
        x = next(iter(self.enemies))[1]

        if move == -1:
            self.enemies = set([(20, min(29, x + move))])
            self.enemy = max(29, x + move)
        elif move == 1:
            self.enemies = set([(20, max(2, x + move))])
            self.enemy = min(29, x + move)

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        done = False
        reward = -1
        self.update_enemies()
        dy, dx = self.action_mapping[action]

        if action in [0, 2]:
            if self.can_go(self.y, self.x + dx):
                self.x += dx

                if self.seen():
                    self.y, self.x = 30, 1
                    return (self.y, self.x, self.enemy), reward, done

                if not self.on_solid_grounds():
                    self.y += 1

                    if self.seen():
                        self.y, self.x = 30, 1
                        return (self.y, self.x, self.enemy), reward, done

                    if self.can_go(self.y, self.x + dx):
                        self.x += dx

                        if self.seen():
                            self.y, self.x = 30, 1
                            return (self.y, self.x, self.enemy), reward, done

                        if not self.on_solid_grounds():
                            self.y += 1

                            if self.seen():
                                self.y, self.x = 30, 1
                                return (self.y, self.x, self.enemy), reward, done
        else:
            if self.on_solid_grounds():
                if self.can_go(self.y + dy, self.x):
                    self.y += dy

                    if self.seen():
                        self.y, self.x = 30, 1
                        return (self.y, self.x), reward, done

                    if self.can_go(self.y + dy, self.x):
                        self.y += dy

                        if self.seen():
                            self.y, self.x = 30, 1
                            return (self.y, self.x, self.enemy), reward, done
            else:
                self.y += 1
                if not self.on_solid_grounds():
                    self.y += 1

                    if self.seen():
                        self.y, self.x = 30, 1
                        return (self.y, self.x, self.enemy), reward, done

        if self.in_end_state():
            done = True
            reward = 0

        return (self.y, self.x, self.enemy), reward, done

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
        return self.y, self.x, self.enemy
