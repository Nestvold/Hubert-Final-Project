from .BaseEnv import BaseEnv

from numpy import ndarray
from numpy.random import uniform


class Environment(BaseEnv):
    def __init__(self, name: str, grid: ndarray, enemies: set, pMM: float, project_path: str, data: dict = None):
        super().__init__(name, grid, project_path, data)
        self.enemies = enemies
        self.pMM = pMM

        # Action mapping
        self.action_mapping = {0: (0, -1), 1: (1, 0), 2: (0, 1)}

        # State
        self.y: int = 30
        self.x: int = 1

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        reward = -1
        done = False

        dy, dx = self.action_mapping[action]
        new_y, new_x = self.y + dy, self.x + dx

        if self.encountered_enemy(new_y, new_x):
            self.y = 30
            self.x = 1

        if self.in_end_state(new_y, new_x):
            done = True
            reward = 0
            self.y, self.x = new_y, new_x

        return (new_y, new_x), reward, done

    def encountered_enemy(self, y: int, x: int) -> bool:
        return (y, x) in self.enemies and self.pMM > uniform()

    @staticmethod
    def in_end_state(y, x) -> bool:
        return y == 1 and x == 30

    def reset(self) -> tuple[int, int]:
        self.y = 30
        self.x = 1
        return self.y, self.x
