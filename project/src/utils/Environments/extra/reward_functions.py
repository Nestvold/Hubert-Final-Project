def reward_func_base(self, prev_pos):
    reward, done = 0, False

    # If encountering MM
    if self.busted():
        return -1.0, True

    # If reached the ceiling
    if self.in_end_state():
        return 1.0, True

    # Penalize getting seen by fans
    self.energy -= self.seen()

    if self.energy < 0:
        return -1.0, True

    # Encourage height
    if self.y < self.best_y:
        change = self.best_y - self.y
        self.energy += change * 15
        self.best_y = self.y
        self.peak = self.best_y / (self.grid.shape[0] - 2) if self.best_y > 1 else 0
        reward += 1

    # Encourage exploration
    if (surroundings := hash(str(list(self.surroundings.flatten())))) not in self.prev_states:
        self.prev_states.add(surroundings)
        reward += 0.1

    if self.x == prev_pos[1] and self.y >= prev_pos[0]:
        reward -= 0.1

    return reward, done

def reward_func_height_focus(self, prev_pos):
    reward, done = 0, False

    # If encountering MM
    if self.busted():
        return -1.0, True

    # If reached the ceiling
    if self.in_end_state():
        return 1.0, True

    # Penalize getting seen by fans
    self.energy -= self.seen()

    # Penalize running out of energy
    if self.energy < 0:
        return -1.0, True

    # Penalize wasted actions
    if (self.y, self.x) == prev_pos:
        reward -= 0.2

    # Penalize not advancing
    if self.y >= self.best_y:
        reward -= 0.1

    # Encourage reaching new heights
    if self.y < self.best_y:
        change = self.best_y - self.y
        self.energy += change * 15
        self.best_y = self.y
        self.peak = self.best_y / (self.grid.shape[0] - 2) if self.best_y > 1 else 0
        reward = 1.0

    return reward, done

def reward_func_height_and_exploration(self, prev_pos):
    reward, done = 0, False

    # If encountering MM
    if self.busted():
        return -1.0, True

    # If reached the ceiling
    if self.in_end_state():
        return 1.0, True

    # Penalize getting seen by fans
    self.energy -= self.seen()

    # Penalize running out of energy
    if self.energy < 0:
        return -1.0, True

    # Penalize wasted actions
    if (self.y, self.x) == prev_pos:
        reward -= 0.2

    # Penalize not advancing
    if self.y >= self.best_y:
        reward -= 0.1

    # Encourage exploration
    if (surroundings := hash(str(list(self.surroundings.flatten())))) not in self.prev_states:
        self.prev_states.add(surroundings)
        reward += 0.1

    # Encourage reaching new heights
    if self.y < self.best_y:
        change = self.best_y - self.y
        self.energy += change * 15
        self.best_y = self.y
        self.peak = self.best_y / (self.grid.shape[0] - 2) if self.best_y > 1 else 0
        reward = 1.0

    return reward, done

def reward_func_no_negative(self, prev_pos):
    reward, done = 0, False

    # If encountering MM
    if self.busted():
        return -1.0, True

    # If reached the ceiling
    if self.in_end_state():
        return 1.0, True

    # Penalize getting seen by fans
    self.energy -= self.seen()

    # Penalize running out of energy
    if self.energy < 0:
        return -1.0, True

    # Encourage reaching new heights
    if self.y < self.best_y:
        change = self.best_y - self.y
        self.energy += change * 15
        self.best_y = self.y
        self.peak = self.best_y / (self.grid.shape[0] - 2) if self.best_y > 1 else 0
        reward = 1.0 * change

    return reward, done

def high_rise(self, prev_pos):
    reward, done = -0.1, False

    # If encountering MM
    if self.busted():
        return -1.0, True

    # If reached the ceiling
    if self.in_end_state():
        return 1.0, True

    # Penalize getting seen by fans
    self.energy -= self.seen()

    # Penalize running out of energy
    if self.energy < 0:
        return -1.0, True

    # Encourage reaching new heights
    if self.y < self.best_y:
        change = self.best_y - self.y
        self.energy += change * 15
        self.best_y = self.y
        self.peak = self.best_y / (self.grid.shape[0] - 2) if self.best_y > 1 else 0
        reward = 0

    return reward, done
