from numpy.random import uniform, choice
from numpy import argmax, zeros, array
from collections import defaultdict
from tqdm import tqdm


class SARSA:
    def __init__(self, environment, epsilon: float = 0.8, alpha: float = 0.1, gamma: float = 0.999):
        self.env = environment
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = [0, 1, 2]

        # Initialize Q(s, a) to 0 for all s and a using dict
        self.Q = defaultdict(lambda: zeros(shape=3))
        self.epsilon_min = 0.005
        self.alpha_min = 0.1

    def train(self, num_episodes: int = 1_000):
        # Repeat for each episode
        for episode in tqdm(range(num_episodes), desc='Episode'):
            # Initialize state

            if self.env.name == 'Level 5':
                state, _ = self.env.reset()
                state = hash(str(list(state)))
            else:
                state = self.env.reset()

            # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
            action = self.epsilon_greedy(state)

            trajectory = [(self.env.y, self.env.x, 0, 0)]

            # Repeat for each step of the episode
            while True:
                # Take action A, observe R and S'
                next_state, reward, done, *_ = self.env.step(action)

                if self.env.name == 'Level 5':
                    next_state = hash(str(list(next_state)))

                # Choose A' from S' using policy derived from Q (e.g., epsilon-greedy)
                next_action = self.epsilon_greedy(next_state)

                # Update Q(S, A) <- Q(S, A) + alpha[R + gamma * Q(S', A') - Q(S, A)]
                self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])

                # S <- S'; A <- A'
                state, action = next_state, next_action

                # If S is terminal, end episode
                if done:
                    break

            if self.epsilon != 0:
                self._decaying_epsilon(), self._decaying_lr()

    def epsilon_greedy(self, state):
        if uniform() < self.epsilon:
            return choice(self.actions)
        return argmax(self.Q[state])

    def _decaying_epsilon(self, value: float = 0.9999):
        if self.epsilon <= self.epsilon_min:
            self.epsilon = self.epsilon_min
        else:
            self.epsilon *= value

    def _decaying_lr(self, value: float = 0.9):
        if self.alpha <= self.alpha_min:
            self.alpha = self.alpha_min
        else:
            self.alpha *= value

    def get_optimal_trajectory(self) -> tuple[list, int]:
        trajectory = []
        if self.env.name == 'Level 5':
            state, _ = self.env.reset()
        else:
            state = self.env.reset()
        # Level 2
        if self.env.name == 'Level 2':
            trajectory.append((30, 1, 0, 0))
        # Level 3
        elif self.env.name == 'Level 3':
            trajectory.append((30, 1, self.env.enemy, 0, 0))
        # Level 4
        elif self.env.name == 'Level 4':
            trajectory.append((30, 1, self.env.walking_fans, 0, 0))
        # Level 5
        elif self.env.name == "Level 5":
            trajectory.append((46, 1, 0, 0))

        t = -1
        total_reward = 0
        while True:
            t += 1

            # Hash
            state = hash(str(list(array(state).flatten())))
            # Get best action from current state
            action = argmax(self.Q[state])
            # Take the step and look around
            if self.env.name == "Level 5":
                next_state, reward, done, _, _ = self.env.step(action, track=True, t=t, trajectory=trajectory)
            else:
                next_state, reward, done = self.env.step(action, track=True, t=t, trajectory=trajectory)
            # Update state
            state = next_state
            total_reward += reward

            if done:
                break

        return trajectory, total_reward, t
