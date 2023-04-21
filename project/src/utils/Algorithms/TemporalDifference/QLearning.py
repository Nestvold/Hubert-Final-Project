# Other modules
from numpy import max, argmax, where, zeros
from numpy.random import choice, uniform
from collections import defaultdict
from tqdm import tqdm


class QLearning:
    def __init__(self, environment, epsilon: float = 0.1, alpha: float = 0.1, gamma: float = 0.99):
        self.env = environment
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = [0, 1, 2, 3]
        # Initialize Q(s, a) to 0 for all s and a using dict
        self.Q = defaultdict(lambda: zeros(shape=3))
        self.epsilon_min = 0.1
        self.alpha_min = 0.1

    def train(self, num_episodes: int = 10_000):
        # Loop through episodes
        for episode in tqdm(range(num_episodes), desc='Episode'):
            # Initialize state
            state = self.env.reset()

            # Repeat for each step of the episode
            while True:
                # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
                action = self.epsilon_greedy(state)

                # Take action and observe next state and reward
                next_state, reward, done = self.env.step(action)

                # Update Q-value using Q-learning update rule
                self.Q[state][action] += self.alpha * (reward + self.gamma * max(self.Q[next_state]) - self.Q[state][action])

                state = next_state

                if done:
                    break

            self._decaying_epsilon(), self._decaying_lr()

    def epsilon_greedy(self, state):
        if uniform() < self.epsilon:
            return choice(self.actions)
        return argmax(self.Q[state])

    def _decaying_epsilon(self, value: float = 0.9):
        if self.epsilon <= self.epsilon_min:
            self.epsilon = self.epsilon_min
        else:
            self.epsilon *= value

    def _decaying_lr(self, value: float = 0.9):
        if self.alpha <= self.alpha_min:
            self.alpha = self.alpha_min
        else:
            self.alpha *= value

    def get_optimal_trajectory(self, start_state) -> tuple[list, int]:
        trajectory = []
        state = start_state
        while True:
            # Add current position to trajectory
            trajectory.append(state)
            # Get best action from current state
            action = argmax(self.Q[state])
            # Take the step and look around
            next_state, _, done = self.env.step(action)
            # Update state
            state = next_state

            if done:
                break

        return trajectory, len(trajectory)
