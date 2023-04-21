from numpy.random import uniform, choice
from numpy import argmax, zeros
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
        self.epsilon_min = 0.1
        self.alpha_min = 0.1

    def train(self, num_episodes: int = 1_000):
        # Repeat for each episode
        for episode in tqdm(range(num_episodes), desc='Episode'):
            # Initialize state
            state = self.env.reset()

            # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
            action = self.epsilon_greedy(state)

            # Repeat for each step of the episode
            for _ in range(4_000_000):
                # Take action A, observe R and S'
                next_state, reward, done = self.env.step(action)

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

    def get_optimal_trajectory(self) -> tuple[list, int]:
        trajectory = []
        state = self.env.reset()
        trajectory.append((30, 1, self.env.walking_fans, 0))
        t = -1

        for _ in range(100):
            t += 1
            # Get best action from current state
            action = choice([0, 1, 2])  # argmax(self.Q[state])
            # Take the step and look around
            next_state, _, done = self.env.step(action, track=True, t=t, trajectory=trajectory)
            # Update state
            state = next_state

            if done:
                break

        return trajectory, len(trajectory)
