from .SARSA import SARSA

# Other modules
from numpy import array
from collections import deque
from tqdm import tqdm


class SARSA3(SARSA):
    def __init__(self, environment, epsilon: float = 0.8, alpha: float = 0.1, gamma: float = 0.999):
        super().__init__(environment, epsilon, alpha, gamma)
        self.n = 3
        self.state_action_buffer = deque(maxlen=self.n)
        self.reward_buffer = deque(maxlen=self.n)

    def train(self, num_episodes: int = 1_000):
        for episode in tqdm(range(num_episodes), desc='Episode'):
            state = self.env.reset()
            state = hash(str(list(state)))
            action = self.epsilon_greedy(state)

            self.state_action_buffer.clear()
            self.reward_buffer.clear()

            for _ in range(2_000_000):
                next_state, reward, done, *_ = self.env.step(action)
                next_state = hash(str(list(next_state)))
                next_action = self.epsilon_greedy(next_state)

                self.state_action_buffer.append((state, action))
                self.reward_buffer.append(reward)

                if len(self.state_action_buffer) == self.n or done:
                    G = sum([self.gamma ** i * r for i, r in enumerate(self.reward_buffer)])
                    if not done:
                        G += self.gamma ** self.n * self.Q[next_state][next_action]

                    prev_state, prev_action = self.state_action_buffer[0]
                    self.Q[prev_state][prev_action] += self.alpha * (G - self.Q[prev_state][prev_action])

                    if done and len(self.state_action_buffer) > 1:
                        for i in range(1, len(self.state_action_buffer)):
                            G -= self.reward_buffer.popleft()
                            G /= self.gamma
                            G += self.gamma ** (self.n - 1) * reward

                            prev_state, prev_action = self.state_action_buffer.popleft()
                            self.Q[prev_state][prev_action] += self.alpha * (G - self.Q[prev_state][prev_action])

                state, action = next_state, next_action

                if done:
                    break

            if self.epsilon != 0:
                self._decaying_epsilon(), self._decaying_lr()
