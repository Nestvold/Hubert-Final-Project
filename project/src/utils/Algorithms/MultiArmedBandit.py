from matplotlib.pyplot import plot, show, legend, figure, title, xlabel, ylabel
from numpy import zeros, argmax, where, divide
from random import uniform, randint


class MultiArmedBandit:
    """
    A class representing the Multi-Armed Bandit problem

    Attributes:
    - n_arms (int): the number of arms in the bandit problem
    - win_rates (int): the true win rates for each arm in the bandit problem
    - n_pulls (ndarray): the number of times each arm has been pulled
    - cumulative_rewards (ndarray): the cumulative rewards obtained from each arm
    """
    def __init__(self, n_arms: int, win_rates: int):
        self.n_arms = n_arms
        self.win_rates = win_rates
        self.n_pulls = zeros(shape=n_arms)
        self.cumulative_rewards = zeros(shape=n_arms)
        self.log = [[] for _ in range(n_arms)]

    def choose_arm(self, epsilon: float = 0.2):
        """
        Randomly chooses an arm to pull, with probability epsilon of choosing a random arm,
        and probability (1-epsilon) of choosing the arm with the highest estimated win rate.

        Parameters
        ----------
        - epsilon (float): the probability of choosing a random arm

        Return
        ------
        - arm (int): the index of the arm to pull
        """
        if uniform(0, 1) < epsilon:
            return randint(0, self.n_arms - 1)
        win_rates = where(self.n_pulls > 0, divide(self.cumulative_rewards, self.n_pulls), 0)
        return argmax(win_rates)

    def update(self, arm, reward):
        """
        Updates the number of pulls and cumulative rewards for a given arm based on the
        reward obtained.

        Parameters
        ----------
        - arm (int): the index of the arm that was pulled
        - reward (float): the reward obtained from pulling the arm
        """
        self.n_pulls[arm] += 1
        self.cumulative_rewards[arm] += reward

        for i in range(self.n_arms):
            self.log[i].append(self.cumulative_rewards[i])

    def plot_log(self):
        figure()
        title('Cumulative Rewards per Room')
        xlabel('Time Step')
        ylabel('Cumulative Reward')

        for arm in range(self.n_arms):
            plot(self.log[arm], label=f'Room {arm+1}')

        legend()
        show()
