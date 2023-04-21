from utils.Algorithms import MultiArmedBandit
from random import uniform


win_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.15, 0.25, 0.35, 0.45, 0.55]
bandit = MultiArmedBandit(len(win_rates), win_rates)
n_steps = 10_000


for i in range(n_steps):
    arm = bandit.choose_arm(epsilon=0.2)
    reward = 1 if uniform(0, 1) < win_rates[arm] else 0
    bandit.update(arm, reward)

print(bandit.n_pulls)
print(bandit.cumulative_rewards)

bandit.plot_log()
