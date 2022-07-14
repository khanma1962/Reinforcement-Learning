
import matplotlib.pyplot as plt
import numpy as np


NUM_TRIALS = 10000 #10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
  def __init__(self, p):
    # p: the win rate
    self.p = p
    self.p_estimate = 20# TODO
    self.N = 1 # TODO

  def pull(self):
    # draw a 1 with probability p
    rand_num = np.random.random()
    print(f"rand_num is {rand_num}, self.p is {self.p}")
    bool_val =  rand_num < self.p
    print(f"bool_val is {bool_val}")
    return bool_val

  def update(self, x):
    self.N += 1# TODO
    self.p_estimate = ((self.N -1) * self.p + x)/ self.N # TODO
    print(f"original p is {self.p} and the new p_estimate is {self.p_estimate}")


def experiment():
  bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

  rewards = np.zeros(NUM_TRIALS)
  for i in range(NUM_TRIALS):
    # use optimistic initial values to select the next bandit
    j = np.argmax([b.p_estimate for b in bandits])# TODO
    print(f"j is {j}")

    # pull the arm for the bandit with the largest sample
    x = bandits[j].pull()

    # update rewards log
    rewards[i] = x
    print(f"rewards is {rewards}")

    # update the distribution for the bandit whose arm we just pulled
    bandits[j].update(x)


  # print mean estimates for each bandit
  for b in bandits:
    print("mean estimate:", b.p_estimate)

  # print total reward
  print("total reward earned:", rewards.sum())
  print("overall win rate:", rewards.sum() / NUM_TRIALS)
  print("num times selected each bandit:", [b.N for b in bandits])

  # plot the results
  cumulative_rewards = np.cumsum(rewards)
  win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
  plt.ylim([0, 1])
  plt.plot(win_rates)
  plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
  plt.show()

if __name__ == "__main__":
  experiment()
