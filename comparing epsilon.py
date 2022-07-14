from __future__ import print_function, division
from builtins import range


import numpy as np
import matplotlib.pyplot as plt


class Bandit:
  def __init__(self, m):
    self.m = m
    self.mean = 0
    self.N = 0
    # print(f"m is {self.m}, self.mean is {self.mean}, and self.N is {self.N}")

  def pull(self):
    gauss_dist = np.random.rand() + self.m
    # print(f"gauss dist is {gauss_dist}")
    return gauss_dist

  def update(self, x):
    self.N += 1
    self.mean = (1 - 1.0/self.N)*self.mean + 1.0/self.N*x
    # print(f"m {self.m} has new mean of {self.mean}")


def run_experiment(m1, m2, m3, eps, N):
  bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

  data = np.empty(N)
  
  for i in range(N):
    # epsilon greedy
    p = np.random.random()
    if p < eps:
    #   print("exploring")
      j = np.random.choice(3)
    else:
    #   print('exploiting')
      j = np.argmax([b.mean for b in bandits])
    
    # print(f"j is {j}")
    x = bandits[j].pull()
    bandits[j].update(x)

    # for the plot
    data[i] = x
    # print(f"data is {data}")
  cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
#   print(f"cumulative average is {cumulative_average}")

  # plot moving average ctr
  plt.plot(cumulative_average)
  plt.plot(np.ones(N)*m1)
  plt.plot(np.ones(N)*m2)
  plt.plot(np.ones(N)*m3)
  plt.xscale('log')
  plt.show()

  for b in bandits:
    print(f"b.mean is {b.mean}")

  return cumulative_average

if __name__ == '__main__':

  num_trials = 100000 #100000
  m1, m2, m3 = 2.5, 5, 7.5
  c_1 = run_experiment(m1, m2, m3, 0.1, num_trials) #10% exploration
  c_05 = run_experiment(m1, m2, m3, 0.05, num_trials) #5% exploration
  c_01 = run_experiment(m1, m2, m3, 0.01, num_trials) #1% exploration

  # log scale plot
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(c_05, label='eps = 0.05')
  plt.plot(c_01, label='eps = 0.01')
  plt.legend()
  plt.xscale('log')
  plt.show()


  # linear plot
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(c_05, label='eps = 0.05')
  plt.plot(c_01, label='eps = 0.01')
  plt.legend()
  plt.show()

