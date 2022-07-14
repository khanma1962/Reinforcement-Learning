

import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS =  10 #10000
EPS = 0.1
BANDIT_PROBS = [0.99, 0.2, 0.75, 0.5, 0.1, 0.75]


class Bandit:
    def __init__(self, prob):
        self.prob = prob
        self.prob_estimate = 0
        self.N = 0

    def pull(self):
        check = np.random.random() < self.prob
        # print(f"check is {check}")
        return check
    
    def update(self, x):
        self.N += 1
        #use the formula here
        self.prob_estimate = ((self.N -1 ) * self.prob + x)/self.N


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBS]
    
    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_j = np.argmax([b.prob for b in bandits])
    print(f"optimal j is {optimal_j}")

    for i in range(NUM_TRIALS):

        #check for EPS
        if np.random.random() < EPS:
            # print(f"Exploring")
            num_times_explored += 1
            j = np.random.randint(len(BANDIT_PROBS))
        else:
            # print("Exploiting")
            num_times_exploited += 1
            j = np.argmax([b.prob for b in bandits])

        if j == optimal_j:
            num_optimal += 1

        # pull the arm of bandit with largest sample
        x = bandits[j].pull()

        #update rewards log
        rewards[i] = x

        #update the distribution of bandit
        bandits[j].update(x)

    for b in bandits:
        print(f"mean estimate is {b.prob_estimate}")
 
    # print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num_times_explored:", num_times_explored)
    print("num_times_exploited:", num_times_exploited)
    print("num times selected optimal bandit:", num_optimal)

    # plot the results
    print(f"rewards are {rewards}")
    cumulative_rewards = np.cumsum(rewards)
    print(f"cumulative reward is {cumulative_rewards}")
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    print(f"win_rate is {win_rates}")
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBS))
    plt.show()

if __name__ == "__main__":
    experiment()


