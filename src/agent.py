"""
This file is used to create the agent that interacts with the environment.
"""

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

"""
Agent specifications:
- The agent uses Parameterized Gradient ascent to learn the optimal policy.
- Since the environment is continuous, an appropriate representation of the policy is required.
- The policy is represented as a quadratic combination of the state features.
"""


class Agent:
    def __init__(self, env):
        self.env = env
        self.theta = np.zeros(5)
        self.alpha = 0.0001
        self.sigma = 1

    def set_theta(self, theta):
        self.theta = theta

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
        return exp_x / exp_x.sum(axis=0)

    def compute_action_probabilities(self, state):
        # Unpack policy parameters
        w1, w2, w3, w4, w5 = self.theta

        # Compute the linear combination of state variables
        policy = (
            w1 * state[0]
            + w2 * state[1]
            + w3 * state[2]
            + w4 * state[3]
            + w5 * state[4]
        )

        # Compute action probabilities using softmax
        action_probabilities = self.softmax(
            np.array([policy, -policy])
        )  # Assuming two actions: left and right

        return action_probabilities

    def run_episode(self):
        """
        This function is used to run an episode using the given policy.
        """
        self.env.reset()
        state = self.env.state
        total_reward = 0
        for t in range(self.env.max_time_steps):
            action_probabilities = self.compute_action_probabilities(state)
            action = np.random.choice(self.env.action_space.n, p=action_probabilities)
            state, reward, done, _, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    def learn(self):
        """
        This function is used to learn the optimal policy using the parameterized gradient ascent algorithm.
        """
        self.env.reset()
        rewards_per_trial = []
        for trials in tqdm(range(300)):
            # grad = np.zeros(np.size(self.theta))
            rewards = [self.run_episode()]
            max_reward = rewards[0]
            for episode in range(100):
                std_dev_matrix = self.sigma * np.eye(*self.theta.shape)
                new_theta = np.random.multivariate_normal(self.theta, std_dev_matrix)
                cur_theta = self.theta
                self.theta = new_theta
                new_reward = self.run_episode()
                if new_reward < max_reward:
                    self.theta = cur_theta
                elif new_reward > max_reward:
                    max_reward = new_reward
                rewards.append(max_reward)
            rewards_per_trial.append(rewards)
        return rewards_per_trial

    def plot_mean_rewards(self, rewards_per_trial):
        """
        This function is used to plot the mean rewards per trial.
        """
        mean_rewards = np.mean(rewards_per_trial, axis=0)
        plt.figure(figsize=(20, 6))
        plt.plot(mean_rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Mean rewards")
        plt.title("Mean rewards per trial")
        plt.show()

    def plot_error_bar(self, rewards_per_trial):
        """
        This function is used to plot the mean rewards per trial with error bars.
        """
        mean_rewards = np.mean(rewards_per_trial, axis=0)
        std_dev = np.std(rewards_per_trial, axis=0)
        plt.figure(figsize=(20, 6))
        plt.errorbar(range(len(mean_rewards)), mean_rewards, yerr=std_dev, fmt="o")
        plt.xlabel("Episodes")
        plt.ylabel("Mean rewards")
        plt.title("Mean rewards per trial with error bars")
        plt.show()

    def cross_entropy(self, eps):
        """
        This function is used to learn the optimal policy using the Cross-Entropy method.
        """
        self.env.reset()
        # Fix K, Keps
        K = 100
        Keps = int(K * eps)
        # Initialize theta
        theta = np.zeros(5)
        self.set_theta(theta)
        max_reward_so_far = np.mean([self.run_episode() for _ in range(10)])
        # Initialize covariance matrix
        sigma = 2 * np.eye(5)
        rewards_per_episode = [[] for i in range(10)]
        for it in tqdm(range(300)):
            rewards_per_theta = []
            for k in range(1, K + 1):
                new_theta = np.random.multivariate_normal(self.theta, sigma)
                cur_theta = self.theta
                self.set_theta(new_theta)
                rewards = []
                for episode_count in range(10):
                    rewards.append(self.run_episode())
                    rewards_per_episode[episode_count].append(rewards[-1])
                rewards_per_theta.append([np.mean(rewards), new_theta])
                if np.mean(rewards) > max_reward_so_far:
                    max_reward_so_far = np.mean(rewards)
                else:
                    self.set_theta(cur_theta)
            rewards_per_theta.sort(key=lambda x: x[0], reverse=True)
            # Pick Keps best thetas
            best_thetas = [theta for _, theta in rewards_per_theta][:Keps]
            # Find the mean and covariance of the best thetas
            mean_theta = np.mean([x[1] for x in best_thetas], axis=0)
            temp = np.sum(
                [
                    np.multiply(
                        np.subtract(theta, mean_theta),
                        np.transpose(np.subtract(theta, mean_theta)),
                    )
                    for theta in best_thetas
                ]
            )
            sigma = (1.0 / (eps + Keps)) * (eps * 2 * np.eye(5) + temp)
        mean_reward_per_episode = [np.mean(x) for x in rewards_per_episode]
        # PLot the mean rewards per episode
        plt.figure(figsize=(20, 6))
        plt.plot(mean_reward_per_episode)
        plt.xlabel("Episodes")
        plt.ylabel("Mean rewards")
        plt.title("Mean rewards per episode")
        plt.show()
        # Plot the error bars
        std_dev_per_episode = [np.std(x) for x in rewards_per_episode]
        plt.figure(figsize=(20, 6))
        plt.errorbar(
            range(len(mean_reward_per_episode)),
            mean_reward_per_episode,
            yerr=std_dev_per_episode,
            fmt="o",
        )
        plt.xlabel("Episodes")
        plt.ylabel("Mean rewards")
        plt.title("Mean rewards per episode with error bars")
        plt.show()
