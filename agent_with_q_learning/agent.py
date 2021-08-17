import gym
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt


class Agent():
    def __init__(self):
        self.q_table = np.zeros((5, 2))
        self.learning_rate = 0.05
        self.discount_factor = 0.95
        self.epsilon = 0.5
        self.decay_factor = 0.999
        self.average_reward_for_each_game = []

    def play(self, env, n_episodies=200):
        for episode in range(n_episodies):
            print("Episode {} of {} ".format(episode + 1, n_episodies))

            state = env.reset()
            self.epsilon *= self.decay_factor
            total_reward = 0
            endgame = False

            # print("Epsilon: ", self.epsilon)

            while not endgame:
                if self.__q_table_is_empty(state) or self.__with_probability(self.epsilon):
                    action = self.__get_random_action(env)
                else:
                    action = self.__get_highest_reward_action(state)

                new_state, reward, endgame, _ = env.step(action)
                total_reward += reward

                self.q_table[state, action] += self.__apply_q_learning(
                    state, action, reward, new_state)

                state = new_state

            # print("state {}, action {}, new_state {}, reward {}".format(
            #    state, action, new_state, reward))

            self.average_reward_for_each_game.append(total_reward / 1000)

            print(tabulate(self.q_table, showindex="always", headers=(
                "State", "Action 0 (forward 1 step)", "Action 1 (back to zero)")))

    def display_average_reward(self):
        plt.plot(self.average_reward_for_each_game)
        plt.title("Performance over time")
        plt.ylabel("Average reward")
        plt.x_label("Episodies")
        plt.show()

    def __apply_q_learning(self, state, action, reward, new_state):
        return self.learning_rate * (reward + self.discount_factor * self.__get_expected_reward_in_next_state(new_state) - self.q_table[state, action])

    def __q_table_is_empty(self, state):
        return np.sum(self.q_table[state, :]) == 0

    def __get_random_action(self, env):
        return env.action_space.sample()

    def __get_highest_reward_action(self, state):
        return np.argmax(self.q_table[state, :])

    def __get_expected_reward_in_next_state(self, next_state):
        return np.max(self.q_table[next_state, :])

    def __with_probability(self, probability):
        return np.random.random() < probability


env = gym.make("NChain-v0")
agent = Agent()
agent.play(env)
agent.display_average_reward()
