import gym
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras import optmizers


class Agent():
    def __init__(self):
        self.learning_rate = 0.05
        self.neural_network = NeuralNetwork(self.learning_rate)
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

            while not endgame:
                if self.__with_probability(self.epsilon):
                    action = self.__get_random_action(env)
                else:
                    action = self.__get_highest_reward_action(state)

                new_state, reward, endgame, _ = env.step(action)
                total_reward += reward

                target_output = self.neural_network.predict_reward(state)

                target_output[action] = reward * \
                    self.discount_factor * \
                    self.__get_expected_reward_in_next_state(new_state)

                self.neural_network.train(state, target_output)

                state = new_state

            self.average_reward_for_each_game.append(total_reward / 1000)

            print(tabulate(self.neural_network.display_result(), showindex="always", headers=(
                "State", "Action 0 (forward 1 step)", "Action 1 (back to zero)")))

    def display_average_reward(self):
        plt.plot(self.average_reward_for_each_game)
        plt.title("Performance over time")
        plt.ylabel("Average reward")
        plt.x_label("Episodies")
        plt.show()

    def __get_random_action(self, env):
        return env.action_space.sample()

    def __get_highest_reward_action(self, state):
        return np.argmax(self.neural_network.predict_reward(state))

    def __get_expected_reward_in_next_state(self, next_state):
        return np.max(self.neural_network.predict_reward(next_state))

    def __with_probability(self, probability):
        return np.random.random() < probability


class NeuralNetwork(Sequential):
    def __init__(self, learning_rate=0.05):
        super().__init__()
        self.add(InputLayer(batch_input_shape=(1, 5)))
        self.add(Dense(10, activation="sigmoid"))
        self.add(Dense(2, activation="linear"))
        self.compile(loss="mse", optimizer=optmizers.Adam(lr=learning_rate))

    def train(self, state, target_output):
        input_signal = self.__convert_state_to_neural_network_input(state)
        target_output = target_output.reshape(-1, 2)
        self.fit(input_signal, target_output, epochs=1, verbose=0)

    def predict_reward(self, state):
        input_signal = self.__convert_state_to_neural_network_input(state)
        return self.predict(input_signal)[0]

    def display_result(self):
        results = []
        for state in range(0, 5):
            results.append(self.predict_reward(state))
        return results

    def __convert_state_to_neural_network_input(self, state):
        input_signal = np.zeros((1, 5))
        input_signal[0, state] = 1
        return input_signal


env = gym.make("NChain-v0")
agent = Agent()
agent.play(env)
agent.display_average_reward()
