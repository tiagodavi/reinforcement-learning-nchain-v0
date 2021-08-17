import gym
import numpy as np
from tabulate import tabulate

class Agent():
  def __init__(self):
    self.reward_table = np.zeros((5,2))
  
  def play(self, env):
    state = env.reset()
    endgame = False
    while not endgame:
      action = self.__get_random_action(env) if self.__reward_table_is_empty(state) else self.__get_highest_reward_action(state)
      new_state, reward, endgame, _ = env.step(action)
      self.reward_table[state, action] += reward
      print("state {}, action {}, new_state {}, reward {}".format(state, action, new_state, reward))
      print(tabulate(self.reward_table, showindex="always", headers=("State", "Action 0 (forward 1 step)", "Action 1 (back to zero)")))
      state = new_state

  def __reward_table_is_empty(self, state):
    return np.sum(self.reward_table[state, :]) == 0

  def __get_random_action(self, env):
    return env.action_space.sample()

  def __get_highest_reward_action(self, state):
    return np.argmax(self.reward_table[state, :])

env = gym.make("NChain-v0")
agent = Agent()
agent.play(env)