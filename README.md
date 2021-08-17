## NChain-v0 Game

It solves Nchain game with 3 different kinds of agents from reward table to neural networks

 1. agent_with_reward_table (simple)
 2. agent_with_q_learning (q-learning algorithm)
 3. agent_with_neural_network (deep q-learning)

n-Chain environment

This game presents moves along a linear chain of states, with two actions:

0.  forward, which moves along the chain but returns no reward
1.  backward, which returns to the beginning and has a small reward

The end of the chain, however, presents a large reward, and by moving 'forward' at the end of the chain this large reward can be repeated.

The goal is to beat this environment and make the agent "wait" for better rewards that are long term instead of getting stuck with small ones. The problem is we have no immediate reward and we want to make this agent "think" of a better strategy.

https://gym.openai.com/envs/NChain-v0/


