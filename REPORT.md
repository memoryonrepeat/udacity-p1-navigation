### Overview

The algorithm being used to train the bot is Deep Q-Learning. It's basically an improved version of Q-Learning where deep neural network is used to approximate the Q-function, which gives expected return when taking an action at the given state and following the same policy afterwards. More theoretical background can be found in the [paper](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning) 

### Experiments

For reference, every experiments are recorded in the notebook with more additional comments along the line. This section will sum up what I did and suggest some future work.

First I started with a simple neural network setup with 128 nodes at layer 1 and 64 nodes at layer 2, with pure deep Q-Learning. The agent (defined in `dqn_agent.py`) use these hyperparameters as config

- BUFFER_SIZE = int(1e5)  # replay buffer size
- BATCH_SIZE = 64         # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR = 5e-4               # learning rate 
- UPDATE_EVERY = 4        # how often to update the network

For training the agent, these parameters are also defined as follow

- n_episodes = 1000       # maximum number of training episodes
- max_t = 1000            # maximum number of timesteps per episode
- eps_start = 1.0         # starting value of epsilon, for epsilon-greedy action selection
- eps_end = 0.01          # minimum value of epsilon
- eps_decay = 0.999       # multiplicative factor (per episode) for decreasing epsilon

Training will stop when agent reach average score of 13 anyway, so I set high n_episodes at the beginning to see how far it can go with minimal setup.

There is no [Dueling DQN](https://arxiv.org/pdf/1511.06581.pdf) or [prioritized experience replay](https://arxiv.org/abs/1511.05952) setup at the beginning either.

First experiment could not finish the task within 1000 episodes. So I tried to improve it by introducing dueling DQN. It's basically about splitting the current deep neural network into 2 streams at the end, one for evaluating state value (irrespective of action) and one for evaluating action advantage (how beneficial this action will bring), to come up with a better approximation for Q-value.

This improvement does not improve much either. But I see that score was still growing at steady rate, so I tried running it a bit longer, with max 2000 episodes with hope that I will keep improving.

And score does go up together with more training episodes. But it starts to show some slow progress near episode 1500, so I decided to try a more complex network, with 1024 nodes at the beginning and reduced by half at each layer.

Nothing seems to change. So I thought the problem must be somewhere else. I thought about the "exploitative" tendency of highly complex network, since they tend to overfit the dataset, and think whether there is a way to balance it. Which reminds me of epsilon parameter, which controls how "explorative" we allow the bot to behave.

Since epsilon and epsilon decay is already at the highest possible at the beginning, the only thing I can do is reduce it. Which means I am making the bot even more "exploitative", since the lower epsilon is, the more likely it will take the action with max expected return, instead of taking a random action (in other words, "explorative").

This might sound counterintuitive at the beginning, but since I don't know how much negative effect is actually coming from the "explorative" or "exploitative" side, experiment is the only way to answer.

So I reduced epsilon_decay from 0.99 to 0.8, to make it epsilon go down faster. And it actually do wonders to the bot! Task was finished at around episode 300. Looks like it's the right direction. I tried reducing epsilon decay even further to 0.5, and got an even better result - task was finished around 215 episodes.

Then I started to think what other improvements can be done. I looked at the dueling layers and see that it's still pretty simple, with no intermediate layers. So I added one more layer for both the state value and the advantage streams, and run them with same setup. Performance dropped slightly - task was finished at about 300 episodes. Maybe those streams are better kept simple.

So I decided to try one final experiment, which is prioritized experience replay. This means instead of random sampling from the experiences, we assign different probabilities to each of them, based on the loss value. This was implemented in the agent file.

This time it takes about 221 episodes to finish. I decided to conclude the experiment here. Biggest takeway is how big an impact it can make by reducing epsilon decay. I will remember this in future experiments. Other potential improvements can be introducing [Double DQN](https://arxiv.org/pdf/1509.06461.pdf), [Noisy DQN](https://arxiv.org/abs/1706.10295). There is also a nice little improvement called [Reward Based Epsilon Decay](https://arxiv.org/abs/1910.13701). Using more complex network such as convolutional net is also an option. And once I have all that, combining them together into [Rainbow DQN](https://arxiv.org/abs/1710.02298) is definitely a must.





