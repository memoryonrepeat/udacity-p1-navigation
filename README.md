### Overview 

This repo contains my submission for Udacity Deep Reinforcement Learning course - Project 1.

The project is about training a bot to pick up yellow bananas and avoid purple bananas.

Each yellow banana yield +1 reward, while each purple banana yield -1 reward.

The state space has 37 dimensions ranging from the bot's velocity to perception of objects around etc...

For available actions are (forward, backward, left, right) which maps to (0,1,2,3)

The task is episodic and is considered successful when the bot can obtain an average score of at least 13 points
over a window of 100 consecutive episodes.

### Setup

The agent is implemented in `dqn_agent.py`.

There is a `Navigation.ipynb` file with all dependency requirements placed on the top cell, including the agent file.

These dependencies are readily available on Udacity workspace. On local, any missing dependency can be installed via `pip install <dependency name>`.

Models are defined in-line within the notebook, to make it easier to fiddle with different models and see the experiment progress.

The banana collection environment can be obtained from the [official Udacity repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation#getting-started).

Once downloaded, unzip and place it in the same folder with the notebook, and change the file path accordingly to point to the environment file.

Now setup is finished. Running all the cells within `Navigation.ipynb` will walk you through the experiments.

Models that led to successful task is saved in the `.pth` files. There are many of them, corresponding to many successful experiments (more details in the report).

Note that to be able to see the bot while training, the notebook needs to be run on local.


### Experiments

This can be found in the `REPORT.md` file
