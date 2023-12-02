from spaceship_agent import Agent
from spaceship_peragent import PERAgent
import gym
import tensorflow as tf
from spaceship_env import Env


# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)
# tf.config.experimental.set_memory_growth(gpus[0], True)

# import cProfile
# import re
# cProfile.run('re.compile("foo|bar")', 'restats')

env = Env()
# env = gym.make("LunarLander-v2")
spec = gym.spec("LunarLander-v2")
train = 0
test = 1
num_episodes = 300
graph = True

file_type = 'tf'
file = 'demo_networks/model_874'

# EPSILON LOWERED FOR NOW

dqn_agent = Agent(lr=0.00075, discount_factor=0.99, num_actions=3, epsilon=1.00, batch_size=8192, input_dims=7)
# dqn_agent = PERAgent(env, num_actions=3, input_dimensions=7, learning_rate=0.00075, train_nums=10000)

if train and not test:
    # dqn_agent.train_model(env, num_episodes, graph, None, None)
    dqn_agent.train_model()
else:
    dqn_agent.test(env, num_episodes, file_type, file, graph)

# dqn_agent.test_all(env, 500)
