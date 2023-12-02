import tensorflow as tf
# print(tf.__version__)

import gym
import time
import numpy as np
import pygame as pg
import tensorflow as tf
# from memory_profiler import profile
from tensorflow.keras.layers import Dense, Input, Add
from tensorflow.keras.optimizers import Adam

#from pympler import muppy, summary
import pandas as pd

np.random.seed(1)
tf.random.set_seed(1)

# replay buffer
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity    # N, the size of replay buffer, so as to the number of sum tree's leaves
        self.tree = np.zeros(2 * capacity - 1)  # equation, to calculate the number of nodes in a sum tree
        self.transitions = np.empty(capacity, dtype=object)
        self.next_idx = 0

    @property
    def total_p(self):
        return self.tree[0]

    def add(self, priority, transition):
        idx = self.next_idx + self.capacity - 1
        self.transitions[self.next_idx] = transition
        self.update(idx, priority)
        self.next_idx = (self.next_idx + 1) % self.capacity

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)    # O(logn)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, s):
        idx = self._retrieve(0, s)   # from root
        trans_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.transitions[trans_idx]

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

def DeepQNetwork(lr, num_actions, input_dims):
    state_input = Input((input_dims,))

    backbone_1 = Dense(100, activation='relu')(state_input)
    backbone_2 = Dense(200, activation='relu')(backbone_1)
    backbone_3 = Dense(100, activation='relu')(backbone_2)

    value_output = Dense(1)(backbone_3)
    advantage_output = Dense(num_actions)(backbone_3)

    output = Add()([value_output, advantage_output])

    model = tf.keras.Model(state_input, output)
    model.compile(loss='mse', optimizer=Adam(lr))

    return model

class PERAgent:  # Double DQN with Proportional Prioritization
    def __init__(self, env, num_actions, input_dimensions, learning_rate=.0012, epsilon=1, epsilon_dacay=0.995, min_epsilon=.03,
                 gamma=.95, batch_size=8, tau = 0.01, train_nums=10000, buffer_size=100, replay_period=20,
                 alpha=0.4, beta=0.4, beta_increment_per_sample=0.001):
        self.model = DeepQNetwork(learning_rate, num_actions, input_dimensions)
        self.target_model = DeepQNetwork(learning_rate, num_actions, input_dimensions)

        # parameters
        self.env = env                              # gym environment
        self.action_space = num_actions
        self.lr = learning_rate                     # learning step
        self.epsilon = epsilon                      # e-greedy when exploring
        self.epsilon_decay = epsilon_dacay          # epsilon decay rate
        self.min_epsilon = min_epsilon              # minimum epsilon
        self.gamma = gamma                          # discount rate
        self.batch_size = batch_size                # minibatch k
        self.tau = tau                              # percentage to soft update
        self.train_nums = train_nums                # total training steps

        # replay buffer params [(state, action, reward, new_state, done), ...]
        self.b_obs = np.empty((self.batch_size,) + self.env.reset().shape)
        self.b_actions = np.empty(self.batch_size, dtype=np.int8)
        self.b_rewards = np.empty(self.batch_size, dtype=np.float32)
        self.b_next_states = np.empty((self.batch_size,) + self.env.reset().shape)
        self.b_dones = np.empty(self.batch_size, dtype=np.bool)

        self.replay_buffer = SumTree(buffer_size)   # sum-tree data structure
        self.buffer_size = buffer_size              # replay buffer size N
        self.replay_period = replay_period          # replay period K
        self.alpha = alpha                          # priority parameter, alpha=[0, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.beta = beta                            # importance sampling parameter, beta=[0, 0.4, 0.5, 0.6, 1]
        self.beta_increment_per_sample = beta_increment_per_sample
        self.num_in_buffer = 0                      # total number of transitions stored in buffer
        self.margin = 0.01                          # pi = |td_error| + margin
        self.p1 = 1                                 # initialize priority for the first transition
        # self.is_weight = np.empty((None, 1))
        self.is_weight = np.power(self.buffer_size, -self.beta)  # because p1 == 1
        self.abs_error_upper = 1

    def _per_loss(self, y_target, y_pred):
        return tf.reduce_mean(self.is_weight * tf.math.squared_difference(y_target, y_pred))
    
    def policy(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_net(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action
    
    def save(self, f):
        print("f:", f)
        self.model.save(("saved_networks/space_permodel{0}".format(f)))
        # self.model.save_weights(("saved_networks/dqn_permodel{0}/net_weights{0}.h5".format(0)))

        print("Network saved")

    def soft_update(self, q_net, target_net):
        for target_weights, q_net_weights in zip(target_net.weights, q_net.weights):
            target_weights.assign(self.tau * q_net_weights + (1.0 - self.tau) * target_weights)

    # @profile
    def train_model(self):
        # initialize the initial observation of the agent
        render = True
        f = 0
        t = 1
        num_mega_steps = 0
        if render:
            self.env.ui.init_render()
        obs = self.env.reset()
        while  True: # infinite run currently in place
            t += 1
            action = self.policy(obs)  # input the obs to the network model
            next_obs, reward, done, info = self.env.step(action)    # take the action in the env to return s', r, done

            for event in pg.event.get():
                        if event.type == pg.QUIT:
                            pg.quit()
                        if event.type == pg.KEYDOWN:
                            if event.key == pg.K_s:
                                render = not render
                            if event.key == pg.K_0:
                                self.save(f)
                                f += 1
                            if event.key == pg.K_1:
                                # memory diagnostic
                                # all_objects = muppy.get_objects()
                                # sum1 = summary.summarize(all_objects)
                                # Prints out a summary of the large objects
                                # summary.print_(sum1)
                                # Get references to certain types of objects such as dataframe
                                dataframes = [ao for ao in all_objects if isinstance(ao, pd.DataFrame)]
                                for d in dataframes:
                                    print(d.columns.values)
                                    print(len(d))
            if render:
                self.env.render()

            if t == 1:
                p = self.p1
            else:
                p = np.max(self.replay_buffer.tree[-self.replay_buffer.capacity:])
            self.store_transition(p, obs, action, reward, next_obs, done)  # store that transition into replay butter
            self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)

            if t > self.buffer_size:
                # if t % self.replay_period == 0:  # transition sampling and update
                if t % 10 == 0:
                    losses = self.train_step()
                if t % 10000 == 0:
                    print('losses each 10000 steps: ', losses)
                    num_mega_steps += 1 
                    if num_mega_steps % 500 == 0:
                        self.save(f)

            if t % 10 == 0:
                self.soft_update(self.model, self.target_model)
            if done:
                obs = self.env.reset()   # one episode end
            else:
                obs = next_obs
    # @profile
    def train_step(self):
        idxes, self.is_weight = self.sum_tree_sample(self.batch_size)
        # Double Q-Learning
        best_action_idxes = self.policy(self.b_next_states)  # get actions through the current network
        target_q = self.get_target_value(self.b_next_states)    # get target q-value through the target network
        # get td_targets of batch states
        td_target = self.b_rewards + \
            self.gamma * target_q[np.arange(target_q.shape[0]), best_action_idxes] * (1 - self.b_dones)
        predict_q = self.model.predict(self.b_obs)
        td_predict = predict_q[np.arange(predict_q.shape[0]), self.b_actions]
        abs_td_error = np.abs(td_target - td_predict) + self.margin
        clipped_error = np.where(abs_td_error < self.abs_error_upper, abs_td_error, self.abs_error_upper)
        ps = np.power(clipped_error, self.alpha)
        # priorities update
        for idx, p in zip(idxes, ps):
            self.replay_buffer.update(idx, p)

        for i, val in enumerate(self.b_actions):
            predict_q[i][val] = td_target[i]

        target_q = predict_q  # just to change a more explicit name
        losses = self.model.train_on_batch(self.b_obs, target_q)

        return losses

    # proportional prioritization sampling
    def sum_tree_sample(self, k):
        idxes = []
        is_weights = np.empty((k, 1))
        self.beta = min(1., self.beta + self.beta_increment_per_sample)
        # calculate max_weight
        min_prob = np.min(self.replay_buffer.tree[-self.replay_buffer.capacity:]) / self.replay_buffer.total_p
        max_weight = np.power(self.buffer_size * min_prob, -self.beta)
        segment = self.replay_buffer.total_p / k
        for i in range(k):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, t = self.replay_buffer.get_leaf(s)
            idxes.append(idx)
            self.b_obs[i], self.b_actions[i], self.b_rewards[i], self.b_next_states[i], self.b_dones[i] = t
            # P(j)
            sampling_probabilities = p / self.replay_buffer.total_p     # where p = p ** self.alpha
            is_weights[i, 0] = np.power(self.buffer_size * sampling_probabilities, -self.beta) / max_weight
        return idxes, is_weights

    def evaluation(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        # one episode until done
        while not done:
            action, q_values = self.model.action_value(obs[None])  # Using [None] to extend its dimension (4,) -> (1, 4)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if render:  # visually show
                env.render()
            time.sleep(0.05)
        env.close()
        return ep_reward

    # store transitions into replay butter, now sum tree.
    def store_transition(self, priority, obs, action, reward, next_state, done):
        transition = [obs, action, reward, next_state, done]
        self.replay_buffer.add(priority, transition)

    # rank-based prioritization sampling
    def rand_based_sample(self, k):
        pass

    # assign the current network parameters to target network
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_target_value(self, obs):
        return self.target_model.predict(obs)

    def e_decay(self):
        self.epsilon *= self.epsilon_decay


# if __name__ == '__main__':
#     test_model()

#     env = gym.make("CartPole-v0")
#     num_actions = env.action_space.n
#     model = Model(num_actions)
#     target_model = Model(num_actions)
#     agent = PERAgent(model, target_model, env)
#     # test before
#     rewards_sum = agent.evaluation(env)
#     print("Before Training: %d out of 200" % rewards_sum)  # 9 out of 200

#     agent.train()
#     # test after
#     # env = gym.wrappers.Monitor(env, './recording', force=True)
#     rewards_sum = agent.evaluation(env)
#     print("After Training: %d out of 200" % rewards_sum)  # 200 out of 200