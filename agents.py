import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, SimpleRNN, GRU, Softmax, GRUCell, LSTM
from tensorflow.keras.layers import Lambda, Activation
import gym
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import Lambda, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects

import argparse
from gym import wrappers, logger
import csv
import time
import os
import datetime
import math
import sys


class PolicyAgent:
    def __init__(self, train_env, test_env):
        self.memory = []
        self.reset_env(train_env, test_env)
        self.model = None

    def reset_env(self, train_env, test_env):
        """Reset new env for rolling training
        """
        self.train_env = train_env
        self.test_env = test_env

    def reset_memory(self):
        """Clear the memory before the start of every episode
        """
        self.memory = []

    def remember(self, item):
        """
        Remember every s,a,r,s' in every step of the episode
        item = [step, state, next_state, reward, done]
        """
        self.memory.append(item)

    def save_weights(self, filepath):
        """Save the actor, critic and encoder weights
            useful for restoring the trained models
        """
        self.model.save_weights(filepath, save_format='h5')

    def load_weights(self, filepath):
        """Load the trained weights
           useful if we are interested in using
                the network right away
        Arguments:
        """
        self.model.load_weights(filepath, by_name=False)

    def act(self, state):
        """Call the policy network to sample an action during training
        Argument:
            state (tensor): environment state
        Return:
            act (tensor): policy action
        """
        return self.model(state)


class DengAgent(PolicyAgent):

    def __init__(self, train_env, test_env):
        """Implements the models and training of Deng(2016)
        Arguments:
            env (Object): OpenAI gym environment
        """
        super().__init__(train_env, test_env)
        self.build_model([128,128,128,20], 20)
        self.opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.opt1 = SGD(nesterov=True, decay=1e-4, lr=0.001, momentum=0.9)
        self.reset_memory()

    def build_model(self, dense_shapes, rnn_hidden):
        self.model_layers = []
        assert dense_shapes
        inp = Input(shape=(1, self.train_env.observation_space.shape[0]), name='state')
        for i, dense_length in enumerate(dense_shapes):
            self.model_layers.append(Dense(dense_length, activation='elu', name='DNN_Layer{}'.format(i+1)))
        self.model_layers.append(GRU(rnn_hidden, name='GRU'))
        self.model_layers.append(Dense(1, activation='tanh', name='Action'))
        x = None
        for i, l in enumerate(self.model_layers):
            if i == 0:
                x = l(inp)
            else:
                x = l(x)

        self.model = Model(inp, x, name='model')
        self.model.summary()
        plot_model(self.model, to_file='plots/model.png', show_shapes=True)

    def reset_memory(self):
        """Clear the memory before the start
            of every episode
        """
        self.memory = []
        self.cum_reward = []
        self.cum_profit = []
        self.grad = None
        self.loss = None
        self.actions = None

    def act_and_get_grad(self):
        """Feed forward step, called for each episode
        Special for Direct RL, we record delta and action for end of episode BP
        Args:
            all_states: a batch of states generated from env

        Returns:

        """
        # get the gradient and do grad descent on total reward
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            self.loss, self.actions, self.reward = self.get_action_and_loss(train=True)
            self.cum_reward = np.cumsum(self.reward.numpy())
            self.grad = tape.gradient(self.loss, self.model.trainable_variables)
            self.grad = [tf.math.l2_normalize(g) for g in self.grad] # no need for Adam
        return self.actions, self.loss, self.grad

    def get_action_and_loss(self, train=True):
        # set 15 to have a approximation of action and also avoid zero gradient
        # actions = tf.reduce_sum(tf.nn.softmax(x * 15) * self.action_range, axis=-1) - 1.0
        if train:
            all_states = self.train_env.state
            price_diff = self.train_env.price_diff
            cost = self.train_env.cost
        else:
            all_states = self.test_env.state
            price_diff = self.test_env.price_diff
            cost = self.test_env.cost
        actions = self.model(all_states)
        reward = price_diff * actions[:-1] - cost * tf.abs(actions[1:] - actions[:-1])
        loss = -1 * tf.reduce_sum(reward)
        return loss, actions, reward


    def train_by_episode(self):
        """Train function triggered at the end of each episode

        Returns:

        """
        self.act_and_get_grad()
        self.opt.apply_gradients(zip(self.grad, self.model.trainable_variables))

        eval_loss, _, _ = self.get_action_and_loss(train=False)
        print("Train loss = {}, Validation loss = {}".format(self.loss, eval_loss))
        return self.loss, eval_loss

