import gym
from gym import spaces
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from sklearn import preprocessing

BUY = 1
NEUTRAL = 0
SELL = -1
PRICE_DIFF = "price_diff"
CLOSE = "close"
ENV_DIR = "env_data/"


class MarketEnv(gym.Env):

    def __init__(self, filename, period, feature_num,
                 mode='TP', use_batch=True, start_date=None, start_idx=None,
                 cost_rate=0.0015):
        assert mode in {'TP', 'SR'}, 'No such mode.'
        self.data = pd.read_hdf(ENV_DIR + filename)
        self.data_datetime = self.data.index
        self.price_diff_col_num = self.data.columns.get_loc(PRICE_DIFF)
        self.close_col_num = self.data.columns.get_loc(CLOSE)
        self.feature_num = feature_num  # number of features
        self.mode = mode
        self.use_batch = use_batch
        self.cost_rate = cost_rate
        self.cost_fixed = 20
        self.period = period

        # set start data and period
        if start_date is not None:
            self.set_start(start_date=start_date)
        elif start_idx is not None:
            self.set_start(start_idx=start_idx)
        else:
            raise ValueError('start date and index is all None')

        # action and observe space
        self.action_space_length = 3  # {-1, 0, 1}
        self.action_space = spaces.Discrete(self.action_space_length)
        if self.use_batch:
            self.observation_space = spaces.Box(np.ones(self.feature_num) * -1,
                                                np.ones(self.feature_num))
        else:
            self.observation_space = spaces.Box(np.ones(self.feature_num + self.action_space_length) * -1,
                                                np.ones(self.feature_num + self.action_space_length))

        self.seed()

    def set_start(self, start_idx=None, start_date=None, period=None):
        if start_date is None:
            if start_idx is None:
                raise ValueError("Need start date or index")
            self.start_idx = start_idx
            self.start_date = self.data_datetime[start_idx]
        else:
            self.start_date = start_date
            self.start_idx = self.data_datetime.get_loc(pd.Timestamp(start_date))

        if period is not None:
            self.period = period
        self.set_period(self.period)


    def set_period(self, period):
        self.period = period
        self.data_slice = self.data.iloc[self.start_idx:self.start_idx+period, :]
        self.reset()

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            state (object): the initial observation.
        """
        self.curr_line = 0
        self.reward = 0
        self.done = False
        self.cum_reward = 0
        self.cum_profit = 0
        self.prev_action = 0
        if self.use_batch:
            self.state = self.data_slice.iloc[:, :self.feature_num].values
            self.price_diff = self.data_slice.iloc[1:, self.price_diff_col_num].values
            self.cost = self.cost_rate * self.data_slice.iloc[1:, self.close_col_num].values
            # self.cost = self.cost_fixed * np.ones(self.data_slice.shape[0]-1)
            # self.scaler = preprocessing.StandardScaler()
            # self.state = self.scaler.fit_transform(self.state)

            # reshape
            self.state = self.state.reshape(self.period, 1, self.feature_num)
            self.price_diff = self.price_diff[:, None]
            self.cost = self.cost[:, None]
        else:
            raise NotImplementedError
            # self.state = np.append(self.data_slice.iloc[self.curr_line, :self.feature_num].values, np.array([0, 1, 0]))
        return self.state, self.done

    def get_reward(self, action):
        """Reward function that maps action to rewards

        Args:
            data_slice:
            action:
            mode:

        Returns:
            reward
        """
        assert self.mode in {'TP', 'SR'}, 'No such mode.'
        if not self.use_batch:
            profit = self.prev_action * self.data_slice.iloc[self.curr_line, self.price_diff_col_num] - \
                     self.cost * np.abs(action - self.prev_action)
        else:
            raise NotImplementedError
            # profit = action[:-1] * self.data_slice.iloc[1:, self.price_diff_col_num].values - \
            #          self.cost_rate * np.abs(action[1:] - action[:-1])
        if self.mode == 'TP':
            reward = profit
        elif self.mode == 'SR':
            ## TODO: add SR as reward
            raise NotImplementedError
        else:
            raise ValueError
        return reward, profit


    def step(self, action):
        if self.done:
            return self.state, self.reward, self.done, {}

        # the action will bring us to the next state
        self.curr_line += 1  # prep for new states
        self.prev_action = action  # save curr action as prev

        # calc reward (current reward depends on curr and prev action)
        if self.curr_line == self.T - 1:
            self.done = True
        self.reward, profit = self.get_reward(action)
        self.cum_reward += self.reward
        self.cum_profit += profit

        # generate next state
        action_state = np.zeros(3)
        action_state[int(action+1)] = 1
        self.state = np.append(self.data_slice.iloc[self.curr_line, :self.feature_num].values, action_state)

        return self.state, \
               self.reward, \
               self.done, \
               {
                   "datetime": self.data_slice.index[self.curr_line],
                   "cum_profit": self.cum_profit,
                   "cum_reward": self.cum_reward,
                   "price_diff": self.data_slice.iloc[self.curr_line, self.price_diff_col_num],
                   "cost": self.cost_rate,
                   "action": self.prev_action,
               }

    def step_batch(self, action):
        # the action will directly bring us to the end

        # self.curr_line = self.start_idx  # prep for new states
        self.done = True
        self.prev_action = action  # save curr action as prev

        # calc reward (current reward depends on curr and prev action)
        self.reward, profit = self.get_reward(action)
        self.cum_reward = np.cumsum(self.reward)
        self.cum_profit = np.cumsum(profit)


        return self.state, \
               self.reward, \
               self.done, \
               {
                   "datetime": self.data_slice.index[1:],
                   "cum_profit": self.cum_profit,
                   "cum_reward": self.cum_reward,
                   "action": self.prev_action,
               }

    def render(self, mode='human', close=False):
        if close:
            return
        return self.state


class FuzzyMarketEnv(gym.Env):

    def __init__(self, filename, period, feature_num,
                 mode='TP', use_batch=True, start_date=None, start_idx=None,
                 cost_rate=0.0015):
        assert mode in {'TP', 'SR'}, 'No such mode.'
        self.data = pd.read_hdf(ENV_DIR + filename)
        self.data_datetime = self.data.index
        self.price_diff_col_num = self.data.columns.get_loc(PRICE_DIFF)
        self.close_col_num = self.data.columns.get_loc(CLOSE)
        self.feature_num = feature_num  # number of features
        self.mode = mode
        self.use_batch = use_batch
        self.cost_rate = cost_rate
        self.cost_fixed = 20
        self.period = period

        # set start data and period
        if start_date is not None:
            self.set_start(start_date=start_date)
        elif start_idx is not None:
            self.set_start(start_idx=start_idx)
        else:
            raise ValueError('start date and index is all None')

        # action and observe space
        self.action_space_length = 3  # {-1, 0, 1}
        self.action_space = spaces.Discrete(self.action_space_length)
        if self.use_batch:
            self.observation_space = spaces.Box(np.ones(self.feature_num*3) * -1,
                                                np.ones(self.feature_num*3))
        else:
            self.observation_space = spaces.Box(np.ones(self.feature_num + self.action_space_length) * -1,
                                                np.ones(self.feature_num + self.action_space_length))

        self.seed()

    def set_start(self, start_idx=None, start_date=None, period=None):
        if start_date is None:
            if not start_idx:
                raise ValueError("Need start date or index")
            self.start_idx = start_idx
            self.start_date = self.data_datetime[start_idx]
        else:
            self.start_date = start_date
            self.start_idx = self.data_datetime.get_loc(pd.Timestamp(start_date))

        if period is not None:
            self.period = period
        self.set_period(self.period)


    def set_period(self, period):
        self.period = period
        self.data_slice = self.data.iloc[self.start_idx:self.start_idx+period, :]
        self.reset()

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            state (object): the initial observation.
        """
        self.curr_line = 0
        self.reward = 0
        self.done = False
        self.cum_reward = 0
        self.cum_profit = 0
        self.prev_action = 0
        if self.use_batch:
            state = self.data_slice.iloc[:, :self.feature_num].values
            kmeans = KMeans(n_clusters=3, random_state=0).fit(state[:, -4:])
            label = kmeans.labels_
            fuzzy_states = []
            for i in range(3):
                cluster = state[np.where(label == i)]
                fuzzy_mean = np.mean(cluster, axis=0)
                fuzzy_var = np.var(cluster, axis=0)
                fuzzy_states.append(np.exp(-np.square(state - fuzzy_mean) / fuzzy_var))
            self.state = np.concatenate(fuzzy_states, axis=1)


            self.price_diff = self.data_slice.iloc[1:, self.price_diff_col_num].values
            self.cost = self.cost_rate * self.data_slice.iloc[1:, self.close_col_num].values
            # self.cost = self.cost_fixed * np.ones(self.data_slice.shape[0]-1)
            # self.scaler = preprocessing.StandardScaler()
            # self.state = self.scaler.fit_transform(self.state)

            # reshape
            self.state = self.state.reshape(self.period, 1, self.feature_num*3)
            self.price_diff = self.price_diff[:, None]
            self.cost = self.cost[:, None]
        else:
            raise NotImplementedError
            # self.state = np.append(self.data_slice.iloc[self.curr_line, :self.feature_num].values, np.array([0, 1, 0]))
        return self.state, self.done

    def get_reward(self, action):
        """Reward function that maps action to rewards

        Args:
            data_slice:
            action:
            mode:

        Returns:
            reward
        """
        assert self.mode in {'TP', 'SR'}, 'No such mode.'
        if not self.use_batch:
            profit = self.prev_action * self.data_slice.iloc[self.curr_line, self.price_diff_col_num] - \
                     self.cost * np.abs(action - self.prev_action)
        else:
            raise NotImplementedError
            # profit = action[:-1] * self.data_slice.iloc[1:, self.price_diff_col_num].values - \
            #          self.cost_rate * np.abs(action[1:] - action[:-1])
        if self.mode == 'TP':
            reward = profit
        elif self.mode == 'SR':
            ## TODO: add SR as reward
            raise NotImplementedError
        else:
            raise ValueError
        return reward, profit


    def step(self, action):
        if self.done:
            return self.state, self.reward, self.done, {}

        # the action will bring us to the next state
        self.curr_line += 1  # prep for new states
        self.prev_action = action  # save curr action as prev

        # calc reward (current reward depends on curr and prev action)
        if self.curr_line == self.T - 1:
            self.done = True
        self.reward, profit = self.get_reward(action)
        self.cum_reward += self.reward
        self.cum_profit += profit

        # generate next state
        action_state = np.zeros(3)
        action_state[int(action+1)] = 1
        self.state = np.append(self.data_slice.iloc[self.curr_line, :self.feature_num].values, action_state)

        return self.state, \
               self.reward, \
               self.done, \
               {
                   "datetime": self.data_slice.index[self.curr_line],
                   "cum_profit": self.cum_profit,
                   "cum_reward": self.cum_reward,
                   "price_diff": self.data_slice.iloc[self.curr_line, self.price_diff_col_num],
                   "cost": self.cost_rate,
                   "action": self.prev_action,
               }

    def step_batch(self, action):
        # the action will directly bring us to the end

        # self.curr_line = self.start_idx  # prep for new states
        self.done = True
        self.prev_action = action  # save curr action as prev

        # calc reward (current reward depends on curr and prev action)
        self.reward, profit = self.get_reward(action)
        self.cum_reward = np.cumsum(self.reward)
        self.cum_profit = np.cumsum(profit)


        return self.state, \
               self.reward, \
               self.done, \
               {
                   "datetime": self.data_slice.index[1:],
                   "cum_profit": self.cum_profit,
                   "cum_reward": self.cum_reward,
                   "action": self.prev_action,
               }

    def render(self, mode='human', close=False):
        if close:
            return
        return self.state
