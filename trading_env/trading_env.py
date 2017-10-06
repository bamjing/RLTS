# -*- coding: utf-8 -*-
from gym import spaces
from Module.trading_env.PortfolioManager import PortfolioManager
import numpy as np
import pandas as pd
import random
from Module.RL import Interpretation

pd.options.display.width = 500
pd.options.display.height = 300
class TradingEnv(object):
    def __init__(self, SESS, para_dict):

        self.initialFunds = para_dict['INITIAL_FUNDS']
        dat_path = para_dict['LABELED_PATH'] + para_dict['LABELED_NAME']
        self.maximumDrawdown = para_dict['MAX_DRAWDOWN']
        self.STOP_LOSS = para_dict['STOP_LOSS']
        self.MAX_POSITION = para_dict['MAX_POSITION']
        self.INITIAL_FUNDS = para_dict['INITIAL_FUNDS']
        self.LOOK_BACK = para_dict['LOOK_BACK']
        self.DAY_HOURS = para_dict['DAY_HOURS']
        self.df = pd.read_csv(dat_path, index_col=0, parse_dates=['Date'])

        self.action_space = spaces.MultiDiscrete([[0, 2]])
        self.observation_space = spaces.Box(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))

        self.init_step_list = self.df[(self.df['Date'].dt.hour == 15) & (self.df['Date'].dt.minute == 0)].index.tolist()
        self.init_step_list.append(0)

        self.dataLength = len(self.df.index)
        self.max_pos = 0

        self.win_los_log = pd.DataFrame()
        self.win_list = []
        self.loss_list = []

        self.error_time = 0
        self.total_time = 0

        self.interperater = Interpretation.Interpreter(SESS, para_dict)

    def step(self, action, CURRENT_EPOCH):
        action = self._action_converter(action)

        if (self.stepCounts < self.dataLength and self.stepCounts + 1 < self.dataLength and self.stepCounts - self.LOOK_BACK >= 0):

            pos = self.portfolioManager.get_position()
            if action in self.next_action_list and (pos + action) in range(-self.MAX_POSITION, self.MAX_POSITION + 1):
                execute_action = action
            else:
                execute_action = 0

            portfolio_state, pos = self.portfolioManager.proceedAction(action=execute_action, currentState=self.df.iloc[self.stepCounts])

            self.max_pos = max(self.max_pos, abs(pos))

            if portfolio_state[1] < 0:
                self.error_time += 1

            if portfolio_state[1] > 0:
                self.error_time = 0

            if action != 0:
                self.total_time += 1


            log_time = np.array(self.df.iloc[self.stepCounts]['Date'])

            log_action = action
            if pos == self.MAX_POSITION or pos == -self.MAX_POSITION:
                if pos != 0 and action != 0:
                    if (action / abs(action)) * (pos / abs(pos)) == 1:
                        log_action = '* %d *' % action

            if pos == self.MAX_POSITION:
                log_pos = '%d *' % pos
            elif pos == -self.MAX_POSITION:
                log_pos = '* %d' % pos
            else:
                log_pos = pos

            if len(self.win_los_log) == 0:
                self.win_los_log = pd.DataFrame({'Time': log_time, 'unrealized': portfolio_state[2], 'Realized': portfolio_state[1],
                     'funds': portfolio_state[0],
                     'action': log_action, 'position': log_pos}, index=[0],
                    columns=['Time', 'unrealized', 'Realized', 'funds', 'action', 'position'])
            else:
                tmp = pd.DataFrame({'Time': log_time, 'unrealized': portfolio_state[2], 'Realized': portfolio_state[1], 'funds': portfolio_state[0],
                                    'action': log_action, 'position': log_pos}, index=[0],
                                   columns=['Time', 'unrealized', 'Realized', 'funds', 'action', 'position'])
                self.win_los_log = pd.concat([self.win_los_log, tmp])


            if portfolio_state[1] > 0:
                self.win_list.append(portfolio_state[1])

            if portfolio_state[1] < 0:
                self.loss_list.append(abs(portfolio_state[1]))


            # 查看是否触及止损线
            #if pos != 0 and (portfolio_state[2] < -self.STOP_LOSS * abs(pos)):
                #portfolio_state, _ = self.portfolioManager.proceedAction(action=-pos, currentState=self.df.iloc[self.stepCounts])
                #print "stop loss order trigged.", self.max_pos

            funds = portfolio_state[0]

            self.stepCounts += 1
            nextState = self.df.iloc[self.stepCounts - self.LOOK_BACK: self.stepCounts]

            self.next_action_list = self.portfolioManager.nextAvailableActions(nextState)
            rewards, self.done = self._get_reward_done(funds, portfolio_state, nextState, action, CURRENT_EPOCH, pos)


            time, volume_percent = self._get_floated_time_volume(nextState)


            candleState = self.interperater.research(np.array(nextState, dtype=pd.Series)[:, 7:10])
            output_state = [candleState, self._get_neural_portfolio_state(portfolio_state, pos), volume_percent, time]

            return output_state, rewards, self.done, [funds, self.portfolioManager.transactionHistory]
        else:
            print "out bound of the training data"
            return None, 0, True, None

    def _get_reward_done(self, funds, portfolio_state, nextState, action, CURRENT_EPOCH, pos):

        # 鼓励好的动作，好的策略，但是不鼓励预测市场，所以奖励和获利幅度无关
        #reward 和积累下来的好行为频率有关
        done = False

        # 没有动作可执行的时候停止
        if self.next_action_list == [0]:
            done = True
            print self.win_los_log
            print "gameover, no action to perform", self.max_pos
            print "win:", len(self.win_list), np.array(self.win_list).mean()
            print "loss:", len(self.loss_list), np.array(self.loss_list).mean()


        # 最大回撤的时候停止

        if funds < self.initialFunds * self.maximumDrawdown:
            done = True
            print self.win_los_log
            print "gameover!, meet maximumDrawdown!!", self.max_pos
            print "win:", len(self.win_list), np.array(self.win_list).mean()
            print "loss:", len(self.loss_list), np.array(self.loss_list).mean()


        # 一天结束的时候停止
        hour = np.array(nextState.tail(1)['Date'].dt.hour)[0]
        minute = np.array(nextState.tail(1)['Date'].dt.minute)[0]

        if hour == 13 and minute == 55:
            done = True
            pos = self.portfolioManager.get_position()
            if pos != 0:
                portfolio_state, _ = self.portfolioManager.proceedAction(action=-pos, currentState=self.df.iloc[self.stepCounts])
                funds = self.portfolioManager.funds

            print self.win_los_log
            print "finished today", funds, self.max_pos
            print "win:", len(self.win_list), np.array(self.win_list).mean()
            print "loss:", len(self.loss_list), np.array(self.loss_list).mean()

        rewards = 0

        """
        if pos != 0:
            if portfolio_state[1] + portfolio_state[2] > 0:
                rewards = 1

        if portfolio_state[1] < 0:
            rewards = -1
        """

        if portfolio_state[1] + portfolio_state[2] > 0:
            rewards = 1

        if portfolio_state[1] + portfolio_state[2] < 0:
            rewards = -1


        """
        if action != 0:
            if pos != 0:
                if (pos / abs(pos)) * (action / abs(action)) and abs(pos) >= self.MAX_POSITION:
                    rewards = -2
        """

        return rewards, done


    def reset(self):
        self.portfolioManager = PortfolioManager(initialFunds=self.initialFunds)
        self.stepCounts = 0
        while self.stepCounts - self.LOOK_BACK < 0:
            self.stepCounts = random.choice(self.init_step_list)
        self.done = False
        self.max_pos = 0

        self.win_los_log = pd.DataFrame()
        self.win_list = []
        self.loss_list = []

        self.error_time = 0
        self.total_time = 0

        nextState = self.df.iloc[self.stepCounts - self.LOOK_BACK : self.stepCounts]
        self.next_action_list = self.portfolioManager.nextAvailableActions(nextState)

        portfolio_state = [self.INITIAL_FUNDS, 0, 0]

        time, volume_percent = self._get_floated_time_volume(nextState)

        candleState = self.interperater.research(np.array(nextState, dtype=pd.Series)[:, 7:10])
        output_state = [candleState, self._get_neural_portfolio_state(portfolio_state, 0), volume_percent, time]
        return output_state

    def _get_neural_portfolio_state(self, portfolio_state, position):
        portfolio_state[0] = ((portfolio_state[0] - self.INITIAL_FUNDS) / 1000) #0.1, 1
        portfolio_state[1] = (portfolio_state[1] / 10)   #0.1, 1
        portfolio_state[2] = (portfolio_state[2] / 10)  #0.1, 1
        portfolio_state.append(position)       #0.2, -0.2
        return portfolio_state

    def _action_converter(self, action):
        if action == 0:
            converted_action = 0
        elif action == 1:
            converted_action = -1
        else:
            converted_action = 1
        return converted_action

    def _get_floated_time_volume(self, nextState):
        hour = np.array(nextState['Date'].dt.hour)[-1]
        minute = np.array(nextState['Date'].dt.minute)[-1]
        time = (((hour + (minute / 100.0))) / 100.0) + 10e-15

        volume_percent = np.array(nextState, dtype=pd.Series)[:, 11:12][-1][0] + 1e-15

        return [time], [volume_percent]

