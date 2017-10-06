# -*- coding: utf-8 -*-
import random
from collections import deque
import numpy as np
from Module.RL.PPO import PPO
from Module import trading_env

class Worker(object):
    def __init__(self, scope, parameter_dict, SESS, MEMORY_DICT, COORD):
        self.env = trading_env.make(SESS, parameter_dict)
        self.ppo = PPO(scope, parameter_dict, self.env)
        self.COORD = COORD
        self.MEMORY_DICT = MEMORY_DICT
        self.name = scope
        self.sess = SESS
        self.CURRENT_EPOCH = 0
        self.EPOCH_MAX = parameter_dict['EPOCH_MAX']
        self.MAX_EPOCH_STEPS = parameter_dict['MAX_EPOCH_STEPS']
        self.MAX_AC_EXP_RATE = parameter_dict['MAX_AC_EXP_RATE']
        self.MIN_AC_EXP_RATE = parameter_dict['MIN_AC_EXP_RATE']

        self.AC_EXP_EPOCH = parameter_dict['AC_EXP_PERCENTAGE'] * parameter_dict['EPOCH_MAX']
        self.SCHEDULE = parameter_dict['SCHEDULE']
        self.GAMMA = parameter_dict['GAMMA']
        self.LAM = parameter_dict['LAM']
        self.ENV_SAMPLE_ITERATIONS = parameter_dict['ENV_SAMPLE_ITERATIONS']
        self.LOG_FILE_PATH = parameter_dict['LOG_FILE_PATH']
        self.push_count = 0

        self.BUFFER = self._initial_buffer()

    def _initial_buffer(self):
        buffer = {}

        buffer['candlestate'] = deque()
        buffer['actions'] = deque()
        buffer['rewards'] = deque()
        buffer['predictedValues'] = deque()
        buffer['done'] = deque()
        buffer['episode_rewards'] = deque()
        buffer['portfolio'] = deque()
        buffer['volume_percent'] = deque()
        buffer['time'] = deque()
        return buffer

    def work(self, PUSH_EVENT, UPDATE_EVENT, log_writer):
        while not self.COORD.should_stop():

            self.BUFFER = self._initial_buffer()

            state = self.env.reset()

            EPISODE_REWARD = 0
            index = 0

            earn_reward = []
            selling = []
            zero = []
            buying = []
            funds = []

            while index < self.MAX_EPOCH_STEPS:
                if not PUSH_EVENT.is_set():
                    PUSH_EVENT.wait()
                    self.sess.run(self.ppo.sync_pis)

                    self.BUFFER = self._initial_buffer()

                    self.push_count = 0
                    state = self.env.reset()

                    EPISODE_REWARD = 0
                    index = 0
                else:
                    action, predictedValue = self.act(state)

                    #action = 0, do nothint
                    #action = 1, selling
                    #action = 2, buying

                    if action == 1:
                        selling.append(action)
                    elif action == 0:
                        zero.append(action)
                    else:
                        buying.append(action)

                    state_, reward, done, info = self.env.step(action, self.CURRENT_EPOCH)

                    earn_reward.append(reward)

                    if info != None:
                        funds.append(info[0])


                    self.BUFFER['candlestate'].append([state[0]])
                    self.BUFFER['portfolio'].append([state[1]])
                    self.BUFFER['volume_percent'].append(state[2])
                    self.BUFFER['time'].append(state[3])
                    self.BUFFER['actions'].append(action)
                    self.BUFFER['rewards'].append(reward)
                    self.BUFFER['predictedValues'].append(predictedValue)
                    self.BUFFER['done'].append(done)

                    state = state_
                    EPISODE_REWARD += reward
                    index += 1

                    if done:
                        self.BUFFER['episode_rewards'].append(EPISODE_REWARD)

                        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                        print "NUM_SELL-AC:", len(selling)
                        print "NUM_NOOP-AC:", len(zero)
                        print "NUM_BUYY-AC:", len(buying), "\n"

                        if len(earn_reward) > 0:
                            print "STEPS:", len(earn_reward), "\tTOTAL_SCORE: ", "%.2f" % np.array(earn_reward).sum()
                        else:
                            print "STEPS:", len(earn_reward), "\tMAX_E: ", "zero length"

                        if len(funds) > 0:
                            print "FUN_MAX:", "%.2f" % np.array(funds).max()
                            print "FUN_MIN:", "%.2f" % np.array(funds).min()
                            print "FUN_MEA:", "%.2f" % np.array(funds).mean()
                        else:
                            print "FUN_MAX:", "zero length"
                            print "FUN_MIN:", "zero length"
                            print "FUN_MEA:", "zero length"
                        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

                        earn_reward = []
                        selling = []
                        zero = []
                        buying = []
                        funds = []

                        state = self.env.reset()

                        EPISODE_REWARD = 0
            self.push_count += 1
            if (self.push_count == self.ENV_SAMPLE_ITERATIONS) :
                self.CURRENT_EPOCH += 1
            print self.name, "finished iterator", self.CURRENT_EPOCH, "\n"

            self.BUFFER['candlestate'] = list(self.BUFFER['candlestate'])
            self.BUFFER['portfolio'] = list(self.BUFFER['portfolio'])
            self.BUFFER['actions'] = list(self.BUFFER['actions'])
            self.BUFFER['rewards'] = list(self.BUFFER['rewards'])
            self.BUFFER['predictedValues'] = list(self.BUFFER['predictedValues'])
            self.BUFFER['done'] = list(self.BUFFER['done'])
            self.BUFFER['episode_rewards'] = list(self.BUFFER['episode_rewards'])


            if self.SCHEDULE == 'constant':
                self.current_learningRate = 1.0
            elif self.SCHEDULE == 'linear':
                self.current_learningRate = max(1.0 - float(self.CURRENT_EPOCH) / self.EPOCH_MAX, 0)

            buffer_done = np.append(self.BUFFER['done'], 0)
            buffer_predictedValues_tmp = np.append(self.BUFFER['predictedValues'], predictedValue * (1 - done))
            T = len(self.BUFFER['rewards'])
            buffer_advantage = np.empty(T, 'float32')
            lastgaelam = 0
            for t in reversed(range(T)):
                nonterminal = 1 - buffer_done[t + 1]
                delta = self.BUFFER['rewards'][t] + self.GAMMA * buffer_predictedValues_tmp[t + 1] * nonterminal - buffer_predictedValues_tmp[t]
                buffer_advantage[t] = lastgaelam = delta + self.GAMMA * self.LAM * nonterminal * lastgaelam
            buffer_estimatedReturn = np.add(buffer_advantage.tolist(), self.BUFFER['predictedValues'])
            #buffer_advantage = (buffer_advantage - buffer_advantage.mean()) / buffer_advantage.std()

            self.BUFFER['candlestate'] = np.vstack(self.BUFFER['candlestate'])
            self.BUFFER['actions'] = np.vstack(self.BUFFER['actions'])
            self.BUFFER['portfolio'] = np.vstack(self.BUFFER['portfolio'])

            buffer_estimatedReturn = np.vstack(buffer_estimatedReturn)
            buffer_advantage = np.vstack(buffer_advantage)

            batchs = deque()
            batchs.append(self.BUFFER['candlestate'])
            batchs.append(self.BUFFER['portfolio'])
            batchs.append(self.BUFFER['volume_percent'])
            batchs.append(self.BUFFER['time'])
            batchs.append(self.BUFFER['actions'])
            batchs.append(buffer_advantage)
            batchs.append(buffer_estimatedReturn)
            batchs.append(self.current_learningRate)


            feed_dict = {
                self.ppo.candlestate: self.BUFFER['candlestate'],
                self.ppo.portfolio_state: self.BUFFER['portfolio'],
                self.ppo.volume_percent: self.BUFFER['volume_percent'],
                self.ppo.time: self.BUFFER['time'],
                self.ppo.action: self.BUFFER['actions'],
                self.ppo.advantage: buffer_advantage,
                self.ppo.estimatedReturn: buffer_estimatedReturn,
                self.ppo.learningRate_multiplier: self.current_learningRate,
                self.ppo.phase: 0
            }

            print "Worker: updating tensorboard logs..."

            if (self.push_count == 1 and self.name == 'Worker_N0'):
                summary = self.sess.run(self.ppo.summary_op, feed_dict)
                log_writer.add_summary(summary, self.CURRENT_EPOCH)

            print "Worker: storing training loss logs..."
            queryItem = [self.ppo.policyLoss, self.ppo.valueLoss, self.ppo.entropyLoss, self.ppo.total_loss]
            policyLoss, valueLoss, entropyLoss, totalLoss = self.sess.run(queryItem, feed_dict)

            buffer_episode_rewards = np.array(self.BUFFER['episode_rewards'])


            logs = deque()

            if buffer_episode_rewards.size > 0:
                logs.append(buffer_episode_rewards.mean() / buffer_episode_rewards.std())
                logs.append(buffer_episode_rewards.min())
                logs.append(buffer_episode_rewards.max())
                logs.append(buffer_episode_rewards.mean())
            else:
                logs.append(0)
                logs.append(0)
                logs.append(0)
                logs.append(0)

            logs.append(policyLoss)
            logs.append(valueLoss)
            logs.append(entropyLoss)
            logs.append(totalLoss)
            logs.append(self.CURRENT_EPOCH)
            batchs.append(list(logs))

            if (len(buffer_episode_rewards) > 0):
                self.MEMORY_DICT[self.name].append(list(batchs))

            UPDATE_EVENT.set()

    def act(self, state):

        if (self.CURRENT_EPOCH >= self.AC_EXP_EPOCH):
            current_action_exploration_rate = self.MIN_AC_EXP_RATE
        else:
            current_action_exploration_rate = self.MAX_AC_EXP_RATE + self.CURRENT_EPOCH * (self.MIN_AC_EXP_RATE - self.MAX_AC_EXP_RATE) / self.AC_EXP_EPOCH

        feed_dict = {
            self.ppo.candlestate: [state[0]],
            self.ppo.phase: 0,
            self.ppo.portfolio_state: [state[1]],
            self.ppo.volume_percent: [state[2]],
            self.ppo.time: [state[3]]
        }

        action, predictedValue = self.sess.run([self.ppo.chooseAction, self.ppo.piPredictedValue], feed_dict)

        #print action, predictedValue

        if random.random() < current_action_exploration_rate:
            self.returnAction = random.randint(0, 2)
        else:
            self.returnAction = action[0][0][0]
        return self.returnAction, predictedValue[0][0][0]