# -*- coding: utf-8 -*-
import sys
from collections import deque

import numpy as np

from Module import trading_env
from Module.RL.PPO import PPO


class Chief(object):
    def __init__(self, scope, parameter_dict, SESS, MEMORY_DICT, COORD, workers):
        env = trading_env.make(SESS, parameter_dict)
        self.ppo = PPO(scope, parameter_dict, env, workers)
        self.sess = SESS
        self.MEMORY_DICT = MEMORY_DICT
        self.workers = workers
        self.EVO_NUM_WORKERS = np.ceil(parameter_dict['NUM_WORKERS'] * parameter_dict['EVOLUTION_RATE']).astype(np.int32)
        self.UPDATE_STEPS = parameter_dict['UPDATE_STEPS']
        self.COORD = COORD
        self.ENV_SAMPLE_ITERATIONS = parameter_dict['ENV_SAMPLE_ITERATIONS']

    def check(self, PUSH_EVENT, UPDATE_EVENT):
        while not self.COORD.should_stop():
            UPDATE_EVENT.wait()
            min_data_size, _ = self._get_data_size()
            if min_data_size >= self.ENV_SAMPLE_ITERATIONS:
                PUSH_EVENT.clear()
                self._train()
                self._update_local_workers_weight()
                PUSH_EVENT.set()
            UPDATE_EVENT.clear()

    def _train(self):
        print "Chief: updating neural network..."
        data_stack = deque()

        _, max_data_size= self._get_data_size()
        while max_data_size > 0:
            for key, value in self.MEMORY_DICT.items():
                if len(value) > 0:
                    value = list(value)

                    tmp = deque()
                    tmp.append(value[0][0])  # buffer_states
                    tmp.append(value[0][1])  # buffer_portfolio
                    tmp.append(value[0][2])  #volume_percent
                    tmp.append(value[0][3])  #time

                    tmp.append(value[0][4])  # buffer_actions
                    tmp.append(value[0][5])  # buffer_advantage
                    tmp.append(value[0][6])  # buffer_estimatedReturn
                    tmp.append(value[0][7])  # current_learningRate
                    tmp.append(value[0][8])  # buffer_score


                    self.MEMORY_DICT[key].popleft()
                    tmp = list(tmp)
                    data_stack.append(tmp)
            _, max_data_size = self._get_data_size()

        data_stack = list(data_stack)
        # [score, buffer_score.min(), buffer_score.max(), buffer_score.mean(), policyLoss, valueLoss, entropyLoss, totalLoss, self.TOTAL_STEP]
        data_stack = reversed(sorted(data_stack,key=lambda x: x[8][2]))
        data_stack = list(data_stack)

        feed_dict = {}

        learningRate_multiplier = data_stack[0][7]
        feed_dict[self.ppo.learningRate_multiplier] = learningRate_multiplier
        for i in range(self.EVO_NUM_WORKERS):
            feed_dict[self.workers[i].ppo.candlestate] = data_stack[i][0]
            feed_dict[self.workers[i].ppo.portfolio_state] = data_stack[i][1]
            feed_dict[self.workers[i].ppo.volume_percent] = data_stack[i][2]
            feed_dict[self.workers[i].ppo.time] = data_stack[i][3]
            feed_dict[self.workers[i].ppo.action] = data_stack[i][4]
            feed_dict[self.workers[i].ppo.advantage] = data_stack[i][5]
            feed_dict[self.workers[i].ppo.estimatedReturn] = data_stack[i][6]
            feed_dict[self.workers[i].ppo.learningRate_multiplier] = learningRate_multiplier
            feed_dict[self.workers[i].ppo.phase] = 1
        [self.sess.run(self.ppo.train, feed_dict=feed_dict) for _ in range(self.UPDATE_STEPS)]

        # [W, b, conv, h, pooled, cnn_out, outputs, sliced, l3, actionDenseL1, actionDense, logits, policyDenseL1, policyDense, predictedValue]
        monitor = self.sess.run(self.workers[0].ppo.variableM, feed_dict)

        for key, value in monitor.items():
            print key, ":", value[:3], value[-3:], np.array(value).shape, "\n"

        self._logs_writer(data_stack)


    def _update_local_workers_weight(self):
        for worker in self.workers:
            update_weight = [localp.assign(chiefp) for chiefp, localp in zip(self.ppo.piNetParameters, worker.ppo.piNetParameters)]
            self.sess.run(update_weight)

    def _get_data_size(self):
        min_data_size = sys.maxint
        max_data_size = -1
        for key, value in self.MEMORY_DICT.items():
            min_data_size = min(min_data_size, len(value))
            max_data_size = max(max_data_size, len(value))
        return min_data_size, max_data_size

    def _logs_writer(self, data_stack):
        # [score, buffer_score.min(), buffer_score.max(), buffer_score.mean(), policyLoss, valueLoss, entropyLoss, totalLoss, self.TOTAL_STEP]
        logs = []
        for item in data_stack:
            logs.append(item[8])

        score_detail = []
        minss = []
        maxss = []
        meanss = []
        policyLossQQQ = []
        valueLossQQQ = []
        entropyLossQQQ = []
        totalLoss = []

        for log in logs:
            score_detail.append(log[0])
            minss.append(log[1])
            maxss.append(log[2])
            meanss.append(log[3])
            policyLossQQQ.append(log[4])
            valueLossQQQ.append(log[5])
            entropyLossQQQ.append(log[6])
            totalLoss.append(log[7])

        print "############################# begin ###############################################"
        print "score", '(%.2f, %.2f)' % (np.array(score_detail).min(), np.array(score_detail).max()), score_detail, "\n"
        print "min", '(%.2f, %.2f)' % (np.array(minss).min(), np.array(minss).max()), minss, "\n"
        print "max", '(%.2f, %.2f)' % (np.array(maxss).min(), np.array(maxss).max()), maxss, "\n"
        print "mean", '(%.2f, %.2f)' % (np.array(meanss).min(), np.array(meanss).max()), meanss, "\n"
        print "totalLoss", '(%.2f, %.2f)' % (np.array(totalLoss).min(), np.array(totalLoss).max()), totalLoss, "\n"
        print "policyLoss", '(%.2f, %.2f)' % (np.array(policyLossQQQ).min(), np.array(policyLossQQQ).max()), policyLossQQQ, "\n"
        print "valueLoss", '(%.2f, %.2f)' % (np.array(valueLossQQQ).min(), np.array(valueLossQQQ).max()), valueLossQQQ, "\n"
        print "entropyLoss", '(%.2f, %.2f)' % (np.array(entropyLossQQQ).min(), np.array(entropyLossQQQ).max()), entropyLossQQQ, "\n"
        print "############################## finished ############################################"

        #if np.array(minss).max() >= 200 and np.array(maxss).min() >= 200 and np.array(totalLoss).min() <= 0.7:
            #self.COORD.request_stop()

    def act(self, state):
        state = np.array([state])
        action = self.sess.run(self.ppo.chooseAction, {self.ppo.candlestate: [state]})
        return action[0][0]