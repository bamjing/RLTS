# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from Module.RL.Model import Model
from Module.RL.Others import tf_util as U

class PPO(object):
    def __init__(self, scope, parameter_dict, env, workerLists = None):

        LEARNING_RATE = parameter_dict['LEARNING_RATE']
        CLIP_PARAM = parameter_dict['CLIP_PARAM']
        ENTCOEFF = parameter_dict['ENTCOEFF']
        VCOEFF = parameter_dict['VCOEFF']
        EMBEDDING_FEATURE_SIZE = parameter_dict['EMBEDDING_FEATURE_SIZE']
        LOOK_BACK = parameter_dict['LOOK_BACK']
        MOMENTUM = parameter_dict['MOMENTUM']
        EVO_NUM_WORKERS = np.ceil(parameter_dict['NUM_WORKERS'] * parameter_dict['EVOLUTION_RATE']).astype(np.int32)

        observation_space = env.observation_space
        action_space = env.action_space
        model = Model()

        with tf.name_scope("FEED_STATE"):
            self.candlestate = tf.placeholder(tf.float32, [None, LOOK_BACK, EMBEDDING_FEATURE_SIZE])

        with tf.name_scope("FEED_PORTFOLIO_STATE"):
            self.portfolio_state = tf.placeholder(tf.float32, [None, 4])

        with tf.name_scope("FEED_VOLUME_PERCENT"):
            self.volume_percent = tf.placeholder(tf.float32, [None, 1])

        with tf.name_scope("FEED_TIME"):
            self.time = tf.placeholder(tf.float32, [None, 1])


        """
        1. how to fit the amazon environment quickly.
        2. if there is benifit of health insurance.
        3. is there any place in amazon to let employee continue improve their skill in spare time.
        4. do i have to do the code review myself or there is a code review team.
        5. I want to know how amazon internally improve the software engineer's code quality.
        6. I hope to know the tipycal project development cycle time in amazon from idea to actual products.
        7. I hope to know if the new graduate software engineer can have the opportunity to invole into the product implementation which will enfluence millions of users.

        8. I hope to know 
        """


        with tf.name_scope("FEED_CLIPPING_PARAMETER"):
            self.learningRate_multiplier = tf.placeholder(name='learningRate_multiplier', dtype=tf.float32, shape=[])
            CLIP_PARAM = CLIP_PARAM * self.learningRate_multiplier

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
             Optimizer = tf.train.AdamOptimizer(LEARNING_RATE * self.learningRate_multiplier)
            #Optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE * self.learningRate_multiplier)
            #Optimizer = tf.train.MomentumOptimizer(LEARNING_RATE * self.learningRate_multiplier, MOMENTUM, use_nesterov=False)


        with tf.name_scope("FEED_PHASE"):
            self.phase = tf.placeholder(tf.bool, name='phase')

        self.piPredictedValue, piPD, self.piNetParameters, self.variableM = model.CNN(scope + 'pi', self.candlestate, self.volume_percent, self.time, self.portfolio_state, action_space, self.phase)
        oldpiPredictedValue, oldpiPD, self.oldpiNetParameters, _ = model.CNN(scope + 'oldpi', self.candlestate, self.volume_percent, self.time, self.portfolio_state, action_space, self.phase)
        self.chooseAction = piPD.sample()


        if scope != "Chief":
            with tf.name_scope("FEED_ACTION"):
                self.action = tf.placeholder(tf.int32, [None, 1])
            with tf.name_scope("FEED_ADVANTAGE"):
                self.advantage = tf.placeholder(tf.float32, [None, 1])
            with tf.name_scope("FEED_ESTIMATED_RETURN"):
                self.estimatedReturn = tf.placeholder(tf.float32, [None, 1])
            with tf.name_scope("ENTROPYLOSS"):
                kloldnew = oldpiPD.kl(piPD)
                ent = piPD.entropy()
                meankl = U.mean(kloldnew)
                meanent = U.mean(ent)
                self.entropyLoss = (-ENTCOEFF) * meanent

            with tf.name_scope("POLICYLOSS"):
                ratio = tf.exp(piPD.logp(self.action) - oldpiPD.logp(self.action))  # pnew / pold
                surr1 = ratio * self.advantage  # surrogate from conservative policy iteration
                surr2 = U.clip(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * self.advantage
                self.policyLoss = - U.mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

            with tf.name_scope("VALUELOSS"):
                vfloss1 = tf.square(self.piPredictedValue - self.estimatedReturn)
                vpredclipped = oldpiPredictedValue + tf.clip_by_value(self.piPredictedValue - oldpiPredictedValue, -CLIP_PARAM, CLIP_PARAM)
                vfloss2 = tf.square(vpredclipped - self.estimatedReturn)
                #self.valueLoss = VCOEFF * U.mean(tf.maximum(vfloss1, vfloss2))
                self.valueLoss = VCOEFF * U.mean(tf.square(self.piPredictedValue- self.estimatedReturn))

            with tf.name_scope("TOTALLOSS"):
                self.total_loss = self.policyLoss + self.entropyLoss + self.valueLoss

            with tf.name_scope("LOGS"):
                tf.summary.scalar("entropyLoss", self.entropyLoss)
                tf.summary.scalar("policyLoss", self.policyLoss)
                tf.summary.scalar("valueLoss", self.valueLoss)
                tf.summary.scalar("total_loss", self.total_loss)
                self.summary_op = tf.summary.merge_all()

            with tf.name_scope("CALCULATE_GRADIENTS"):
                with tf.control_dependencies(update_ops):
                    self.gradient = Optimizer.compute_gradients(self.total_loss, self.piNetParameters)


            with tf.name_scope("SYNC"):
                self.sync_pis = [oldp.assign(p) for p, oldp in zip(self.piNetParameters, self.oldpiNetParameters)]
        else:
            gradientList = []
            for i in range(EVO_NUM_WORKERS):
                gradientList.append(workerLists[i].ppo.gradient)

            averaged_gradients = self._average_gradients(gradientList)
            with tf.name_scope("TRAIN_WITH_AVERAGE_GRADIENTS"):
                with tf.control_dependencies(update_ops):
                    self.train = Optimizer.apply_gradients(zip(averaged_gradients, self.piNetParameters))

    def _average_gradients(self, tower_grads):

        average_grads = []
        for grad_and_vars in zip(*tower_grads):

            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, 0)
            average_grads.append(grad)
        return average_grads