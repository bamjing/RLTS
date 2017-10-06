# -*- coding: utf-8 -*-
import tensorflow as tf
from Module.RL.Others.distributions import make_pdtype
from Module.RL.Others import tf_util as U
from Module import Config

class Model(object):
    def __init__(self):
        config = Config.Configuration()
        parameter_dict = config.parameter_dict
        self.CNN_FILTERS = parameter_dict['CNN_FILTERS']
        self.DROP_RATE = parameter_dict['DROP_RATE']
        self.EMBEDDING_FEATURE_SIZE = parameter_dict['EMBEDDING_FEATURE_SIZE']
        self.NUM_FILTERS = parameter_dict['NUM_FILTERS']
        self.LOOK_BACK = parameter_dict['LOOK_BACK']
        self.RNN_SIZE = parameter_dict['RNN_SIZE']
        self.filter_sizes = [1, 4]
        self.num_filters = 20

    def LSTM_dynamic(self, scope, candlestate, volume_percent, time, portfolio_state, action_space, phase):
        with tf.variable_scope(scope):
            pdtype = make_pdtype(action_space)
            initializer = U.normc_initializer(1)

            #########################################################################################################

            candlestate_rnn_cell = tf.contrib.rnn.MultiRNNCell(self._get_lstm(self.DROP_RATE, 200, 2))
            with tf.variable_scope(scope + 'candle', reuse=False):
                candlestate_outputs, candlestate_states = tf.nn.dynamic_rnn(candlestate_rnn_cell, candlestate, dtype=tf.float32)

            candlestate_outputs = tf.transpose(candlestate_outputs, [1, 0, 2])
            candlestate_outputs = tf.gather(candlestate_outputs, int(candlestate_outputs.get_shape()[0]) - 1)
            candlestate_outputs = tf.expand_dims(candlestate_outputs, 1)

            #########################################################################################################
            #########################################################################################################

            candlestate_layer = tf.contrib.layers.batch_norm(candlestate_outputs, center=False, scale=False, is_training=phase)
            candlestate_layer = tf.layers.dense(candlestate_layer, 64, tf.nn.elu, kernel_initializer=initializer)

            candlestate_layer = tf.contrib.layers.batch_norm(candlestate_layer, center=False, scale=False, is_training=phase)
            candlestate_layer = tf.layers.dense(candlestate_layer, 64, tf.nn.elu, kernel_initializer=initializer)

            predictedValue_layer = tf.contrib.layers.batch_norm(candlestate_layer, center=False, scale=False, is_training=phase)
            predictedValue_layer = tf.layers.dense(predictedValue_layer, 64, tf.nn.elu, kernel_initializer=initializer)

            predictedValue_layer = tf.contrib.layers.batch_norm(predictedValue_layer, center=False, scale=False, is_training=phase)
            predictedValue_layer = tf.layers.dense(predictedValue_layer, 64, tf.nn.elu, kernel_initializer=initializer)
            predictedValue = tf.layers.dense(predictedValue_layer, 1, kernel_initializer=tf.zeros_initializer)

            logits_layer = tf.contrib.layers.batch_norm(candlestate_layer, center=False, scale=False, is_training=phase)
            logits_layer = tf.layers.dense(logits_layer, 64, tf.nn.elu, kernel_initializer=initializer)

            logits_layer = tf.contrib.layers.batch_norm(logits_layer, center=False, scale=False, is_training=phase)
            logits_layer = tf.layers.dense(logits_layer, 64, tf.nn.elu, kernel_initializer=initializer)
            logits = tf.layers.dense(logits_layer, 3, kernel_initializer=tf.zeros_initializer)
            pd = pdtype.pdfromflat(logits)

            #########################################################################################################

            variableMonitorList = {
                #'candlestate_layer_4 xx1': candlestate_layer_4,
                #'volume_percent_layer_4 xx2': volume_percent_layer_4,
                #'time_layer_4 xx3': time_layer_4,
                #'whole_state_layer_6 xx4': whole_state_layer_6,
                #'portfolio_state_layer_6 xx05': portfolio_state_layer_6,
                'portfolio_state xx00': portfolio_state,
                'logits xx6': logits,
                'predictedValue xx7': predictedValue,
                'volume_percent xx8': volume_percent,
                'time xx9': time

            }

        netParameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return predictedValue, pd, netParameters, variableMonitorList


    def CNN(self, scope, candlestate, volume_percent, time, portfolio_state, action_space, phase):

        with tf.variable_scope(scope):
            candlestate = tf.expand_dims(candlestate, 3)
            pdtype = make_pdtype(action_space)
            initializer = U.normc_initializer(1)
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                # Convolution Layer
                filter_shape = [filter_size, self.EMBEDDING_FEATURE_SIZE, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    candlestate,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID")
                # Apply nonlinearity

                h0 = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv, b), center=False, scale=False, is_training=phase)
                h = tf.nn.elu(h0, name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.LOOK_BACK - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID')
                pooled_outputs.append(pooled)

            # Combine all the pooled features
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat0 = tf.contrib.layers.flatten(h_pool)
            h_pool_flat = tf.expand_dims(h_pool_flat0, 1)

            #########################################################################################################
            #########################################################################################################

            candlestate_layer = tf.contrib.layers.batch_norm(h_pool_flat, center=False, scale=False, is_training=phase)
            candlestate_layer = tf.layers.dense(candlestate_layer, 64, tf.nn.elu, kernel_initializer=initializer)

            candlestate_layer = tf.contrib.layers.batch_norm(candlestate_layer, center=False, scale=False, is_training=phase)
            candlestate_layer = tf.layers.dense(candlestate_layer, 3, tf.nn.elu, kernel_initializer=initializer)

            #########################################################################################################

            volume_percent = tf.expand_dims(volume_percent, 1)
            volume_percent_layer = tf.layers.dense(volume_percent, 64, tf.nn.elu, kernel_initializer=initializer)

            volume_percent_layer = tf.contrib.layers.batch_norm(volume_percent_layer, center=False, scale=False, is_training=phase)
            volume_percent_layer = tf.layers.dense(volume_percent_layer, 3, tf.nn.elu, kernel_initializer=initializer)

            #########################################################################################################

            time = tf.expand_dims(time, 1)
            time_layer = tf.layers.dense(time, 64, tf.nn.elu, kernel_initializer=initializer)

            time_layer = tf.contrib.layers.batch_norm(time_layer, center=False, scale=False, is_training=phase)
            time_layer = tf.layers.dense(time_layer, 3, tf.nn.elu, kernel_initializer=initializer)

            #########################################################################################################

            portfolio_state = tf.expand_dims(portfolio_state, 1)
            portfolio_state_layer = tf.layers.dense(portfolio_state, 64, tf.nn.elu, kernel_initializer=initializer)

            portfolio_state_layer = tf.contrib.layers.batch_norm(portfolio_state_layer, center=False, scale=False, is_training=phase)
            portfolio_state_layer = tf.layers.dense(portfolio_state_layer, 3, tf.nn.elu, kernel_initializer=initializer)

            #########################################################################################################


            join_state = tf.concat([candlestate_layer, volume_percent_layer], axis=2)
            join_state = tf.concat([join_state, time_layer], axis=2)
            join_state = tf.concat([join_state, portfolio_state_layer], axis=2)

            #########################################################################################################


            whole_state_layer = tf.contrib.layers.batch_norm(join_state, center=False, scale=False, is_training=phase)
            whole_state_layer = tf.layers.dense(whole_state_layer, 64, tf.nn.elu, kernel_initializer=initializer)

            whole_state_layer = tf.contrib.layers.batch_norm(whole_state_layer, center=False, scale=False, is_training=phase)
            whole_state_layer = tf.layers.dense(whole_state_layer, 3, tf.nn.elu, kernel_initializer=initializer)

            #########################################################################################################


            #########################################################################################################
            #########################################################################################################

            predictedValue_layer = tf.contrib.layers.batch_norm(whole_state_layer, center=False, scale=False, is_training=phase)
            predictedValue_layer = tf.layers.dense(predictedValue_layer, 64, tf.nn.elu, kernel_initializer=initializer)

            predictedValue_layer = tf.contrib.layers.batch_norm(predictedValue_layer, center=False, scale=False, is_training=phase)
            predictedValue_layer = tf.layers.dense(predictedValue_layer, 64, tf.nn.elu, kernel_initializer=initializer)
            predictedValue = tf.layers.dense(predictedValue_layer, 1, kernel_initializer=tf.zeros_initializer)

            logits_layer = tf.contrib.layers.batch_norm(whole_state_layer, center=False, scale=False, is_training=phase)
            logits_layer = tf.layers.dense(logits_layer, 64, tf.nn.elu, kernel_initializer=initializer)

            logits_layer = tf.contrib.layers.batch_norm(logits_layer, center=False, scale=False, is_training=phase)
            logits_layer = tf.layers.dense(logits_layer, 64, tf.nn.elu, kernel_initializer=initializer)
            logits = tf.layers.dense(logits_layer, 3, kernel_initializer=tf.zeros_initializer)
            pd = pdtype.pdfromflat(logits)

            #########################################################################################################

            variableMonitorList = {
                'candlestate_layer xx1': candlestate_layer,
                #'volume_percent_layer xx2': volume_percent_layer,
                #'time_layer xx3': time_layer,
                # 'whole_state_layer_6 xx4': whole_state_layer_6,
                #'portfolio_state_layer xx05': portfolio_state_layer,
                'portfolio_state xx00': portfolio_state,
                'logits xx6': logits,
                'predictedValue xx7': predictedValue,
                'volume_percent xx8': volume_percent,
                'time xx9': time

            }

        netParameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return predictedValue, pd, netParameters, variableMonitorList



    def _get_lstm(self, prob, size, level):
        cells = []
        for _ in range(level):
            lstm = tf.contrib.rnn.BasicLSTMCell(size)
            out = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = prob)
            cells.append(out)
        return cells

    def MLP(self, scope, candlestate, volume_percent, time, portfolio_state, action_space, phase):
        with tf.variable_scope(scope):
            pdtype = make_pdtype(action_space)
            initializer = U.normc_initializer(0.01)

            # initializer = U.normc_initializer(1)

            # initializer = tf.random_normal_initializer(0.5, 1)
            # initializer = tf.zeros_initializer

            candlestate = tf.contrib.layers.flatten(candlestate)

            candlestate = tf.expand_dims(candlestate, 1)

            outputs = tf.layers.dense(candlestate, 16, tf.nn.relu, kernel_initializer=initializer)

            pre = tf.layers.dense(outputs, 8, tf.nn.relu, kernel_initializer=initializer)
            predictedValue = tf.layers.dense(pre, 1, kernel_initializer=initializer)

            log = tf.layers.dense(outputs, 8, tf.nn.relu, kernel_initializer=initializer)
            logits = tf.layers.dense(log, 3, kernel_initializer=tf.zeros_initializer)

            pd = pdtype.pdfromflat(logits)

            variableMonitorList = {
                'outputs xx07': outputs,
                #'finall1 xx09': finall1,
                #'finall2 xx10': finall2,
                #'finall3 xx11': finall3,
                #'l3 xx09': l3,
                #'actionDenseL1 xx10': actionDenseL1,
                'logits xx12': logits,
                #'policyDenseL1 xx13': policyDenseL1,
                #'policyDense xx14': policyDense,
                'predictedValue xx15': predictedValue
            }

        netParameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return predictedValue, pd, netParameters, variableMonitorList