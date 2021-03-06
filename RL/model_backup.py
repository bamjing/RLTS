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
        self.num_filters = 200

    def LSTM_dynamic(self, scope, candlestate, volume_percent, time, portfolio_state, action_space, phase):
        with tf.variable_scope(scope):
            pdtype = make_pdtype(action_space)
            initializer = U.normc_initializer(1)

            #########################################################################################################

            candlestate_rnn_cell = tf.contrib.rnn.MultiRNNCell(self._get_lstm(self.DROP_RATE, 250, 5))
            with tf.variable_scope(scope + 'candle', reuse=False):
                candlestate_outputs, candlestate_states = tf.nn.dynamic_rnn(candlestate_rnn_cell, candlestate, dtype=tf.float32)

            candlestate_outputs = tf.transpose(candlestate_outputs, [1, 0, 2])
            candlestate_outputs = tf.gather(candlestate_outputs, int(candlestate_outputs.get_shape()[0]) - 1)
            candlestate_outputs = tf.expand_dims(candlestate_outputs, 1)

            #########################################################################################################
            #########################################################################################################

            candlestate_layer_1 = tf.contrib.layers.batch_norm(candlestate_outputs, center=False, scale=False, is_training=phase)
            candlestate_layer_2 = tf.layers.dense(candlestate_layer_1, 64, tf.nn.elu, kernel_initializer=initializer)

            candlestate_layer_3 = tf.contrib.layers.batch_norm(candlestate_layer_2, center=False, scale=False, is_training=phase)
            candlestate_layer_4 = tf.layers.dense(candlestate_layer_3, 64, tf.nn.elu, kernel_initializer=initializer)

            #########################################################################################################

            volume_percent_layer_1 = tf.contrib.layers.batch_norm(volume_percent, center=False, scale=False, is_training=phase)
            volume_percent_layer_2 = tf.layers.dense(volume_percent_layer_1, 64, tf.nn.elu, kernel_initializer=initializer)

            volume_percent_layer_3 = tf.contrib.layers.batch_norm(volume_percent_layer_2, center=False, scale=False, is_training=phase)
            volume_percent_layer_4 = tf.layers.dense(volume_percent_layer_3, 64, tf.nn.elu, kernel_initializer=initializer)

            #########################################################################################################

            time_layer_1 = tf.contrib.layers.batch_norm(time, center=False, scale=False, is_training=phase)
            time_layer_2 = tf.layers.dense(time_layer_1, 64, tf.nn.elu, kernel_initializer=initializer)

            time_layer_3 = tf.contrib.layers.batch_norm(time_layer_2, center=False, scale=False, is_training=phase)
            time_layer_4 = tf.layers.dense(time_layer_3, 64, tf.nn.elu, kernel_initializer=initializer)

            #########################################################################################################


            join_state = tf.concat([candlestate_layer_4, tf.expand_dims(volume_percent_layer_4, 1)], axis=2)
            join_state = tf.concat([join_state, tf.expand_dims(time_layer_4, 1)], axis=2)

            #########################################################################################################

            whole_state_layer_1 = tf.contrib.layers.batch_norm(join_state, center=False, scale=False, is_training=phase)
            whole_state_layer_2 = tf.layers.dense(whole_state_layer_1, 64, tf.nn.elu, kernel_initializer=initializer)

            whole_state_layer_3 = tf.contrib.layers.batch_norm(whole_state_layer_2, center=False, scale=False, is_training=phase)
            whole_state_layer_4 = tf.layers.dense(whole_state_layer_3, 64, tf.nn.elu, kernel_initializer=initializer)

            whole_state_layer_5 = tf.contrib.layers.batch_norm(whole_state_layer_4, center=False, scale=False, is_training=phase)
            whole_state_layer_6 = tf.layers.dense(whole_state_layer_5, 64, tf.nn.elu, kernel_initializer=initializer)

            #########################################################################################################
            #########################################################################################################

            portfolio_state_layer_1 = tf.contrib.layers.batch_norm(portfolio_state, center=False, scale=False, is_training=phase)
            portfolio_state_layer_2 = tf.layers.dense(portfolio_state_layer_1, 64, tf.nn.elu, kernel_initializer=initializer)

            portfolio_state_layer_3 = tf.contrib.layers.batch_norm(portfolio_state_layer_2, center=False, scale=False, is_training=phase)
            portfolio_state_layer_4 = tf.layers.dense(portfolio_state_layer_3, 64, tf.nn.elu, kernel_initializer=initializer)

            portfolio_state_layer_5 = tf.contrib.layers.batch_norm(portfolio_state_layer_4, center=False, scale=False, is_training=phase)
            portfolio_state_layer_6 = tf.layers.dense(portfolio_state_layer_5, 64, tf.nn.elu, kernel_initializer=initializer)

            #########################################################################################################

            join_market_portfolio = tf.concat([whole_state_layer_6, tf.expand_dims(portfolio_state_layer_6, 1)], axis=2)

            #########################################################################################################

            l1_w = tf.contrib.layers.batch_norm(join_market_portfolio, center=False, scale=False, is_training=phase)
            l2_w = tf.layers.dense(l1_w, 64, tf.nn.elu, kernel_initializer=initializer)

            l3_w = tf.contrib.layers.batch_norm(l2_w, center=False, scale=False, is_training=phase)
            l4_w = tf.layers.dense(l3_w, 64, tf.nn.elu, kernel_initializer=initializer)

            pre1 = tf.contrib.layers.batch_norm(l4_w, center=False, scale=False, is_training=phase)
            pre2 = tf.layers.dense(pre1, 64, tf.nn.elu, kernel_initializer=initializer)

            pre3 = tf.contrib.layers.batch_norm(pre2, center=False, scale=False, is_training=phase)
            pre4 = tf.layers.dense(pre3, 64, tf.nn.elu, kernel_initializer=initializer)
            predictedValue = tf.layers.dense(pre4, 1, kernel_initializer=initializer)

            log1 = tf.contrib.layers.batch_norm(l4_w, center=False, scale=False, is_training=phase)
            log2 = tf.layers.dense(log1, 64, tf.nn.elu, kernel_initializer=initializer)

            log3 = tf.contrib.layers.batch_norm(log2, center=False, scale=False, is_training=phase)
            log4 = tf.layers.dense(log3, 64, tf.nn.elu, kernel_initializer=initializer)
            logits = tf.layers.dense(log4, 3, kernel_initializer=initializer)
            pd = pdtype.pdfromflat(logits)

            #########################################################################################################

            variableMonitorList = {
                #'candlestate_layer_4 xx1': candlestate_layer_4,
                'volume_percent_layer_4 xx2': volume_percent_layer_4,
                'time_layer_4 xx3': time_layer_4,
                #'whole_state_layer_6 xx4': whole_state_layer_6,
                'portfolio_state_layer_6 xx05': portfolio_state_layer_6,
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
                h = tf.nn.relu(h0, name="relu")
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

            candlestate_layer_1 = tf.contrib.layers.batch_norm(h_pool_flat, center=False, scale=False, is_training=phase)
            candlestate_layer_2 = tf.layers.dense(candlestate_layer_1, 64, tf.nn.relu, kernel_initializer=initializer)

            candlestate_layer_3 = tf.contrib.layers.batch_norm(candlestate_layer_2, center=False, scale=False, is_training=phase)
            candlestate_layer_4 = tf.layers.dense(candlestate_layer_3, 64, tf.nn.relu, kernel_initializer=initializer)

            candlestate_layer_5 = tf.contrib.layers.batch_norm(candlestate_layer_4, center=False, scale=False, is_training=phase)
            candlestate_layer_6 = tf.layers.dense(candlestate_layer_5, 64, tf.nn.relu, kernel_initializer=initializer)

            candlestate_layer_7 = tf.contrib.layers.batch_norm(candlestate_layer_6, center=False, scale=False, is_training=phase)
            candlestate_layer_8 = tf.layers.dense(candlestate_layer_7, 64, tf.nn.relu, kernel_initializer=initializer)

            candlestate_layer_9 = tf.contrib.layers.batch_norm(candlestate_layer_8, center=False, scale=False, is_training=phase)
            candlestate_layer_10 = tf.layers.dense(candlestate_layer_9, 64, tf.nn.relu, kernel_initializer=initializer)

            candlestate_layer_11 = tf.contrib.layers.batch_norm(candlestate_layer_10, center=False, scale=False, is_training=phase)
            candlestate_layer_12 = tf.layers.dense(candlestate_layer_11, 64, tf.nn.relu, kernel_initializer=initializer)

            candlestate_layer_13 = tf.contrib.layers.batch_norm(candlestate_layer_12, center=False, scale=False, is_training=phase)
            candlestate_layer_14 = tf.layers.dense(candlestate_layer_13, 64, tf.nn.relu, kernel_initializer=initializer)

            candlestate_layer_15 = tf.contrib.layers.batch_norm(candlestate_layer_14, center=False, scale=False, is_training=phase)
            candlestate_layer_16 = tf.layers.dense(candlestate_layer_15, 64, tf.nn.relu, kernel_initializer=initializer)

            candlestate_layer_17 = tf.contrib.layers.batch_norm(candlestate_layer_16, center=False, scale=False, is_training=phase)
            candlestate_layer_18 = tf.layers.dense(candlestate_layer_17, 64, tf.nn.relu, kernel_initializer=initializer)

            candlestate_layer_19 = tf.contrib.layers.batch_norm(candlestate_layer_18, center=False, scale=False, is_training=phase)
            candlestate_layer_20 = tf.layers.dense(candlestate_layer_19, 64, tf.nn.relu, kernel_initializer=initializer)

            #########################################################################################################

            volume_percent_layer_1 = tf.contrib.layers.batch_norm(volume_percent, center=False, scale=False, is_training=phase)
            volume_percent_layer_2 = tf.layers.dense(volume_percent_layer_1, 64, tf.nn.relu, kernel_initializer=initializer)

            volume_percent_layer_3 = tf.contrib.layers.batch_norm(volume_percent_layer_2, center=False, scale=False, is_training=phase)
            volume_percent_layer_4 = tf.layers.dense(volume_percent_layer_3, 64, tf.nn.relu, kernel_initializer=initializer)

            #########################################################################################################

            time_layer_1 = tf.contrib.layers.batch_norm(time, center=False, scale=False, is_training=phase)
            time_layer_2 = tf.layers.dense(time_layer_1, 64, tf.nn.relu, kernel_initializer=initializer)

            time_layer_3 = tf.contrib.layers.batch_norm(time_layer_2, center=False, scale=False, is_training=phase)
            time_layer_4 = tf.layers.dense(time_layer_3, 64, tf.nn.relu, kernel_initializer=initializer)

            #########################################################################################################


            join_state = tf.concat([candlestate_layer_20, tf.expand_dims(volume_percent_layer_4, 1)], axis=2)
            join_state = tf.concat([join_state, tf.expand_dims(time_layer_4, 1)], axis=2)

            #########################################################################################################

            whole_state_layer_1 = tf.contrib.layers.batch_norm(join_state, center=False, scale=False, is_training=phase)
            whole_state_layer_2 = tf.layers.dense(whole_state_layer_1, 64, tf.nn.relu, kernel_initializer=initializer)

            whole_state_layer_3 = tf.contrib.layers.batch_norm(whole_state_layer_2, center=False, scale=False, is_training=phase)
            whole_state_layer_4 = tf.layers.dense(whole_state_layer_3, 64, tf.nn.relu, kernel_initializer=initializer)

            whole_state_layer_5 = tf.contrib.layers.batch_norm(whole_state_layer_4, center=False, scale=False, is_training=phase)
            whole_state_layer_6 = tf.layers.dense(whole_state_layer_5, 64, tf.nn.relu, kernel_initializer=initializer)

            whole_state_layer_7 = tf.contrib.layers.batch_norm(whole_state_layer_6, center=False, scale=False, is_training=phase)
            whole_state_layer_8 = tf.layers.dense(whole_state_layer_7, 64, tf.nn.relu, kernel_initializer=initializer)

            whole_state_layer_9 = tf.contrib.layers.batch_norm(whole_state_layer_8, center=False, scale=False, is_training=phase)
            whole_state_layer_10 = tf.layers.dense(whole_state_layer_9, 64, tf.nn.relu, kernel_initializer=initializer)

            whole_state_layer_11 = tf.contrib.layers.batch_norm(whole_state_layer_10, center=False, scale=False, is_training=phase)
            whole_state_layer_12 = tf.layers.dense(whole_state_layer_11, 64, tf.nn.relu, kernel_initializer=initializer)

            whole_state_layer_13 = tf.contrib.layers.batch_norm(whole_state_layer_12, center=False, scale=False, is_training=phase)
            whole_state_layer_14 = tf.layers.dense(whole_state_layer_13, 64, tf.nn.relu, kernel_initializer=initializer)

            whole_state_layer_15 = tf.contrib.layers.batch_norm(whole_state_layer_14, center=False, scale=False, is_training=phase)
            whole_state_layer_16 = tf.layers.dense(whole_state_layer_15, 64, tf.nn.relu, kernel_initializer=initializer)

            whole_state_layer_17 = tf.contrib.layers.batch_norm(whole_state_layer_16, center=False, scale=False, is_training=phase)
            whole_state_layer_18 = tf.layers.dense(whole_state_layer_17, 64, tf.nn.relu, kernel_initializer=initializer)

            whole_state_layer_19 = tf.contrib.layers.batch_norm(whole_state_layer_18, center=False, scale=False, is_training=phase)
            whole_state_layer_20 = tf.layers.dense(whole_state_layer_19, 64, tf.nn.relu, kernel_initializer=initializer)

            #########################################################################################################
            #########################################################################################################

            portfolio_state_layer_1 = tf.contrib.layers.batch_norm(portfolio_state, center=False, scale=False, is_training=phase)
            portfolio_state_layer_2 = tf.layers.dense(portfolio_state_layer_1, 64, tf.nn.relu, kernel_initializer=initializer)

            portfolio_state_layer_3 = tf.contrib.layers.batch_norm(portfolio_state_layer_2, center=False, scale=False, is_training=phase)
            portfolio_state_layer_4 = tf.layers.dense(portfolio_state_layer_3, 64, tf.nn.relu, kernel_initializer=initializer)

            portfolio_state_layer_5 = tf.contrib.layers.batch_norm(portfolio_state_layer_4, center=False, scale=False, is_training=phase)
            portfolio_state_layer_6 = tf.layers.dense(portfolio_state_layer_5, 64, tf.nn.relu, kernel_initializer=initializer)

            portfolio_state_layer_7 = tf.contrib.layers.batch_norm(portfolio_state_layer_6, center=False, scale=False, is_training=phase)
            portfolio_state_layer_8 = tf.layers.dense(portfolio_state_layer_7, 64, tf.nn.relu, kernel_initializer=initializer)

            portfolio_state_layer_9 = tf.contrib.layers.batch_norm(portfolio_state_layer_8, center=False, scale=False, is_training=phase)
            portfolio_state_layer_10 = tf.layers.dense(portfolio_state_layer_9, 64, tf.nn.relu, kernel_initializer=initializer)

            portfolio_state_layer_11 = tf.contrib.layers.batch_norm(portfolio_state_layer_10, center=False, scale=False, is_training=phase)
            portfolio_state_layer_12 = tf.layers.dense(portfolio_state_layer_11, 64, tf.nn.relu, kernel_initializer=initializer)

            #########################################################################################################

            join_market_portfolio = tf.concat([whole_state_layer_20, tf.expand_dims(portfolio_state_layer_12, 1)], axis=2)

            #########################################################################################################

            l1_w = tf.contrib.layers.batch_norm(join_market_portfolio, center=False, scale=False, is_training=phase)
            l2_w = tf.layers.dense(l1_w, 64, tf.nn.relu, kernel_initializer=initializer)

            l3_w = tf.contrib.layers.batch_norm(l2_w, center=False, scale=False, is_training=phase)
            l4_w = tf.layers.dense(l3_w, 64, tf.nn.relu, kernel_initializer=initializer)

            pre1 = tf.contrib.layers.batch_norm(l4_w, center=False, scale=False, is_training=phase)
            pre2 = tf.layers.dense(pre1, 64, tf.nn.relu, kernel_initializer=initializer)

            pre3 = tf.contrib.layers.batch_norm(pre2, center=False, scale=False, is_training=phase)
            pre4 = tf.layers.dense(pre3, 64, tf.nn.relu, kernel_initializer=initializer)
            predictedValue = tf.layers.dense(pre4, 1, kernel_initializer=initializer)

            log1 = tf.contrib.layers.batch_norm(l4_w, center=False, scale=False, is_training=phase)
            log2 = tf.layers.dense(log1, 64, tf.nn.relu, kernel_initializer=initializer)

            log3 = tf.contrib.layers.batch_norm(log2, center=False, scale=False, is_training=phase)
            log4 = tf.layers.dense(log3, 64, tf.nn.relu, kernel_initializer=initializer)
            logits = tf.layers.dense(log4, 3, kernel_initializer=initializer)
            pd = pdtype.pdfromflat(logits)

            #########################################################################################################

            variableMonitorList = {
                'candlestate_layer_20 xx1': candlestate_layer_20,
                'volume_percent_layer_4 xx2': volume_percent_layer_4,
                'time_layer_4 xx3': time_layer_4,
                # 'whole_state_layer_6 xx4': whole_state_layer_6,
                'portfolio_state_layer_6 xx05': portfolio_state_layer_6,
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

    def MLP(self, scope, candlestate, action_space):
        with tf.variable_scope(scope):
            pdtype = make_pdtype(action_space)
            initializer = U.normc_initializer(0.01)

            candlestate = tf.contrib.layers.flatten(candlestate)

            candlestate = tf.expand_dims(candlestate, 1)

            outputs = tf.layers.dense(candlestate, 128, tf.nn.relu, kernel_initializer=initializer)
            outputs = tf.layers.dense(outputs, 128, tf.nn.relu, kernel_initializer=initializer)
            outputs = tf.layers.dense(outputs, 128, tf.nn.relu, kernel_initializer=initializer)

            pre = tf.layers.dense(outputs, 64, tf.nn.relu, kernel_initializer=initializer)
            pre = tf.layers.dense(pre, 64, tf.nn.relu, kernel_initializer=initializer)
            pre = tf.layers.dense(pre, 32, tf.nn.relu, kernel_initializer=initializer)
            predictedValue = tf.layers.dense(pre, 1, kernel_initializer=initializer)

            log = tf.layers.dense(outputs, 64, tf.nn.relu, kernel_initializer=initializer)
            log = tf.layers.dense(log, 64, tf.nn.relu, kernel_initializer=initializer)
            log = tf.layers.dense(log, 32, tf.nn.relu, kernel_initializer=initializer)
            logits = tf.layers.dense(log, 3, kernel_initializer=initializer)

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
                h = tf.nn.relu(h0, name="relu")
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
            candlestate_layer = tf.layers.dense(candlestate_layer, 64, tf.nn.relu, kernel_initializer=initializer)

            candlestate_layer = tf.contrib.layers.batch_norm(candlestate_layer, center=False, scale=False, is_training=phase)
            candlestate_layer = tf.layers.dense(candlestate_layer, 64, tf.nn.relu, kernel_initializer=initializer)

            #########################################################################################################

            volume_percent_layer = tf.contrib.layers.batch_norm(volume_percent, center=False, scale=False, is_training=phase)
            volume_percent_layer = tf.layers.dense(volume_percent_layer, 64, tf.nn.relu, kernel_initializer=initializer)

            volume_percent_layer = tf.contrib.layers.batch_norm(volume_percent_layer, center=False, scale=False, is_training=phase)
            volume_percent_layer = tf.layers.dense(volume_percent_layer, 64, tf.nn.relu, kernel_initializer=initializer)

            #########################################################################################################

            time_layer = tf.contrib.layers.batch_norm(time, center=False, scale=False, is_training=phase)
            time_layer = tf.layers.dense(time_layer, 64, tf.nn.relu, kernel_initializer=initializer)

            time_layer = tf.contrib.layers.batch_norm(time_layer, center=False, scale=False, is_training=phase)
            time_layer = tf.layers.dense(time_layer, 64, tf.nn.relu, kernel_initializer=initializer)

            #########################################################################################################


            join_state = tf.concat([candlestate_layer, tf.expand_dims(volume_percent_layer, 1)], axis=2)
            join_state = tf.concat([join_state, tf.expand_dims(time_layer, 1)], axis=2)

            #########################################################################################################

            whole_state_layer = tf.contrib.layers.batch_norm(join_state, center=False, scale=False, is_training=phase)
            whole_state_layer = tf.layers.dense(whole_state_layer, 64, tf.nn.relu, kernel_initializer=initializer)

            whole_state_layer = tf.contrib.layers.batch_norm(whole_state_layer, center=False, scale=False, is_training=phase)
            whole_state_layer = tf.layers.dense(whole_state_layer, 64, tf.nn.relu, kernel_initializer=initializer)

            #########################################################################################################
            #########################################################################################################

            portfolio_state_layer = tf.contrib.layers.batch_norm(portfolio_state, center=False, scale=False, is_training=phase)
            portfolio_state_layer = tf.layers.dense(portfolio_state_layer, 64, tf.nn.relu, kernel_initializer=initializer)

            portfolio_state_layer = tf.contrib.layers.batch_norm(portfolio_state_layer, center=False, scale=False, is_training=phase)
            portfolio_state_layer = tf.layers.dense(portfolio_state_layer, 64, tf.nn.relu, kernel_initializer=initializer)

            #########################################################################################################

            join_market_portfolio = tf.concat([whole_state_layer, tf.expand_dims(portfolio_state_layer, 1)], axis=2)

            #########################################################################################################

            final_layer = tf.contrib.layers.batch_norm(join_market_portfolio, center=False, scale=False, is_training=phase)
            final_layer = tf.layers.dense(final_layer, 64, tf.nn.relu, kernel_initializer=initializer)

            final_layer = tf.contrib.layers.batch_norm(final_layer, center=False, scale=False, is_training=phase)
            final_layer = tf.layers.dense(final_layer, 64, tf.nn.relu, kernel_initializer=initializer)

            predictedValue_layer = tf.contrib.layers.batch_norm(final_layer, center=False, scale=False, is_training=phase)
            predictedValue_layer = tf.layers.dense(predictedValue_layer, 64, tf.nn.relu, kernel_initializer=initializer)

            predictedValue_layer = tf.contrib.layers.batch_norm(predictedValue_layer, center=False, scale=False, is_training=phase)
            predictedValue_layer = tf.layers.dense(predictedValue_layer, 64, tf.nn.relu, kernel_initializer=initializer)
            predictedValue = tf.layers.dense(predictedValue_layer, 1, kernel_initializer=initializer)

            logits_layer = tf.contrib.layers.batch_norm(final_layer, center=False, scale=False, is_training=phase)
            logits_layer = tf.layers.dense(logits_layer, 64, tf.nn.relu, kernel_initializer=initializer)

            logits_layer = tf.contrib.layers.batch_norm(logits_layer, center=False, scale=False, is_training=phase)
            logits_layer = tf.layers.dense(logits_layer, 64, tf.nn.relu, kernel_initializer=initializer)
            logits = tf.layers.dense(logits_layer, 3, kernel_initializer=tf.zeros_initializer)
            pd = pdtype.pdfromflat(logits)

            #########################################################################################################

            variableMonitorList = {
                'candlestate_layer xx1': candlestate_layer,
                'volume_percent_layer xx2': volume_percent_layer,
                'time_layer xx3': time_layer,
                # 'whole_state_layer_6 xx4': whole_state_layer_6,
                'portfolio_state_layer xx05': portfolio_state_layer,
                'portfolio_state xx00': portfolio_state,
                'logits xx6': logits,
                'predictedValue xx7': predictedValue,
                'volume_percent xx8': volume_percent,
                'time xx9': time

            }

        netParameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return predictedValue, pd, netParameters, variableMonitorList