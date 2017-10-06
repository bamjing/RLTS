# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from pandas import read_csv
from Module.RL.Others import tf_util as U
from Module import Config
import threading, os, time

seed = 7
np.random.seed(seed)

class Classifier(object):
    def __init__(self, para_dict):
        self.NUM_CLUSTERING = para_dict['NUM_CLUSTERING']
        self.ACCURACY_TARGET = para_dict['ACCURACY_TARGET']
        self.BATCH_SIZE = para_dict['BATCH_SIZE']
        self.input = tf.placeholder(tf.float32, [None, 3], name="CLASSIFICATION_INPUT")

        self.label = tf.placeholder(tf.int64, [None, 1])
        logits, self.prediction = self._model()

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=logits)
        self.cost = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        # Check prediction
        correct = tf.equal(self.label, self.prediction)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def _model(self):
        initializer = U.normc_initializer(0.01)
        input = tf.expand_dims(self.input, axis=1)
        l1 = tf.layers.dense(input, 64, tf.nn.relu, kernel_initializer=initializer)
        l2 = tf.layers.dense(l1, 64, tf.nn.relu, kernel_initializer=initializer)
        logits = tf.layers.dense(l2, self.NUM_CLUSTERING, kernel_initializer=initializer)
        prediction = tf.argmax(logits, axis=2, name='CLASSIFICATION_PRED')
        return logits, prediction

    def train(self, data, COORD):
        epoch = 0
        training_target = 0
        while not COORD.should_stop():
            epoch_loss = 0
            epoch_finished = False
            while not epoch_finished:
                epoch_finished, batch_x, batch_y = data.next_batch(self.BATCH_SIZE)
                _, cost = sess.run([self.optimizer, self.cost], feed_dict={self.input: batch_x, self.label: batch_y})
                epoch_loss += cost
            epoch += 1
            result = sess.run(self.accuracy, feed_dict={self.input: data.testX, self.label: data.testY})
            print('Epoch', epoch, 'loss', epoch_loss, "{0:f}%".format(result * 100))

            if result >= self.ACCURACY_TARGET:
                training_target += 1
                if training_target >= 3:
                    COORD.request_stop()

class DATA(object):
    def __init__(self, para_dict):
        LABELED_PATH = para_dict['LABELED_PATH']
        TRAINING_PERCENT = para_dict['TRAINING_PERCENT']
        self.LABELED_NAME = para_dict['LABELED_NAME']
        csv_path = LABELED_PATH + self.LABELED_NAME
        self.dataframe = read_csv(csv_path, parse_dates=True)
        dataset = self.dataframe.as_matrix()
        train_size = int(len(dataset) * TRAINING_PERCENT)
        np.random.shuffle(dataset)
        self.train, self.test = dataset[0:train_size], dataset[train_size:len(dataset)]
        self.testX, self.testY = self._create_dataset(self.test)
        self.step = 0

    def _create_dataset(self, data):
        np.random.shuffle(data)
        dataX = data[:, 8:11].astype(float)
        dataY = data[:, 11]
        dataY = np.reshape(dataY, [-1, 1])
        return dataX, dataY

    def next_batch(self, batch_size):
        epoch_finished = False
        if self.step >= len(self.train):
            self.step = 0
            np.random.shuffle(self.train)
            epoch_finished = True
        batch = self.train[self.step: self.step + batch_size]
        x, y = self._create_dataset(batch)
        self.step += batch_size
        return epoch_finished, x, y

if __name__ == "__main__":
    sess = tf.Session()
    COORD = tf.train.Coordinator()

    config = Config.Configuration()
    para_dict = config.parameter_dict
    classifier = Classifier(para_dict)
    data = DATA(para_dict)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    start_time = time.time()
    threads = []
    job = lambda: classifier.train(data, COORD)
    t = threading.Thread(target=job)
    t.start()
    threads.append(t)
    COORD.join(threads)

    print "training finished!"
    print "Train time elapsed:", time.time() - start_time, "seconds"
    duration = 1  # second
    freq = 440  # Hz
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))

    model_name = 'classifier-' + para_dict['LABELED_NAME']
    path = para_dict['CLASSIFICATION_MODEL_PATH'] + model_name
    saver.save(sess, path)