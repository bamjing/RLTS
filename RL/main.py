# -*- coding: utf-8 -*-
import os
import threading
import time
from collections import deque

from Module import trading_env
import tensorflow as tf

from Module.RL.Chief import Chief
from Module.RL.Worker import Worker
from Module import Config


"""
多个worker更多的作用是多样化，探索不同的环境，带来robust的效果，而不是为了快速。
一个worker多次采样取最好值则是为了保证当前数据可以体现当前的梯度。
"""
if __name__ == "__main__":
    config = Config.Configuration()
    parameter_dict = config.parameter_dict

    SESS = tf.Session()
    COORD = tf.train.Coordinator()

    PUSH_EVENT, UPDATE_EVENT = threading.Event(), threading.Event() #set:可以， clear:停止，  wait等待s
    PUSH_EVENT.clear()
    UPDATE_EVENT.clear()

    MEMORY_DICT = {}
    workers = []
    for i in range(parameter_dict['NUM_WORKERS']):
        i_name = 'Worker_N%i' % i
        MEMORY_DICT[i_name] = deque()
        workers.append(Worker(i_name, parameter_dict, SESS, MEMORY_DICT, COORD))
    chief = Chief('Chief', parameter_dict, SESS, MEMORY_DICT, COORD, workers)
    SESS.run(tf.global_variables_initializer())
    log_writer = tf.summary.FileWriter(parameter_dict['LOG_FILE_PATH'], graph=tf.get_default_graph())

    for worker in workers:
        SESS.run([localp.assign(chiefp) for chiefp, localp in zip(chief.ppo.piNetParameters, worker.ppo.piNetParameters)])
        SESS.run([localp.assign(chiefp) for chiefp, localp in zip(chief.ppo.oldpiNetParameters, worker.ppo.oldpiNetParameters)])

    start_time = time.time()
    threads = []
    for worker in workers:
        job = lambda : worker.work(PUSH_EVENT, UPDATE_EVENT, log_writer)
        t = threading.Thread(target=job)
        t.start()
        threads.append(t)
    PUSH_EVENT.set()
    threads.append(threading.Thread(target=chief.check(PUSH_EVENT, UPDATE_EVENT)))
    threads[-1].start()
    COORD.join(threads)

    print "TRAINING FINISHED."
    print "Train time elapsed:", time.time() - start_time, "seconds"
    duration = 3  # second
    freq = 440  # Hz
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))

    env = trading_env.make(parameter_dict)
    state = env.reset()
    ep_reward = 0
    while True:
        env.render()
        action = chief.act([state])
        state_, reward, done, _ = env.step(action)
        ep_reward += reward
        state = state_
        if done:
            print ep_reward
            state = env.reset()
            ep_reward = 0