# -*- coding: utf-8 -*-
"""

birch 算法相同数据产生的结果是一样的。

从IB获取数据的格式：
           Date, Open, Close, High, Low, Count, Volume
加上K形态以后：
    Index, Date, Open, Close, High, Low, Count, Volume, US, RB, LS
加上label之后：
    Index, Date, Open, Close, High, Low, Count, Volume, US, RB, LS, Label, VOLUME_P
"""
import multiprocessing
class Configuration(object):
    def __init__(self):
        self.parameter_dict = {
            'LEARNING_RATE': 1e-4,
            'MOMENTUM': 0.99,
            'ENTCOEFF': 0.01,
            'VCOEFF': 0.5,
            'CLIP_PARAM': 0.2,
            'GAMMA': 0.7,
            'LAM': 0.3,
            'SCHEDULE': 'linear',
            'MAX_AC_EXP_RATE': 0.4,
            'MIN_AC_EXP_RATE': 0,
            #'DATA/Labeled/90D-5mins-CLV7-birch_clustering-256-labeled.csv'
            'AC_EXP_PERCENTAGE': 1,
            'UPDATE_STEPS': 1,
            'MAX_EPOCH_STEPS': 900,
            'EPOCH_MAX': 2000,
            'NUM_WORKERS': 1,  # multiprocessing.cpu_count(),
            'EVOLUTION_RATE': 0.4,
            'ENV_SAMPLE_ITERATIONS': 1,
            'INITIAL_FUNDS': 10000,
            'MAX_DRAWDOWN': 0.90,
            'STOP_LOSS': 300,
            'MAX_POSITION': 1,
            'DAY_HOURS': 23,

            'LOOK_BACK': 200,
            'WORDS_SIZE': 13,
            'NUM_FILTERS': 128,
            'CNN_FILTERS': 3,
            'DROP_RATE': 0.9,
            'RNN_SIZE': 250,
            'STATE_LENGTH': 3,
            'CNN_FILTER_SIZE': [1],

            'LOG_FILE_PATH': '/Users/YJ/Desktop/tensorFlowData/WHOLE',
            'RAW_PATH': '/Users/YJ/PycharmProjects/RLTrading/DATA/RAW/',
            'LABELED_PATH': '/Users/YJ/PycharmProjects/RLTrading/DATA/Labeled/',
            'CLASSIFICATION_MODEL_PATH': '/Users/YJ/PycharmProjects/RLTrading/PRETRAINED/Classification_model/',
            'EMBEDDING_MODEL_PATH': '/Users/YJ/PycharmProjects/RLTrading/PRETRAINED/Candle2Vec_model/',

            'RAW_NAME': '90D-5mins-CLV7',
            'LABELED_NAME': '90D-5mins-CLV7-birch_clustering-256-labeled.csv',
            'EMBEDDING_MODEL_NAME': '90D-5mins-CLV7-birch_clustering-256-labeled.csv-EMBEDDING-200-20000',
            'CLASSIFIER_MODEL_META_NAME': 'classifier-90D-5mins-CLV7-birch_clustering-256-labeled.csv.meta',
            'COLUMNS_TITLE': ['Date', 'Open', 'Close', 'High', 'Low', 'Count', 'Volume'],
            'CLUSTERING_LIST': ['fast_clustering', 'hierarchicalclustering', 'birch_clustering'],

            'BIRCH_threshold': 0.000000000001,
            'NUM_CLUSTERING': 256,
            'TRAINING_PERCENT': 0.66,
            'ACCURACY_TARGET': 0.983,
            'BATCH_SIZE': 128,
            'EMBEDDING_FEATURE_SIZE': 100,
            'WINDOW': 50,
            'WORKERS': 8,
            'BATCH_WORDS': 10000,
            'MIN_COUNT': 0,
            'EPOCHS': 20000,
            'SENTENCE_LENGTH': 10,
        }